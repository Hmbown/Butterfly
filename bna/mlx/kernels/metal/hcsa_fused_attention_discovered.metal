// K6 fused BNA attention kernel — permute + window + causal SDPA + inv-permute
// Grid:  (dh, B * Hq * Qblk, 1)
// TG:    (dh, 1, 1)       — one threadgroup per output token-head
//
// Inputs (positional):
//   q               [B, Hq, Qblk, dh]   float16/bfloat16/float32
//   k               [B, Hkv, Tk, dh]     same dtype
//   v               [B, Hkv, Tk, dh]     same dtype
//   all_perms       [Hq, Tg]             int32
//   all_inv_perms   [Hq, Tg]             int32
//   query_positions [Qblk]               int32
//   window          [1]                  int32
//
// Output:
//   out             [B, Hq, Qblk, dh]    same dtype as q

// --- Thread mapping ---
uint d   = thread_position_in_threadgroup.x;   // dimension [0, dh)
uint bhq = thread_position_in_grid.y;          // flat (b, h, q) index

// --- Decode shapes from input metadata ---
uint B    = q_shape[0];
uint Hq   = q_shape[1];
uint Qblk = q_shape[2];
uint dh   = q_shape[3];
uint Hkv  = k_shape[1];
uint Tk   = k_shape[2];
uint Tg   = all_perms_shape[1];
int  W    = window[0];

if (d >= dh) return;

// --- Decode (b, h, q_idx) from flat index ---
uint q_idx = bhq % Qblk;
uint h     = (bhq / Qblk) % Hq;
uint b     = bhq / (Qblk * Hq);
if (b >= B) return;

// --- GQA: map query head → KV head ---
uint hkv = h / (Hq / Hkv);

// --- Permutation lookups ---
int q_orig = query_positions[q_idx];
int q_perm = all_inv_perms[h * Tg + q_orig];  // cycle-order position

// --- Offsets for contiguous layout ---
uint q_base = b * Hq * Qblk * dh + h * Qblk * dh + q_idx * dh;
uint k_base = b * Hkv * Tk * dh + hkv * Tk * dh;
uint v_base = k_base;  // same layout

float q_d = (float)q[q_base + d];

// --- Shared memory ---
// 129 = max window degree 2*64+1
threadgroup float tg_scores[129];
threadgroup float tg_reduce[8];     // cross-warp reduction
threadgroup int   tg_k_orig[129];   // cached original positions
threadgroup int   tg_n_valid;

uint lane = d & 31;
uint warp = d >> 5;
uint n_warps = (dh + 31) >> 5;

// ==================================================================
// Phase 0: Precompute valid neighbor positions (thread 0 only)
// ==================================================================
if (d == 0) {
    int count = 0;
    for (int j = -W; j <= W; ++j) {
        int kp = q_perm + j;
        if (kp < 0 || kp >= (int)Tg) continue;
        int ko = all_perms[h * Tg + kp];
        if (ko > q_orig) continue;   // causality
        if (ko >= (int)Tk) continue;  // bounds
        tg_k_orig[count] = ko;
        count++;
    }
    tg_n_valid = count;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

int n_valid = tg_n_valid;
if (n_valid == 0) {
    kk_store(out, q_base + d, 0.0f);
    return;
}

// ==================================================================
// Phase 1: Compute attention scores via SIMD-reduced dot products
// ==================================================================
float scale = rsqrt((float)dh);

for (int j = 0; j < n_valid; ++j) {
    int ko = tg_k_orig[j];
    float k_d = (float)k[k_base + ko * dh + d];
    float partial = q_d * k_d;

    // SIMD reduction within each 32-thread group
    float simd_dot = simd_sum(partial);
    if (lane == 0) tg_reduce[warp] = simd_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 sums across warps
    if (d == 0) {
        float score = 0.0f;
        for (uint w = 0; w < n_warps; ++w) score += tg_reduce[w];
        tg_scores[j] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ==================================================================
// Phase 2: Softmax (thread 0, broadcast via shared memory)
// ==================================================================
if (d == 0) {
    float max_s = -1e30f;
    for (int j = 0; j < n_valid; ++j)
        max_s = metal::max(max_s, tg_scores[j]);

    float sum_exp = 0.0f;
    for (int j = 0; j < n_valid; ++j) {
        tg_scores[j] = metal::exp(tg_scores[j] - max_s);
        sum_exp += tg_scores[j];
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int j = 0; j < n_valid; ++j)
        tg_scores[j] *= inv_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ==================================================================
// Phase 3: Weighted V accumulation
// ==================================================================
float acc = 0.0f;
for (int j = 0; j < n_valid; ++j) {
    float w = tg_scores[j];
    if (w > 0.0f) {
        int ko = tg_k_orig[j];
        float v_d = (float)v[v_base + ko * dh + d];
        acc = metal::fma(w, v_d, acc);
    }
}

kk_store(out, q_base + d, acc);
