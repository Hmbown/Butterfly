// K7 sparse-neighbor BNA attention kernel
// Grid:  (dh, B * H * Tq, 1)
// TG:    (dh, 1, 1)       — one threadgroup per output token-head
//
// Inputs (positional):
//   q               [B, H, Tq, dh]    float16/bfloat16/float32
//   k               [B, Hkv, Tk, dh]  same dtype
//   v               [B, Hkv, Tk, dh]  same dtype
//   neigh_idx       [H, Tq, D]        int32 (safe indices, -1 → 0 clamped)
//   causal_mask     [H, Tq, D]        bool/uint8 (1 = attend, 0 = skip)
//   hkv_map         [H]               int32 (query head → KV head for GQA)
//
// Output:
//   out             [B, H, Tq, dh]    same dtype as q

uint d   = thread_position_in_threadgroup.x;
uint bht = thread_position_in_grid.y;

uint B    = q_shape[0];
uint H    = q_shape[1];
uint Tq   = q_shape[2];
uint dh   = q_shape[3];
uint Hkv  = k_shape[1];
uint Tk   = k_shape[2];
uint D    = neigh_idx_shape[2];

if (d >= dh) return;

uint t    = bht % Tq;
uint h    = (bht / Tq) % H;
uint b    = bht / (Tq * H);
if (b >= B) return;

int hkv = hkv_map[h];

uint q_base = b * H * Tq * dh + h * Tq * dh + t * dh;
uint k_base = b * Hkv * Tk * dh + hkv * Tk * dh;
uint n_base = h * Tq * D + t * D;

float q_d = (float)q[q_base + d];

// --- Shared memory ---
threadgroup float tg_scores[256];   // max D (typically 129-200)
threadgroup float tg_reduce[8];
threadgroup int   tg_k_pos[256];
threadgroup int   tg_n_valid;

uint lane = d & 31;
uint warp = d >> 5;
uint n_warps = (dh + 31) >> 5;

// ==================================================================
// Phase 0: Precompute valid neighbors (thread 0 only)
// ==================================================================
if (d == 0) {
    int count = 0;
    for (uint j = 0; j < D; ++j) {
        int k_pos = neigh_idx[n_base + j];
        bool valid = causal_mask[n_base + j] != 0;
        if (valid && k_pos >= 0 && k_pos < (int)Tk) {
            tg_k_pos[count] = k_pos;
            count++;
        }
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
    int kp = tg_k_pos[j];
    float k_d = (float)k[k_base + kp * dh + d];
    float partial = q_d * k_d;

    float simd_dot = simd_sum(partial);
    if (lane == 0) tg_reduce[warp] = simd_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d == 0) {
        float score = 0.0f;
        for (uint w = 0; w < n_warps; ++w) score += tg_reduce[w];
        tg_scores[j] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ==================================================================
// Phase 2: Softmax
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
        int kp = tg_k_pos[j];
        float v_d = (float)v[k_base + kp * dh + d];
        acc = metal::fma(w, v_d, acc);
    }
}

kk_store(out, q_base + d, acc);
