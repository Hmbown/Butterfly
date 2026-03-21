// Auto-exported from zmlx.discover session hcsa_active_row
// Session: 8c7e5652b2ad46a2
// Speedup: 3.987x over naive baseline
// Device: Apple M4
// Optimizations: hierarchical SIMD reduction, shared memory, FMA
constexpr uint D = 128;
constexpr uint W = 65;
uint d_idx = thread_position_in_grid.x;
if (d_idx >= D) return;

threadgroup float shared_weights[W];
threadgroup float shared_reduce[8];  // 4 warps worth of reduction space
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31;
uint warp = tid >> 5;

// Hierarchical max: SIMD -> threadgroup
float local_max = (tid < W) ? (float)scores[tid] : -INFINITY;
if (tid + D < W) local_max = metal::max(local_max, (float)scores[tid + D]);
float warp_max = simd_max(local_max);
if (lane == 0) shared_reduce[warp] = warp_max;
threadgroup_barrier(mem_flags::mem_threadgroup);

float max_s = shared_reduce[0];
for (uint i = 1; i < 4; ++i) max_s = metal::max(max_s, shared_reduce[i]);

// Exp computation
if (tid < W) {
    shared_weights[tid] = metal::exp((float)scores[tid] - max_s);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Hierarchical sum for denom: SIMD -> threadgroup
float local_sum = 0.0f;
for (uint j = tid; j < W; j += D) {
    local_sum += shared_weights[j];
}
float warp_sum = simd_sum(local_sum);
if (lane == 0) shared_reduce[warp] = warp_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

float denom = shared_reduce[0] + shared_reduce[1] + shared_reduce[2] + shared_reduce[3];

// Accumulation
float acc = 0.0f;
for (uint j = 0; j < W; ++j) {
    float w = shared_weights[j];
    float v = (float)values[j * D + d_idx];
    acc = metal::fma(w, v, acc);
}

out[d_idx] = (T)(acc / denom);
