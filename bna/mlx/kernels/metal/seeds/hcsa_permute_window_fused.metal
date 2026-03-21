// Setup-only seed for hcsa_permute_window_fused
// This file is a scaffold for future ZMLX Discover runs.
// It is intentionally non-executable until a full kernel implementation is discovered.

#include <metal_stdlib>
using namespace metal;

kernel void hcsa_permute_window_fused(
    device const half* in0 [[buffer(0)]],
    device half* out0 [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // TODO: replace with discovered implementation.
    out0[gid] = in0[gid];
}
