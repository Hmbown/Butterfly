"""Auto-generated kernel from ZMLX Discover.

Target: hcsa_permute_window
Speedup: 1.977x
Device: Apple M4
Session: real claude-code search (10 steps, 81 candidates)
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel() -> Any:
    """Build the discovered kernel."""
    source = """// 1.977x speedup — 2-element vectorization per thread
constexpr uint SEQ = 256;
constexpr uint D = 128;
constexpr uint W = 65;
constexpr uint ELEMS_PER_THREAD = 2;

uint t_idx = thread_position_in_grid.y;
uint d_base = thread_position_in_grid.x * ELEMS_PER_THREAD;
if (t_idx >= SEQ || d_base >= D) return;

float acc0 = 0.0f, acc1 = 0.0f;
for (uint j = 0; j < W; ++j) {
    uint src = (uint)neighbor_idx[t_idx * W + j];
    if (src >= SEQ) src = 0;
    uint offset = src * D + d_base;
    acc0 += (float)inp[offset];
    if (d_base + 1 < D) acc1 += (float)inp[offset + 1];
}
out[t_idx * D + d_base] = (T)(acc0 / (float)W);
if (d_base + 1 < D) out[t_idx * D + d_base + 1] = (T)(acc1 / (float)W);"""

    return metal_kernel(
        name="kk_discovered_hcsa_permute_window",
        input_names=['inp', 'neighbor_idx'],
        output_names=['out'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )
