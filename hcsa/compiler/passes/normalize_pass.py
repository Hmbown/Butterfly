from __future__ import annotations

from dataclasses import replace

from hcsa.compiler.graph_ir import GraphIR


def normalize_pass(ir: GraphIR) -> GraphIR:
    out = ir

    if out.permute_window_enabled and out.permute_window_size <= 0:
        out = replace(out, permute_window_size=max(8, out.window_size))

    if out.window_schedule is not None:
        out = replace(out, window_size=int(out.window_schedule.start))

    if out.landmark_stride is not None and out.landmark_stride == 0:
        out = replace(out, landmark_stride=None)

    return out
