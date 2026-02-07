from __future__ import annotations

import pytest

from hcsa.compiler.graph_ir import GraphIR, ScheduleSpec
from hcsa.compiler.passes.normalize_pass import normalize_pass
from hcsa.compiler.passes.validate_pass import validate_pass
from hcsa.compiler.sexp import parse_graph_ir, parse_sexp


def test_parse_default_graph_ir() -> None:
    src = """
    (wayfinder
      (degree 64)
      (backbone (cycle :type random :seed 42 :k 2))
      (local (window :size (schedule :linear 64 16 :steps 2000)))
      (highways (landmarks :stride 64))
      (permute_window :enabled true :window 32))
    """
    ir = parse_graph_ir(parse_sexp(src))
    assert ir.degree == 64
    assert ir.num_cycles == 2
    assert ir.window_schedule is not None
    assert ir.window_schedule.kind == "linear"


def test_normalize_uses_window_schedule_start() -> None:
    ir = GraphIR(window_size=8, window_schedule=ScheduleSpec(kind="linear", start=64, end=16, steps=100))
    out = normalize_pass(ir)
    assert out.window_size == 64


def test_validate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        validate_pass(GraphIR(degree=0))
    with pytest.raises(ValueError):
        validate_pass(GraphIR(num_cycles=0))
    with pytest.raises(ValueError):
        validate_pass(GraphIR(window_size=-1))
    with pytest.raises(ValueError):
        validate_pass(GraphIR(strategy="unknown"))
