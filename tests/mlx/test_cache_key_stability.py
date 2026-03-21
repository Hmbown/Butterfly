from __future__ import annotations

from bna.compiler.graph_ir import GraphIR
from bna.compiler.passes.cache_key_pass import cache_key_pass


def test_cache_key_is_stable_for_same_inputs() -> None:
    ir = GraphIR()
    a = cache_key_pass(ir, T=2048, H=4, dtype="float16")
    b = cache_key_pass(ir, T=2048, H=4, dtype="float16")
    assert a["hash"] == b["hash"]


def test_cache_key_changes_with_shape_or_dtype() -> None:
    ir = GraphIR()
    base = cache_key_pass(ir, T=2048, H=4, dtype="float16")
    t_changed = cache_key_pass(ir, T=1024, H=4, dtype="float16")
    h_changed = cache_key_pass(ir, T=2048, H=8, dtype="float16")
    dtype_changed = cache_key_pass(ir, T=2048, H=4, dtype="float32")

    assert base["hash"] != t_changed["hash"]
    assert base["hash"] != h_changed["hash"]
    assert base["hash"] != dtype_changed["hash"]
