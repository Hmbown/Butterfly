from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class KernelTargetSpec:
    id: str
    kernel_name: str
    discover_target: str
    priority: str
    question: str
    reference_path: str
    seed_kernel_path: str
    session_stub_name: str
    expected_inputs: List[str]
    expected_outputs: List[str]
    quality_gate_ref: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_TARGETS: Dict[str, KernelTargetSpec] = {
    "k1": KernelTargetSpec(
        id="k1",
        kernel_name="hcsa_permute_window_fused",
        discover_target="hcsa_permute_window",
        priority="P0",
        question="Can we fuse permute+window+attention+unpermute into one Metal kernel?",
        reference_path="hcsa/mlx/attention.py:260",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_permute_window_fused.metal",
        session_stub_name="hcsa_permute_window_session.stub.json",
        expected_inputs=[
            "q_pi [B,H,T,dh]",
            "k_pi [B,H,T,dh]",
            "v_pi [B,H,T,dh]",
            "perm [H,T]",
            "inv_perm [H,T]",
            "window int",
        ],
        expected_outputs=["out [B,H,T,dh]"],
        quality_gate_ref="Q1,Q2",
    ),
    "k2": KernelTargetSpec(
        id="k2",
        kernel_name="hcsa_sparse_gather_fused",
        discover_target="hcsa_sparse_gather",
        priority="P2",
        question="Can we fuse sparse gather attention (general path) into one Metal kernel?",
        reference_path="hcsa/mlx/attention.py:170",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_sparse_gather_fused.metal",
        session_stub_name="hcsa_sparse_gather_session.stub.json",
        expected_inputs=[
            "q [B,H,T,dh]",
            "k [B,H,T,dh]",
            "v [B,H,T,dh]",
            "neigh_idx [H,T,D] int32",
            "edge_type [H,T,D] optional",
        ],
        expected_outputs=["out [B,H,T,dh]"],
        quality_gate_ref="Q1,Q3,Q4",
    ),
    "k3": KernelTargetSpec(
        id="k3",
        kernel_name="hcsa_graph_construct",
        discover_target="hcsa_graph_construct",
        priority="P1",
        question="Can cycle permutations and inverse indices be built on GPU to remove CPU stalls?",
        reference_path="hcsa/cycles.py",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_graph_construct.metal",
        session_stub_name="hcsa_graph_construct_session.stub.json",
        expected_inputs=["T int", "H int", "seed int"],
        expected_outputs=["perm [H,T] int32", "inv_perm [H,T] int32", "window_idx [H,T,W] int32"],
        quality_gate_ref="Q5",
    ),
    "k4": KernelTargetSpec(
        id="k4",
        kernel_name="hcsa_active_row_fused",
        discover_target="hcsa_active_row",
        priority="P1",
        question="Can we support Q_len < K_len without dense fallback in chunked prefill?",
        reference_path="hcsa/mlx/attention.py:866",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_active_row_fused.metal",
        session_stub_name="hcsa_active_row_session.stub.json",
        expected_inputs=[
            "q_active [B,H,Q,dh]",
            "k_cache [B,H,K,dh]",
            "v_cache [B,H,K,dh]",
            "query_positions [Q]",
            "perm [H,K]",
            "inv_perm [H,K]",
            "window int",
        ],
        expected_outputs=["out [B,H,Q,dh]"],
        quality_gate_ref="Q1,Q2",
    ),
    "k5": KernelTargetSpec(
        id="k5",
        kernel_name="hcsa_wayfinder_ttt_fused",
        discover_target="hcsa_wayfinder_ttt",
        priority="P2",
        question="Can we fuse HCSA attention and TTT updates to improve global propagation and memory?",
        reference_path="zmlx/src/zmlx/ttt/kernel.py",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_wayfinder_ttt_fused.metal",
        session_stub_name="hcsa_wayfinder_ttt_session.stub.json",
        expected_inputs=[
            "x [B,H,T,dh]",
            "perm [H,T]",
            "W_ttt [H,F,F]",
            "window int",
            "alpha float",
            "lr float",
        ],
        expected_outputs=["out [B,H,T,dh]", "W_ttt_next [H,F,F]"],
        quality_gate_ref="Q1,Q2,Q5",
    ),
    "k6": KernelTargetSpec(
        id="k6",
        kernel_name="hcsa_fused_attention",
        discover_target="hcsa_fused_attention",
        priority="P0",
        question=(
            "Can we fuse all-head active-row permute-window attention into one Metal dispatch "
            "to eliminate Python head-chunk x query-chunk overhead?"
        ),
        reference_path="hcsa/mlx/attention.py:1294",
        seed_kernel_path="hcsa/mlx/kernels/metal/seeds/hcsa_fused_attention.metal",
        session_stub_name="hcsa_fused_attention_session.stub.json",
        expected_inputs=[
            "q [B,Hq,Q,dh]",
            "k [B,Hkv,Tk,dh]",
            "v [B,Hkv,Tk,dh]",
            "all_perms [Hq,Tg] int32",
            "all_inv_perms [Hq,Tg] int32",
            "query_positions [Q] int32",
            "window int",
        ],
        expected_outputs=["out [B,Hq,Q,dh]"],
        quality_gate_ref="Q1,Q2",
    ),
}

_ALIASES = {
    "all": "all",
    "hcsa_permute_window_fused": "k1",
    "hcsa_sparse_gather_fused": "k2",
    "hcsa_graph_construct": "k3",
    "hcsa_active_row_fused": "k4",
    "hcsa_wayfinder_ttt_fused": "k5",
    "hcsa_fused_attention": "k6",
}


def list_targets() -> Dict[str, KernelTargetSpec]:
    return dict(_TARGETS)


def get_target(name_or_id: str) -> KernelTargetSpec:
    key = name_or_id.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _TARGETS:
        available = ", ".join(sorted(_TARGETS))
        raise KeyError(f"Unknown target '{name_or_id}'. Available ids: {available}")
    return _TARGETS[key]


def resolve_targets(names: Iterable[str] | None) -> List[KernelTargetSpec]:
    if names is None:
        names = ["all"]

    selected = [x.strip() for x in names if x.strip()]
    if not selected or any(x.lower() == "all" for x in selected):
        return [*_TARGETS.values()]

    out: List[KernelTargetSpec] = []
    seen = set()
    for name in selected:
        spec = get_target(name)
        if spec.id in seen:
            continue
        seen.add(spec.id)
        out.append(spec)
    return out
