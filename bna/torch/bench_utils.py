from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from bna.graph.abi import EdgeType, WayfinderGraphABI, graph_metrics, validate_graph_abi


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query K/V heads to query-head count."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


@dataclass
class AttentionProfile:
    graph_build_ms: float = 0.0
    permute_ms: float = 0.0
    attention_ms: float = 0.0
    total_ms: float = 0.0
    path: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "graph_build_ms": float(self.graph_build_ms),
            "permute_ms": float(self.permute_ms),
            "attention_ms": float(self.attention_ms),
            "total_ms": float(self.total_ms),
            "path": self.path,
        }
        out.update(self.notes)
        return out


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def safe_neighbor_idx(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    return neigh_idx.clamp(min=0, max=max(0, int(seq_len) - 1))


def causal_neighbor_mask(neigh_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    valid = neigh_idx >= 0
    safe = safe_neighbor_idx(neigh_idx, seq_len)
    if neigh_idx.ndim == 3:
        i_idx = torch.arange(seq_len, device=neigh_idx.device).view(1, seq_len, 1)
    elif neigh_idx.ndim == 4:
        i_idx = torch.arange(seq_len, device=neigh_idx.device).view(1, 1, seq_len, 1)
    else:
        raise ValueError(f"Expected neigh_idx ndim 3 or 4, got {neigh_idx.ndim}")
    return valid & (safe <= i_idx)


def stable_masked_softmax(scores_f32: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable masked softmax that returns zeros on all-masked rows."""
    if scores_f32.dtype != torch.float32:
        raise ValueError("scores_f32 must be float32")

    neg_inf = torch.tensor(-1e30, dtype=torch.float32, device=scores_f32.device)
    masked = torch.where(mask, scores_f32, neg_inf)
    row_max = masked.max(dim=dim, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    expv = torch.exp(masked - row_max)
    expv = torch.where(mask, expv, torch.zeros_like(expv))

    denom = expv.sum(dim=dim, keepdim=True)
    safe_denom = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    out = expv / safe_denom
    return torch.where(denom > 0.0, out, torch.zeros_like(out))


def normalize_graph_tensors(
    neigh_idx: torch.Tensor | np.ndarray,
    edge_type: torch.Tensor | np.ndarray | None,
    *,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize graph tensors to [B,H,T,D] with padded -1 neighbor convention."""
    neigh = torch.as_tensor(neigh_idx, device=device, dtype=torch.long)
    if edge_type is None:
        edge = torch.where(
            neigh >= 0,
            torch.full_like(neigh, int(EdgeType.WINDOW), dtype=torch.long),
            torch.zeros_like(neigh, dtype=torch.long),
        ).to(dtype=torch.uint8)
    else:
        edge = torch.as_tensor(edge_type, device=device, dtype=torch.uint8)

    if neigh.shape != edge.shape:
        raise ValueError(f"neigh_idx and edge_type shape mismatch: {neigh.shape} vs {edge.shape}")

    if neigh.ndim == 2:
        if neigh.shape[0] != seq_len:
            raise ValueError(f"Expected [T,D] with T={seq_len}, got {neigh.shape}")
        neigh = neigh.unsqueeze(0).unsqueeze(0)
        edge = edge.unsqueeze(0).unsqueeze(0)
    elif neigh.ndim == 3:
        if neigh.shape[0] != n_heads or neigh.shape[1] != seq_len:
            raise ValueError(f"Expected [H,T,D]=[{n_heads},{seq_len},D], got {neigh.shape}")
        neigh = neigh.unsqueeze(0)
        edge = edge.unsqueeze(0)
    elif neigh.ndim == 4:
        if neigh.shape[0] not in (1, batch_size) or neigh.shape[1] != n_heads or neigh.shape[2] != seq_len:
            raise ValueError(
                f"Expected [B,H,T,D]=[{batch_size},{n_heads},{seq_len},D], got {neigh.shape}"
            )
    else:
        raise ValueError(f"Expected neigh_idx ndim in {{2,3,4}}, got {neigh.ndim}")

    if neigh.shape[0] == 1 and batch_size > 1:
        neigh = neigh.expand(batch_size, -1, -1, -1)
        edge = edge.expand(batch_size, -1, -1, -1)

    return neigh.contiguous(), edge.contiguous()


def wayfinder_abi_to_torch_graph(
    abi: WayfinderGraphABI,
    *,
    device: torch.device,
    batch_size: int,
    n_heads: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_graph_abi(abi, expect_heads=n_heads, expect_tokens=seq_len, enforce_hamiltonian=True)
    return normalize_graph_tensors(
        np.asarray(abi.neigh_idx, dtype=np.int64),
        np.asarray(abi.edge_type, dtype=np.uint8),
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len=seq_len,
        device=device,
    )


def load_compiled_graph_abi(
    compiled_graph_dir: str | Path,
    *,
    n_heads: int,
    seq_len: int,
) -> WayfinderGraphABI | None:
    art_dir = Path(compiled_graph_dir)
    ni_path = art_dir / "neighborindex.npz"
    meta_path = art_dir / "meta.json"
    if not ni_path.exists():
        return None

    payload = np.load(ni_path)
    neigh_idx = np.asarray(payload["neigh_idx"], dtype=np.int32)
    edge_type = np.asarray(payload["edge_type"], dtype=np.uint8)

    if neigh_idx.shape != edge_type.shape:
        return None

    if neigh_idx.ndim == 2:
        neigh_idx = np.broadcast_to(neigh_idx[None, :, :], (n_heads, *neigh_idx.shape)).copy()
        edge_type = np.broadcast_to(edge_type[None, :, :], (n_heads, *edge_type.shape)).copy()

    if neigh_idx.ndim != 3:
        return None

    if int(neigh_idx.shape[0]) != n_heads or int(neigh_idx.shape[1]) != seq_len:
        return None

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    meta.setdefault("cycle_perms", [])
    meta.setdefault("max_degree", int(neigh_idx.shape[-1]))
    meta.setdefault("seq_len", int(seq_len))
    meta.setdefault("n_heads", int(n_heads))

    abi = WayfinderGraphABI(neigh_idx=neigh_idx, edge_type=edge_type, meta=meta)
    validate_graph_abi(abi, expect_heads=n_heads, expect_tokens=seq_len, enforce_hamiltonian=True)
    return abi


def edge_utilization_by_type(attn_weights: np.ndarray, edge_type: np.ndarray) -> Dict[str, float]:
    """Aggregate attention mass by edge type; mirrors the MLX metric schema."""
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be [B,H,T,D], got {attn_weights.shape}")

    if edge_type.ndim == 4:
        edge_h = edge_type[0]
    elif edge_type.ndim == 3:
        edge_h = edge_type
    else:
        raise ValueError(f"edge_type must be [H,T,D] or [B,H,T,D], got {edge_type.shape}")

    if attn_weights.shape[1:] != edge_h.shape:
        raise ValueError(
            f"Shape mismatch: weights {attn_weights.shape} vs edge_type {edge_h.shape}"
        )

    total = float(attn_weights.sum())
    if total <= 0:
        return {k: 0.0 for k in ["cycle", "window", "landmark", "rewire"]}

    mapping = {
        "cycle": int(EdgeType.CYCLE),
        "window": int(EdgeType.WINDOW),
        "landmark": int(EdgeType.LANDMARK),
        "rewire": int(EdgeType.REWIRE),
    }

    out: Dict[str, float] = {}
    for name, code in mapping.items():
        mask = (edge_h == code)[None, ...]
        mass = float((attn_weights * mask).sum())
        out[name] = mass / total
    return out


def graph_metrics_from_tensor_graph(
    neigh_idx: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    bfs_hops: int = 4,
) -> Dict[str, Any]:
    neigh_np = neigh_idx.detach().cpu().numpy()
    edge_np = edge_type.detach().cpu().numpy()

    if neigh_np.ndim == 4:
        neigh_np = neigh_np[0]
        edge_np = edge_np[0]

    abi = WayfinderGraphABI(
        neigh_idx=neigh_np.astype(np.int32),
        edge_type=edge_np.astype(np.uint8),
        meta={},
    )
    return graph_metrics(abi, bfs_hops=bfs_hops)


def largest_intermediate_bytes(
    *,
    B: int,
    H: int,
    T: int,
    D: int,
    dh: int,
    path: str,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """Best-effort tensor size proxy for benchmark reporting."""
    if path == "dense":
        scores = B * H * T * T * 4
        weights = B * H * T * T * 4
        return {
            "scores_fp32": int(scores),
            "weights_fp32": int(weights),
            "largest": int(max(scores, weights)),
        }

    gather = B * H * T * D * dh * dtype_bytes
    scores = B * H * T * D * 4
    weights = B * H * T * D * 4
    return {
        "kv_gather": int(gather),
        "scores_fp32": int(scores),
        "weights_fp32": int(weights),
        "largest": int(max(gather, scores, weights)),
    }
