from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hcsa.graph.abi import WayfinderGraphABI


def specialize_perm_window_pass(
    abi: WayfinderGraphABI,
    *,
    window: int,
) -> Dict[str, Any]:
    cycle_perms = abi.meta.get("cycle_perms")
    if not isinstance(cycle_perms, list):
        raise ValueError("Graph ABI meta must include cycle_perms list")

    H, T, _D = abi.neigh_idx.shape
    W = 2 * int(window) + 1

    perms = np.zeros((H, T), dtype=np.int32)
    inv_perms = np.zeros((H, T), dtype=np.int32)
    window_idx = np.zeros((H, T, W), dtype=np.int32)
    valid_mask = np.zeros((H, T, W), dtype=np.bool_)

    offsets = np.arange(-window, window + 1, dtype=np.int32)
    base = np.arange(T, dtype=np.int32)[:, None] + offsets[None, :]
    base_valid = (base >= 0) & (base < T)
    base_clamped = np.clip(base, 0, T - 1)

    for h in range(H):
        perm = cycle_perms[h]
        if perm is None:
            raise ValueError(f"Missing cycle permutation for head {h}")
        perm_arr = np.asarray(perm, dtype=np.int32)
        if perm_arr.shape != (T,):
            raise ValueError(f"Head {h} cycle_perm must be shape ({T},), got {perm_arr.shape}")
        perms[h] = perm_arr
        inv_perms[h] = np.argsort(perm_arr)
        window_idx[h] = base_clamped
        valid_mask[h] = base_valid

    return {
        "perm": perms,
        "inv_perm": inv_perms,
        "window_idx": window_idx,
        "valid_mask": valid_mask,
    }
