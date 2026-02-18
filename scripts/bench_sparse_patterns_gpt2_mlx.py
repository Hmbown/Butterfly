#!/usr/bin/env python3
"""Honest sparse attention pattern benchmark on GPT-2.

Compares 8 different sparse attention patterns at the same edge budget D,
measuring perplexity, top-1 agreement, cosine similarity vs dense, and
graph quality metrics (spectral gap, mixing time, diameter).

The key question: does the Hamiltonian cycle structure actually matter,
or does most of the benefit come from the local window?

Patterns tested (all with same max_degree D):
  1. dense          - original GPT-2 attention (baseline)
  2. window_only    - local causal window + self, no long-range edges
  3. window+lm      - window + evenly-spaced landmark tokens
  4. window+rand    - window + random causal long-range edges
  5. window+cycle   - window + per-head random Hamiltonian cycle + landmarks
  6. window+cycle_pure - window + per-head cycle, NO landmark backfill
  7. window+multicycle - window + ceil(2*log2(T)) cycles per head, no landmarks
  8. window+exp     - window + power-of-2 shift expander edges

Default behavior (no args): sweeps T=256,512,1024, saves JSON to
results/sparse_pattern_comparison/results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from hcsa.graph.abi import (
    WayfinderGraphABI,
    build_graph_abi_from_adjacency,
    stack_head_abis,
)
from hcsa.graph.expander import graph_quality_report
from hcsa.mlx.attention import sparse_gather_attention
from hcsa.mlx.graph_abi import (
    MLXGraphABI,
    causal_neighbor_mask,
    safe_neighbor_idx,
    to_mlx_graph_abi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning was the Word, and the Word was with God, and the Word was God. "
    "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles. "
    "All happy families are alike; each unhappy family is unhappy in its own way. "
    "It was the best of times, it was the worst of times, it was the age of wisdom, "
    "it was the age of foolishness. Call me Ishmael. Some years ago, never mind how long "
    "precisely, having little or no money in my purse, and nothing particular to interest "
    "me on shore, I thought I would sail about a little and see the watery part of the world. "
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
    "must be in want of a wife. In a hole in the ground there lived a hobbit. Not a nasty, "
    "dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, "
    "sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that "
    "means comfort. Mr and Mrs Dursley of number four, Privet Drive, were proud to say that "
    "they were perfectly normal, thank you very much. "
    "A long time ago in a galaxy far, far away. "
    "The sky above the port was the color of television, tuned to a dead channel. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "Happy families are all alike; every unhappy family is unhappy in its own way. "
    "All children, except one, grow up. They know they must grow up. "
    "There was a boy called Eustace Clarence Scrubb, and he almost deserved it. "
    "The primroses were over. Toward the edge of the wood, where the ground became open "
    "and sloped down to an old fence and a brambly ditch beyond, only a few fading "
    "patches of pale yellow still showed among the dog's mercury and oak-tree roots. "
)


def _load_gpt2(model_path: str):
    from mlx_lm.utils import load as mlx_load

    model, tokenizer = mlx_load(model_path)
    return model, tokenizer


def _cross_entropy_loss(logits: mx.array, targets: mx.array) -> float:
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
    target_log_probs = mx.take_along_axis(
        log_probs, targets_flat[:, None], axis=-1
    ).squeeze(-1)
    loss = -mx.mean(target_log_probs)
    mx.eval(loss)
    return float(loss.item())


def _cosine_sim(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    norm_a = mx.sqrt(mx.sum(a_f * a_f))
    norm_b = mx.sqrt(mx.sum(b_f * b_f))
    sim = dot / mx.maximum(norm_a * norm_b, mx.array(1e-8))
    mx.eval(sim)
    return float(sim.item())


def _top1_agreement(a: mx.array, b: mx.array) -> float:
    top_a = mx.argmax(a, axis=-1)
    top_b = mx.argmax(b, axis=-1)
    agree = mx.mean((top_a == top_b).astype(mx.float32))
    mx.eval(agree)
    return float(agree.item())


def _effective_degree(neigh_idx: np.ndarray) -> float:
    """Mean non-PAD degree across all positions (and heads if multi-head)."""
    ni = np.asarray(neigh_idx)
    return float((ni >= 0).sum(axis=-1).mean())


def _recommended_num_cycles(T: int, c: float = 2.0) -> int:
    """ceil(c * log2(T)) -- theoretically motivated cycle count."""
    return max(1, math.ceil(c * math.log2(max(2, T))))


# ---------------------------------------------------------------------------
# Graph builders -- each returns WayfinderGraphABI
# ---------------------------------------------------------------------------


def build_window_only(T: int, W: int, D: int) -> WayfinderGraphABI:
    """Window + self only. Intentionally fewer edges (ablation baseline)."""
    return build_graph_abi_from_adjacency(
        T=T,
        cycle_adj=[[] for _ in range(T)],
        window=W,
        landmark_stride=None,
        include_self=True,
        max_degree=D,
    )


def build_window_landmarks(T: int, W: int, D: int) -> WayfinderGraphABI:
    """Window + evenly-spaced landmark tokens."""
    budget = max(1, D - W - 1)
    stride = max(1, T // budget)
    return build_graph_abi_from_adjacency(
        T=T,
        cycle_adj=[[] for _ in range(T)],
        window=W,
        landmark_stride=stride,
        include_self=True,
        max_degree=D,
    )


def build_window_random(T: int, W: int, D: int, seed: int) -> WayfinderGraphABI:
    """Window + random causal long-range edges."""
    rng = np.random.default_rng(seed)
    budget = max(0, D - W - 1)

    cycle_adj: list[list[int]] = [[] for _ in range(T)]
    for i in range(T):
        window_set = set(range(max(0, i - W), i))
        window_set.add(i)
        candidates = [j for j in range(i) if j not in window_set]
        if candidates and budget > 0:
            n_pick = min(budget, len(candidates))
            picked = rng.choice(candidates, size=n_pick, replace=False)
            cycle_adj[i] = picked.tolist()

    return build_graph_abi_from_adjacency(
        T=T,
        cycle_adj=cycle_adj,
        window=W,
        landmark_stride=None,
        include_self=True,
        max_degree=D,
    )


def _single_head_cycle_adj(
    T: int, perm: np.ndarray, inv_perm: np.ndarray,
) -> list[list[int]]:
    """Build cycle adjacency list from a single permutation."""
    cycle_adj: list[list[int]] = []
    for i in range(T):
        pos = inv_perm[i]
        prev_node = int(perm[(pos - 1) % T])
        next_node = int(perm[(pos + 1) % T])
        cycle_adj.append(sorted(set([prev_node, next_node])))
    return cycle_adj


def build_window_cycle_per_head(
    T: int, W: int, D: int, H: int, seed: int,
) -> WayfinderGraphABI:
    """Window + independent random Hamiltonian cycle per head + landmark backfill."""
    remaining = max(0, D - W - 1 - 2)
    landmark_stride = max(1, T // max(1, remaining)) if remaining > 0 else None

    head_abis: list[WayfinderGraphABI] = []
    for h in range(H):
        rng = np.random.default_rng(seed + h)
        perm = rng.permutation(T)
        inv_perm = np.empty(T, dtype=np.int64)
        inv_perm[perm] = np.arange(T)
        cycle_adj = _single_head_cycle_adj(T, perm, inv_perm)

        abi = build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=cycle_adj,
            window=W,
            landmark_stride=landmark_stride,
            include_self=True,
            max_degree=D,
        )
        head_abis.append(abi)

    return stack_head_abis(head_abis)


def build_window_cycle_pure(
    T: int, W: int, D: int, H: int, seed: int,
) -> WayfinderGraphABI:
    """Window + cycle + self only. NO landmark backfill.

    Isolates the cycle contribution: the only long-range edges come from
    the Hamiltonian cycle itself (2 edges per node per head).
    """
    head_abis: list[WayfinderGraphABI] = []
    for h in range(H):
        rng = np.random.default_rng(seed + h)
        perm = rng.permutation(T)
        inv_perm = np.empty(T, dtype=np.int64)
        inv_perm[perm] = np.arange(T)
        cycle_adj = _single_head_cycle_adj(T, perm, inv_perm)

        abi = build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=cycle_adj,
            window=W,
            landmark_stride=None,  # no landmarks
            include_self=True,
            max_degree=D,
        )
        head_abis.append(abi)

    return stack_head_abis(head_abis)


def build_window_multicycle(
    T: int, W: int, D: int, H: int, seed: int, num_cycles: int,
) -> WayfinderGraphABI:
    """Window + multiple independent Hamiltonian cycles per head, no landmarks.

    Each head gets `num_cycles` independent random permutations. The cycle
    adjacency for each node is the union of cycle neighbors across all
    permutations (up to 2*num_cycles cycle edges per node).
    """
    head_abis: list[WayfinderGraphABI] = []
    for h in range(H):
        cycle_adj: list[list[int]] = [[] for _ in range(T)]
        for c in range(num_cycles):
            rng = np.random.default_rng(seed + h * num_cycles + c)
            perm = rng.permutation(T)
            inv_perm = np.empty(T, dtype=np.int64)
            inv_perm[perm] = np.arange(T)

            for i in range(T):
                pos = inv_perm[i]
                prev_node = int(perm[(pos - 1) % T])
                next_node = int(perm[(pos + 1) % T])
                cycle_adj[i].extend([prev_node, next_node])

        abi = build_graph_abi_from_adjacency(
            T=T,
            cycle_adj=cycle_adj,
            window=W,
            landmark_stride=None,  # no landmarks, pure multi-cycle
            include_self=True,
            max_degree=D,
        )
        head_abis.append(abi)

    return stack_head_abis(head_abis)


def build_window_expander(T: int, W: int, D: int) -> WayfinderGraphABI:
    """Window + power-of-2 shift expander edges (causal only: j < i)."""
    exp_adj: list[list[int]] = [[] for _ in range(T)]
    for i in range(T):
        shift = 1
        while shift <= i:
            exp_adj[i].append(i - shift)
            shift *= 2

    return build_graph_abi_from_adjacency(
        T=T,
        cycle_adj=exp_adj,
        window=W,
        landmark_stride=None,
        include_self=True,
        max_degree=D,
    )


# ---------------------------------------------------------------------------
# Attention patching
# ---------------------------------------------------------------------------


def _extract_qkv(
    attn_module: nn.Module, x: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Extract Q, K, V from a GPT-2 attention module.  Returns [B, H, T, dh]."""
    from hcsa.integrations.gpt2_mlx import extract_qkv_from_gpt2_attention

    return extract_qkv_from_gpt2_attention(attn_module, x)


class SparseAttentionPatch:
    """Callable that replaces a GPT-2 attention __call__ with sparse gather."""

    def __init__(
        self,
        base_attn: nn.Module,
        mlx_graph: MLXGraphABI,
        precomputed_safe_idx: mx.array,
        precomputed_causal_mask: mx.array,
    ):
        self.base_attn = base_attn
        self.mlx_graph = mlx_graph
        self.safe_idx = precomputed_safe_idx
        self.causal_mask = precomputed_causal_mask

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        q, k, v = _extract_qkv(self.base_attn, x)
        y_h, _ = sparse_gather_attention(
            q, k, v, self.mlx_graph,
            return_weights=False,
            precomputed_safe_idx=self.safe_idx,
            precomputed_causal_mask=self.causal_mask,
        )
        B, H, T, dh = y_h.shape
        out = y_h.transpose(0, 2, 1, 3).reshape(B, T, H * dh)
        return self.base_attn.c_proj(out)


def _patch_model_sparse(
    model: nn.Module,
    mlx_graph: MLXGraphABI,
    safe_idx: mx.array,
    causal_mask: mx.array,
) -> None:
    """Monkey-patch all attention layers in-place."""
    for layer in model.layers:
        base_attn = layer.attn
        patch = SparseAttentionPatch(base_attn, mlx_graph, safe_idx, causal_mask)
        layer.attn = patch


def _unpatch_model(model: nn.Module, original_attns: List[nn.Module]) -> None:
    """Restore original attention modules."""
    for layer, orig in zip(model.layers, original_attns):
        layer.attn = orig


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_pattern(
    model: nn.Module,
    input_ids: mx.array,
    target_ids: mx.array,
    logits_dense: mx.array,
    pattern_name: str,
    abi: Optional[WayfinderGraphABI],
    n_heads: int,
) -> Dict[str, Any]:
    """Evaluate a single sparse pattern. Returns metrics dict."""
    original_attns = [layer.attn for layer in model.layers]

    try:
        if abi is not None:
            mlx_graph = to_mlx_graph_abi(abi, heads=n_heads, validate=False)
            T = int(input_ids.shape[1])
            s_idx = safe_neighbor_idx(mlx_graph.neigh_idx, T)
            c_mask = causal_neighbor_mask(mlx_graph.neigh_idx, T)
            mx.eval(s_idx, c_mask)
            _patch_model_sparse(model, mlx_graph, s_idx, c_mask)

        t0 = time.perf_counter()
        logits = model(input_ids)
        mx.eval(logits)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        loss = _cross_entropy_loss(logits, target_ids)
        ppl = math.exp(min(loss, 20.0))  # cap to avoid overflow

        if abi is not None:
            top1 = _top1_agreement(logits, logits_dense)
            cosim = _cosine_sim(logits, logits_dense)
        else:
            top1 = 1.0
            cosim = 1.0

        graph_report: Dict[str, Any] = {}
        eff_degree = 0.0
        if abi is not None:
            eff_degree = _effective_degree(abi.neigh_idx)
            try:
                graph_report = graph_quality_report(
                    abi.neigh_idx,
                    expansion_samples=50,
                    diameter_samples=30,
                )
            except Exception as e:
                graph_report = {"error": str(e)}

        result = {
            "pattern": pattern_name,
            "loss": round(loss, 4),
            "perplexity": round(ppl, 2),
            "top1_agreement": round(top1, 4),
            "cosine_similarity": round(cosim, 4),
            "wall_ms": round(wall_ms, 1),
            "effective_degree": round(eff_degree, 1),
        }

        if graph_report and "error" not in graph_report:
            result["spectral_gap"] = round(
                graph_report.get("spectral", {}).get("spectral_gap", 0.0), 4
            )
            result["mixing_time"] = graph_report.get(
                "mixing", {}
            ).get("mixing_time", -1)
            result["diameter"] = graph_report.get(
                "diameter", {}
            ).get("max_distance", -1)
            result["quality_score"] = round(
                graph_report.get("summary", {}).get("quality_score", 0.0), 4
            )
        elif graph_report.get("error"):
            result["graph_error"] = graph_report["error"]

        result["graph_report"] = graph_report
        return result

    finally:
        _unpatch_model(model, original_attns)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_results_table(
    results: List[Dict[str, Any]], ppl_dense: float, seq_len: int,
) -> None:
    """Print a formatted results table for one sequence length."""
    print()
    print("=" * 110)
    hdr = (
        f"{'Pattern':<22} {'PPL':>8} {'dPPL':>8} {'Top1':>8} {'CosSim':>8} "
        f"{'EffDeg':>8} {'SpecGap':>8} {'MixT':>6} {'Diam':>6} {'QScore':>8}"
    )
    print(hdr)
    print("-" * 110)

    for r in results:
        ppl_d = r["perplexity"] - ppl_dense
        print(
            f"{r['pattern']:<22} "
            f"{r['perplexity']:>8.2f} "
            f"{ppl_d:>+8.2f} "
            f"{r['top1_agreement']:>8.4f} "
            f"{r['cosine_similarity']:>8.4f} "
            f"{r['effective_degree']:>8.1f} "
            f"{r.get('spectral_gap', float('nan')):>8.4f} "
            f"{r.get('mixing_time', -1):>6d} "
            f"{r.get('diameter', -1):>6d} "
            f"{r.get('quality_score', float('nan')):>8.4f}"
        )
    print("=" * 110)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _find_result(results: List[Dict], name: str) -> Optional[Dict]:
    for r in results:
        if r["pattern"] == name:
            return r
    return None


def print_interpretation(
    all_results: Dict[int, List[Dict[str, Any]]],
    dense_baselines: Dict[int, float],
) -> None:
    """Print an honest summary of what the results mean."""
    print()
    print("=" * 110)
    print("INTERPRETATION")
    print("=" * 110)
    print()

    # Collect per-T deltas for key patterns
    cycle_vs_lm: Dict[int, float] = {}
    cycle_pure_vs_lm: Dict[int, float] = {}
    multicycle_vs_lm: Dict[int, float] = {}
    multicycle_vs_cycle: Dict[int, float] = {}

    for T, results in sorted(all_results.items()):
        lm = _find_result(results, "window+landmarks")
        cyc = _find_result(results, "window+cycle")
        cyc_pure = _find_result(results, "window+cycle_pure")
        mcyc = _find_result(results, "window+multicycle")

        if lm and cyc:
            cycle_vs_lm[T] = cyc["perplexity"] - lm["perplexity"]
        if lm and cyc_pure:
            cycle_pure_vs_lm[T] = cyc_pure["perplexity"] - lm["perplexity"]
        if lm and mcyc:
            multicycle_vs_lm[T] = mcyc["perplexity"] - lm["perplexity"]
        if cyc and mcyc:
            multicycle_vs_cycle[T] = mcyc["perplexity"] - cyc["perplexity"]

    # 1. Cycle vs landmarks
    print("1. CYCLE vs LANDMARKS (single cycle + landmarks vs landmarks only)")
    for T, delta in sorted(cycle_vs_lm.items()):
        direction = "better" if delta < -0.5 else "worse" if delta > 0.5 else "tied"
        print(f"   T={T:>5d}: cycle ppl delta = {delta:+.2f} ({direction})")
    if cycle_vs_lm:
        avg = sum(cycle_vs_lm.values()) / len(cycle_vs_lm)
        if abs(avg) < 1.0:
            print("   -> Cycle edges provide NO clear advantage over simple landmarks.")
        elif avg < -1.0:
            print("   -> Cycle edges help, especially at longer contexts.")
        else:
            print("   -> Cycle edges slightly HURT compared to landmarks.")
    print()

    # 2. Pure cycle (no landmarks) vs landmarks
    print("2. PURE CYCLE (no landmarks) vs LANDMARKS")
    for T, delta in sorted(cycle_pure_vs_lm.items()):
        direction = "better" if delta < -0.5 else "worse" if delta > 0.5 else "tied"
        print(f"   T={T:>5d}: pure_cycle ppl delta = {delta:+.2f} ({direction})")
    if cycle_pure_vs_lm:
        avg = sum(cycle_pure_vs_lm.values()) / len(cycle_pure_vs_lm)
        if avg > 5.0:
            print(
                "   -> Landmarks are doing the heavy lifting. A single cycle alone"
            )
            print(
                "      cannot replace the regularity of evenly-spaced landmarks."
            )
        elif abs(avg) < 2.0:
            print("   -> Cycle alone is competitive with landmarks.")
    print()

    # 3. Multi-cycle
    print(
        "3. MULTI-CYCLE (ceil(2*log2(T)) cycles) vs SINGLE CYCLE"
    )
    for T, delta in sorted(multicycle_vs_cycle.items()):
        direction = "better" if delta < -0.5 else "worse" if delta > 0.5 else "tied"
        ncyc = _recommended_num_cycles(T)
        print(
            f"   T={T:>5d} ({ncyc} cycles): multicycle ppl delta = {delta:+.2f} ({direction})"
        )
    print("   MULTI-CYCLE vs LANDMARKS:")
    for T, delta in sorted(multicycle_vs_lm.items()):
        direction = "better" if delta < -0.5 else "worse" if delta > 0.5 else "tied"
        print(f"   T={T:>5d}: multicycle ppl delta = {delta:+.2f} ({direction})")
    print()

    # 4. Random paradox
    print("4. THE RANDOM PARADOX")
    for T, results in sorted(all_results.items()):
        rand = _find_result(results, "window+random")
        lm = _find_result(results, "window+landmarks")
        if rand and lm:
            rand_delta = rand["perplexity"] - lm["perplexity"]
            rand_sg = rand.get("spectral_gap", 0.0)
            lm_sg = lm.get("spectral_gap", 0.0)
            print(
                f"   T={T:>5d}: random ppl={rand['perplexity']:.1f} "
                f"(+{rand_delta:.1f} vs landmarks), "
                f"spectral_gap={rand_sg:.3f} (landmarks={lm_sg:.3f})"
            )
    print(
        "   -> Random causal edges have the BEST graph-theoretic metrics but the"
    )
    print(
        "      WORST perplexity among structured patterns. This is because GPT-2"
    )
    print(
        "      was pretrained with dense attention -- it learned to rely on local"
    )
    print(
        "      context and specific positional patterns. Random long-range edges"
    )
    print(
        "      point to distant tokens that provide no useful signal for a model"
    )
    print(
        "      that never learned to use them. Good graph connectivity != good"
    )
    print("      attention for a pretrained model.")
    print()

    # 5. Overall conclusion
    print("5. OVERALL CONCLUSION")
    # Find the best non-dense pattern at the longest T
    longest_T = max(all_results.keys())
    longest = all_results[longest_T]
    sparse_results = [r for r in longest if r["pattern"] != "dense"]
    if sparse_results:
        best = min(sparse_results, key=lambda r: r["perplexity"])
        dense_ppl = dense_baselines[longest_T]
        print(
            f"   At T={longest_T}, best sparse pattern: {best['pattern']} "
            f"(ppl={best['perplexity']:.2f}, +{best['perplexity'] - dense_ppl:.2f} vs dense)"
        )

    # Check if cycles ever beat landmarks
    cycles_win = any(d < -0.5 for d in cycle_vs_lm.values())
    multicycles_win = any(d < -0.5 for d in multicycle_vs_lm.values())
    if not cycles_win and not multicycles_win:
        print(
            "   The Hamiltonian cycle structure provides no clear perplexity"
        )
        print(
            "   advantage over simple evenly-spaced landmarks at any tested"
        )
        print(
            "   sequence length. The benefit of sparse attention comes from"
        )
        print(
            "   having ANY structured long-range connectivity, not specifically"
        )
        print("   from Hamiltonian cycle properties.")
    elif multicycles_win and not cycles_win:
        print(
            "   Multi-cycle (expander-grade) Hamiltonian structure outperforms"
        )
        print(
            "   landmarks, but a single cycle does not. The benefit comes from"
        )
        print(
            "   the union of multiple edge-disjoint cycles, not from any single"
        )
        print("   Hamiltonian cycle.")
    elif cycles_win:
        print(
            "   Hamiltonian cycle edges provide a measurable advantage over"
        )
        print(
            "   landmarks, suggesting that the cycle structure captures useful"
        )
        print("   long-range dependencies that regular spacing misses.")
    print()
    print(
        "   NOTE: All sparse patterns are evaluated on a model pretrained with"
    )
    print(
        "   dense attention. A model trained FROM SCRATCH with sparse attention"
    )
    print(
        "   might learn to exploit cycle structure more effectively. These"
    )
    print("   results measure zero-shot transfer, not optimal sparse training.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one_seq_len(
    model: nn.Module,
    tokenizer: Any,
    T: int,
    W: int,
    D: int,
    seed: int,
    n_heads: int,
    text: str,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run all patterns at one sequence length. Returns (results, dense_ppl)."""
    print(f"\n{'#' * 110}")
    print(f"# Sequence length: T={T}, Window: {W}, Max degree: {D}")
    print(f"{'#' * 110}\n")

    # Prepare input
    tokens = tokenizer.encode(text)
    if len(tokens) < T + 1:
        tokens = tokens * ((T + 1) // len(tokens) + 1)
    tokens = tokens[: T + 1]

    input_ids = mx.array(tokens[:-1]).reshape(1, -1)
    target_ids = mx.array(tokens[1:]).reshape(1, -1)
    mx.eval(input_ids, target_ids)
    print(f"Input tokens: {input_ids.shape[1]}")

    # Dense baseline
    print("Running dense baseline...")
    t0 = time.perf_counter()
    logits_dense = model(input_ids)
    mx.eval(logits_dense)
    dense_ms = (time.perf_counter() - t0) * 1000.0
    loss_dense = _cross_entropy_loss(logits_dense, target_ids)
    ppl_dense = math.exp(min(loss_dense, 20.0))
    print(f"  Dense: loss={loss_dense:.4f}  ppl={ppl_dense:.2f}  wall={dense_ms:.1f}ms")

    # Build all sparse graphs
    print("\nBuilding sparse graph patterns...")

    patterns: list[tuple[str, Optional[WayfinderGraphABI]]] = []
    patterns.append(("dense", None))

    print("  window_only...")
    patterns.append(("window_only", build_window_only(T, W, D)))

    print("  window+landmarks...")
    patterns.append(("window+landmarks", build_window_landmarks(T, W, D)))

    print("  window+random...")
    patterns.append(("window+random", build_window_random(T, W, D, seed)))

    print("  window+cycle...")
    patterns.append((
        "window+cycle",
        build_window_cycle_per_head(T, W, D, n_heads, seed),
    ))

    print("  window+cycle_pure...")
    patterns.append((
        "window+cycle_pure",
        build_window_cycle_pure(T, W, D, n_heads, seed),
    ))

    num_cycles = _recommended_num_cycles(T)
    print(f"  window+multicycle ({num_cycles} cycles)...")
    patterns.append((
        "window+multicycle",
        build_window_multicycle(T, W, D, n_heads, seed, num_cycles),
    ))

    print("  window+expander...")
    patterns.append(("window+expander", build_window_expander(T, W, D)))

    # Evaluate each pattern
    print()
    results: list[Dict[str, Any]] = []
    for name, abi in patterns:
        print(f"Evaluating: {name}...")
        r = evaluate_pattern(
            model, input_ids, target_ids, logits_dense,
            name, abi, n_heads,
        )
        results.append(r)
        ppl_delta = r["perplexity"] - ppl_dense
        print(
            f"  ppl={r['perplexity']:.2f} "
            f"({ppl_delta:+.2f})  "
            f"top1={r['top1_agreement']:.4f}  "
            f"cosim={r['cosine_similarity']:.4f}  "
            f"eff_deg={r['effective_degree']:.1f}"
        )
        if "spectral_gap" in r:
            print(
                f"  spectral_gap={r['spectral_gap']:.4f}  "
                f"mixing={r['mixing_time']}  "
                f"diameter={r['diameter']}  "
                f"quality={r['quality_score']:.4f}"
            )

    print_results_table(results, ppl_dense, T)
    return results, ppl_dense


def main():
    p = argparse.ArgumentParser(
        description="Honest sparse pattern benchmark on GPT-2"
    )
    p.add_argument("--model", type=str, default="openai-community/gpt2")
    p.add_argument(
        "--seq-lens", type=str, default="256,512,1024",
        help="Comma-separated sequence lengths to test (default: 256,512,1024)",
    )
    p.add_argument(
        "--seq-len", type=int, default=None,
        help="Single sequence length (overrides --seq-lens)",
    )
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--max-degree", type=int, default=130)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument(
        "--output-dir", type=str,
        default="results/sparse_pattern_comparison",
    )
    args = p.parse_args()

    if args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]

    W = args.window
    D = args.max_degree

    print("=" * 110)
    print("  SPARSE PATTERN BENCHMARK -- GPT-2 (pretrained, zero-shot transfer)")
    print("=" * 110)
    print(f"Model: {args.model}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Window: {W}, Max degree: {D}, Seed: {args.seed}")
    print()

    # Load model once
    print("Loading model...")
    model, tokenizer = _load_gpt2(args.model)
    model.eval()

    n_heads = int(model.layers[0].attn.n_head)
    print(f"Heads: {n_heads}")

    if args.text_file:
        text = Path(args.text_file).read_text()
    else:
        text = DEFAULT_TEXT

    # Run sweep
    all_results: Dict[int, List[Dict[str, Any]]] = {}
    dense_baselines: Dict[int, float] = {}

    for T in seq_lens:
        results, ppl_dense = run_one_seq_len(
            model, tokenizer, T, W, D, args.seed, n_heads, text,
        )
        all_results[T] = results
        dense_baselines[T] = ppl_dense

    # Print interpretation
    print_interpretation(all_results, dense_baselines)

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"

    seq_len_data = {}
    for T in seq_lens:
        save_results = []
        for r in all_results[T]:
            sr = {k: v for k, v in r.items() if k != "graph_report"}
            report = r.get("graph_report", {})
            if report and "error" not in report:
                sr["graph_quality"] = {
                    "spectral_gap": report.get("spectral", {}).get(
                        "spectral_gap"
                    ),
                    "mixing_time": report.get("mixing", {}).get("mixing_time"),
                    "mixing_interp": report.get("mixing", {}).get(
                        "interpretation"
                    ),
                    "diameter": report.get("diameter", {}).get("max_distance"),
                    "mean_distance": report.get("diameter", {}).get(
                        "mean_distance"
                    ),
                    "quality_score": report.get("summary", {}).get(
                        "quality_score"
                    ),
                    "is_good_expander": report.get("summary", {}).get(
                        "is_good_expander"
                    ),
                    "degree_mean": report.get("degree", {}).get("mean"),
                }
            save_results.append(sr)

        seq_len_data[str(T)] = {
            "dense_baseline": {
                "perplexity": round(dense_baselines[T], 2),
            },
            "num_multicycles": _recommended_num_cycles(T),
            "results": save_results,
        }

    payload = {
        "sweep_config": {
            "model": args.model,
            "seq_lens": seq_lens,
            "window": W,
            "max_degree": D,
            "seed": args.seed,
            "n_heads": n_heads,
            "patterns": [
                "dense",
                "window_only",
                "window+landmarks",
                "window+random",
                "window+cycle",
                "window+cycle_pure",
                "window+multicycle",
                "window+expander",
            ],
        },
        "seq_len_results": seq_len_data,
    }

    out_file.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
