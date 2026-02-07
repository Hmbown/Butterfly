"""TTT-Discover reward function.

Reward = f(tokens/sec, peak_memory) subject to correctness + perplexity
constraints.  The reward trades off throughput against quality (perplexity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for the reward function."""

    # Target perplexity: configs exceeding this by a large margin are penalized.
    target_ppl: float = 50.0

    # How harshly to penalize perplexity above target.
    # reward *= (1 - max(0, (ppl - target_ppl) / target_ppl))
    ppl_penalty_scale: float = 1.0

    # Maximum acceptable perplexity (hard constraint: reward = 0 if exceeded).
    max_ppl: float = 500.0

    # Peak memory budget in bytes (hard constraint).
    max_memory_bytes: Optional[int] = None

    # Throughput normalization: divide tok/s by this to get [0, ~1] range.
    throughput_normalizer: float = 100_000.0

    # Correctness constraint: if loss is NaN/Inf, reward = 0.
    require_finite_loss: bool = True


def compute_reward(
    tokens_per_sec: float,
    val_ppl: float,
    peak_memory_bytes: int = 0,
    loss_is_finite: bool = True,
    config: Optional[RewardConfig] = None,
) -> float:
    """Compute reward for a TTT-Discover configuration evaluation.

    Parameters
    ----------
    tokens_per_sec : float
        Training throughput.
    val_ppl : float
        Validation perplexity.
    peak_memory_bytes : int
        Peak GPU/device memory used.
    loss_is_finite : bool
        Whether all losses during training were finite.
    config : RewardConfig, optional
        Reward function parameters.

    Returns
    -------
    float
        Reward value (higher is better).  Returns 0.0 for constraint violations.
    """
    if config is None:
        config = RewardConfig()

    # Hard constraints
    if config.require_finite_loss and not loss_is_finite:
        return 0.0

    if not math.isfinite(val_ppl) or val_ppl <= 0:
        return 0.0

    if val_ppl > config.max_ppl:
        return 0.0

    if config.max_memory_bytes is not None and peak_memory_bytes > config.max_memory_bytes:
        return 0.0

    # Throughput component: normalized to rough [0, 1+] range
    throughput_score = tokens_per_sec / config.throughput_normalizer

    # Perplexity penalty: multiplicative factor in [0, 1]
    ppl_excess = max(0.0, val_ppl - config.target_ppl) / config.target_ppl
    ppl_factor = max(0.0, 1.0 - config.ppl_penalty_scale * ppl_excess)

    # Combined reward
    reward = throughput_score * ppl_factor

    return reward


def compute_reward_detailed(
    tokens_per_sec: float,
    val_ppl: float,
    peak_memory_bytes: int = 0,
    loss_is_finite: bool = True,
    config: Optional[RewardConfig] = None,
) -> dict[str, float]:
    """Like compute_reward but returns a breakdown of components."""
    if config is None:
        config = RewardConfig()

    result = {
        "reward": 0.0,
        "throughput_score": 0.0,
        "ppl_factor": 0.0,
        "ppl_excess": 0.0,
        "constraint_violated": 0.0,
    }

    # Check constraints
    if config.require_finite_loss and not loss_is_finite:
        result["constraint_violated"] = 1.0
        return result
    if not math.isfinite(val_ppl) or val_ppl <= 0:
        result["constraint_violated"] = 1.0
        return result
    if val_ppl > config.max_ppl:
        result["constraint_violated"] = 1.0
        return result
    if config.max_memory_bytes is not None and peak_memory_bytes > config.max_memory_bytes:
        result["constraint_violated"] = 1.0
        return result

    throughput_score = tokens_per_sec / config.throughput_normalizer
    ppl_excess = max(0.0, val_ppl - config.target_ppl) / config.target_ppl
    ppl_factor = max(0.0, 1.0 - config.ppl_penalty_scale * ppl_excess)

    result["throughput_score"] = throughput_score
    result["ppl_factor"] = ppl_factor
    result["ppl_excess"] = ppl_excess
    result["reward"] = throughput_score * ppl_factor

    return result
