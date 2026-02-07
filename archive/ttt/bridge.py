"""Bridge between hamiltonian_ttt_kit and TTT-Discover loop.

Connects the Hamiltonian Neural Network test-time tuning paradigm from
hamiltonian_ttt_kit with the graph strategy optimization in TTT-Discover.

The key idea: the hamiltonian_ttt_kit provides a TTT framework where a
small "head" network is tuned at test time.  We reuse that pattern for
HCSA graph optimization:
- The "base model" is the frozen LM with a fixed graph strategy
- The "head" is the graph strategy parameters
- "Test-time tuning" means adapting the graph to the current input

This module provides utilities to bridge the two systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .environment import TTTConfig, GraphConfig
from .reward import RewardConfig
from .loop import TTTLoop


@dataclass
class BridgeConfig:
    """Configuration for the hamiltonian_ttt_kit bridge."""

    # TTT-Discover parameters
    n_iterations: int = 10
    eval_steps: int = 100

    # Reward function
    target_ppl: float = 50.0
    max_ppl: float = 500.0

    # Whether to use energy-conservation-inspired constraints
    # (from hamiltonian_ttt_kit: prefer graph strategies that
    #  maintain information flow = "energy conservation" in the
    #  attention graph)
    use_energy_constraint: bool = False
    energy_threshold: float = 0.1


def graph_config_to_dict(cfg: GraphConfig) -> Dict[str, Any]:
    """Serialize a GraphConfig for logging."""
    return {
        "cycle": cfg.cycle,
        "window": cfg.window,
        "landmark_stride": cfg.landmark_stride,
        "num_cycles": cfg.num_cycles,
    }


def dict_to_graph_config(d: Dict[str, Any]) -> GraphConfig:
    """Deserialize a GraphConfig from dict."""
    return GraphConfig(
        cycle=d.get("cycle", "random"),
        window=d.get("window", 64),
        landmark_stride=d.get("landmark_stride", 64),
        num_cycles=d.get("num_cycles", 1),
    )


class TTTKitBridge:
    """Bridge hamiltonian_ttt_kit patterns into TTT-Discover.

    The Hamiltonian TTT kit uses the pattern:
    1. Train a base model
    2. At test time, tune a small "head" on observed data
    3. The head corrects the base model's predictions

    We adapt this for graph strategy optimization:
    1. Base: the HCSA attention mechanism with default graph
    2. Head: graph strategy parameters (cycle type, window, landmarks)
    3. Tuning: TTT-Discover loop optimizes graph parameters

    This bridge provides:
    - Conversion utilities between the two config formats
    - Shared evaluation infrastructure
    - Energy-conservation constraints inspired by Hamiltonian mechanics
    """

    def __init__(
        self,
        bridge_config: Optional[BridgeConfig] = None,
    ):
        self.config = bridge_config or BridgeConfig()

    def create_ttt_config(
        self,
        graph: GraphConfig,
        n_layers: int = 4,
        n_heads: int = 4,
        n_embd: int = 256,
        seq_len: int = 256,
    ) -> TTTConfig:
        """Create a TTTConfig from graph parameters + model spec."""
        from .environment import ModelConfig
        return TTTConfig(
            graph=graph,
            model=ModelConfig(
                n_layers=n_layers,
                n_heads=n_heads,
                n_embd=n_embd,
                seq_len=seq_len,
            ),
            eval_steps=self.config.eval_steps,
        )

    def configs_from_ttt_kit_sweep(
        self,
        cycle_types: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        landmarks: Optional[List[int]] = None,
    ) -> List[GraphConfig]:
        """Generate graph configs for a systematic sweep.

        Inspired by the hamiltonian_ttt_kit's approach of systematically
        varying parameters to find optimal configurations.
        """
        if cycle_types is None:
            cycle_types = ["random", "greedy"]
        if windows is None:
            windows = [8, 16, 32, 64]
        if landmarks is None:
            landmarks = [8, 16, 32, 64]

        configs = []
        for cycle in cycle_types:
            for w in windows:
                for lm in landmarks:
                    configs.append(GraphConfig(
                        cycle=cycle,
                        window=w,
                        landmark_stride=lm,
                    ))
        return configs

    def evaluate_with_energy_constraint(
        self,
        metrics: Dict[str, Any],
    ) -> float:
        """Apply energy-conservation-inspired constraint to reward.

        In Hamiltonian mechanics, energy is conserved. We use this as
        an analogy: a good graph strategy should "conserve information
        flow" - the attention pattern should maintain roughly constant
        total attention weight to non-local tokens across layers.

        When energy_constraint is enabled, we penalize configs where
        the model's attention entropy varies wildly across layers
        (suggesting information bottlenecks in the graph).
        """
        if not self.config.use_energy_constraint:
            return 1.0

        # Check if attention entropy information is available
        entropy_per_layer = metrics.get("attn_entropy_per_layer")
        if entropy_per_layer is None or len(entropy_per_layer) < 2:
            return 1.0

        # Compute variance of entropy across layers
        mean_h = sum(entropy_per_layer) / len(entropy_per_layer)
        if mean_h < 1e-8:
            return 0.5

        variance = sum((h - mean_h) ** 2 for h in entropy_per_layer) / len(entropy_per_layer)
        cv = (variance ** 0.5) / mean_h  # coefficient of variation

        # Penalize high variation (information bottleneck)
        if cv > self.config.energy_threshold:
            penalty = max(0.0, 1.0 - (cv - self.config.energy_threshold))
            return penalty
        return 1.0

    def run_discovery(
        self,
        train_data,
        val_data,
        vocab_size: int,
        device,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run TTT-Discover loop with bridge configuration.

        This is a convenience wrapper that creates and runs a TTTLoop
        with the bridge's reward configuration.
        """
        reward_config = RewardConfig(
            target_ppl=self.config.target_ppl,
            max_ppl=self.config.max_ppl,
        )

        loop = TTTLoop(
            train_data=train_data,
            val_data=val_data,
            vocab_size=vocab_size,
            device=device,
            reward_config=reward_config,
            n_iterations=self.config.n_iterations,
        )

        return loop.run(verbose=verbose)
