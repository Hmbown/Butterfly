"""TTT-Discover environment.

Defines the state (graph strategy parameters + kernel choices) and action
(propose a new config) for the meta-learning loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GraphConfig:
    """Parameters defining an HCSA graph strategy configuration."""

    cycle: str = "random"
    window: int = 64
    landmark_stride: int = 64
    num_cycles: int = 1
    routing_dim: Optional[int] = None

    # Block-wise cycle construction parameters
    block_size: Optional[int] = None  # None = full sequence
    update_every_k: int = 1  # rebuild cycle every k steps


@dataclass
class ModelConfig:
    """Model architecture parameters explored by TTT."""

    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 512
    seq_len: int = 256
    dropout: float = 0.0


@dataclass
class TTTConfig:
    """Full configuration explored by the TTT-Discover loop."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training parameters
    lr: float = 3e-4
    batch_size: int = 32
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    amp: bool = False

    # Evaluation budget
    eval_steps: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TTTConfig":
        graph = GraphConfig(**d.get("graph", {}))
        model = ModelConfig(**d.get("model", {}))
        return cls(
            graph=graph,
            model=model,
            lr=d.get("lr", 3e-4),
            batch_size=d.get("batch_size", 32),
            weight_decay=d.get("weight_decay", 0.1),
            grad_clip=d.get("grad_clip", 1.0),
            amp=d.get("amp", False),
            eval_steps=d.get("eval_steps", 200),
        )


@dataclass
class TTTState:
    """State of the TTT-Discover loop."""

    current_config: TTTConfig
    best_config: Optional[TTTConfig] = None
    best_reward: float = float("-inf")
    iteration: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TTTAction:
    """An action in the TTT-Discover loop: propose a new config."""

    proposed_config: TTTConfig
    reason: str = ""


class TTTEnvironment:
    """TTT-Discover environment.

    Manages state transitions and tracks exploration history.
    """

    def __init__(self, initial_config: Optional[TTTConfig] = None):
        self.state = TTTState(
            current_config=initial_config or TTTConfig(),
        )

    def step(self, action: TTTAction, reward: float, metrics: Dict[str, Any]) -> TTTState:
        """Apply an action and update state with the observed reward.

        Parameters
        ----------
        action : TTTAction
            The proposed configuration.
        reward : float
            Observed reward for this configuration.
        metrics : dict
            Additional metrics (val_loss, tokens/sec, memory, etc.).

        Returns
        -------
        TTTState
            Updated environment state.
        """
        entry = {
            "iteration": self.state.iteration,
            "config": action.proposed_config.to_dict(),
            "reward": reward,
            "metrics": metrics,
            "reason": action.reason,
        }
        self.state.history.append(entry)

        if reward > self.state.best_reward:
            self.state.best_reward = reward
            self.state.best_config = action.proposed_config

        self.state.current_config = action.proposed_config
        self.state.iteration += 1
        return self.state

    def reset(self, config: Optional[TTTConfig] = None) -> TTTState:
        """Reset the environment."""
        self.state = TTTState(
            current_config=config or TTTConfig(),
        )
        return self.state

    def get_top_k(self, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k configurations by reward."""
        sorted_history = sorted(
            self.state.history, key=lambda x: x["reward"], reverse=True
        )
        return sorted_history[:k]
