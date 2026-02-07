"""TTT-Discover config proposal model.

A lightweight policy that proposes new graph strategy configurations given
the current best config and device characteristics.  This is intentionally
simple (not a full LLM) - it uses parameter perturbation and heuristic rules.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Any, Dict, List, Optional

from .environment import GraphConfig, ModelConfig, TTTConfig
from .buffer import ReplayBuffer


class ConfigProposer:
    """Proposes new TTT configurations for evaluation.

    Uses a combination of:
    1. Random perturbation of current best config
    2. Crossover between top-k configs
    3. Heuristic parameter sweeps

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, seed: int = 0):
        self.rng = stdlib_random.Random(seed)

    # Valid parameter ranges
    CYCLE_OPTIONS = ["random", "greedy", "online_insertion"]
    WINDOW_RANGE = (2, 256)
    LANDMARK_RANGE = (4, 256)
    NUM_CYCLES_RANGE = (1, 4)

    def propose_random(self) -> TTTConfig:
        """Propose a completely random configuration."""
        return TTTConfig(
            graph=GraphConfig(
                cycle=self.rng.choice(self.CYCLE_OPTIONS),
                window=self.rng.choice([2, 4, 8, 16, 32, 64, 128]),
                landmark_stride=self.rng.choice([4, 8, 16, 32, 64, 128]),
                num_cycles=self.rng.randint(*self.NUM_CYCLES_RANGE),
            ),
        )

    def propose_perturbation(self, base: TTTConfig) -> TTTConfig:
        """Perturb a base configuration slightly."""
        graph = GraphConfig(
            cycle=base.graph.cycle,
            window=base.graph.window,
            landmark_stride=base.graph.landmark_stride,
            num_cycles=base.graph.num_cycles,
        )

        # Randomly perturb one parameter
        choice = self.rng.randint(0, 3)
        if choice == 0:
            # Perturb window
            factor = self.rng.choice([0.5, 0.75, 1.5, 2.0])
            graph.window = max(2, min(256, int(graph.window * factor)))
        elif choice == 1:
            # Perturb landmark stride
            factor = self.rng.choice([0.5, 0.75, 1.5, 2.0])
            graph.landmark_stride = max(4, min(256, int(graph.landmark_stride * factor)))
        elif choice == 2:
            # Change cycle strategy
            graph.cycle = self.rng.choice(self.CYCLE_OPTIONS)
        else:
            # Perturb num_cycles
            graph.num_cycles = self.rng.randint(*self.NUM_CYCLES_RANGE)

        return TTTConfig(
            graph=graph,
            model=base.model,
            lr=base.lr,
            batch_size=base.batch_size,
            eval_steps=base.eval_steps,
        )

    def propose_crossover(
        self,
        parent1: TTTConfig,
        parent2: TTTConfig,
    ) -> TTTConfig:
        """Crossover two configurations (take parameters from each parent)."""
        graph = GraphConfig(
            cycle=self.rng.choice([parent1.graph.cycle, parent2.graph.cycle]),
            window=self.rng.choice([parent1.graph.window, parent2.graph.window]),
            landmark_stride=self.rng.choice(
                [parent1.graph.landmark_stride, parent2.graph.landmark_stride]
            ),
            num_cycles=self.rng.choice(
                [parent1.graph.num_cycles, parent2.graph.num_cycles]
            ),
        )

        return TTTConfig(
            graph=graph,
            model=parent1.model,  # keep model fixed
            lr=self.rng.choice([parent1.lr, parent2.lr]),
            batch_size=parent1.batch_size,
            eval_steps=parent1.eval_steps,
        )

    def propose(
        self,
        buffer: Optional[ReplayBuffer] = None,
        exploration_rate: float = 0.3,
    ) -> TTTConfig:
        """Propose a new configuration using available history.

        Parameters
        ----------
        buffer : ReplayBuffer, optional
            Buffer of previously evaluated configs.
        exploration_rate : float
            Probability of pure random exploration.
        """
        # Pure random exploration
        if buffer is None or len(buffer) < 2 or self.rng.random() < exploration_rate:
            if buffer and len(buffer) > 0:
                top = buffer.top_k(1)[0]
                return self.propose_perturbation(top.config)
            return self.propose_random()

        # Exploit: perturb best or crossover top-2
        top_entries = buffer.top_k(3)

        if self.rng.random() < 0.5 and len(top_entries) >= 2:
            # Crossover
            return self.propose_crossover(
                top_entries[0].config,
                top_entries[1].config,
            )
        else:
            # Perturbation of best
            return self.propose_perturbation(top_entries[0].config)
