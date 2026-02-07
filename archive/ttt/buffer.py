"""TTT-Discover reuse buffer with PUCT-style selection.

Stores evaluated configurations with their rewards.  Uses a PUCT-like
upper confidence bound for balancing exploration (untried configs) vs
exploitation (high-reward configs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .environment import TTTConfig


@dataclass
class BufferEntry:
    """A single evaluated configuration."""

    config: TTTConfig
    reward: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    visit_count: int = 1


class ReplayBuffer:
    """Store and select from previously evaluated configurations.

    Parameters
    ----------
    capacity : int
        Maximum number of entries to store.
    c_puct : float
        Exploration constant for PUCT selection.  Higher values favour
        exploration of less-visited configs.
    """

    def __init__(self, capacity: int = 1000, c_puct: float = 1.0):
        self.capacity = capacity
        self.c_puct = c_puct
        self.entries: List[BufferEntry] = []
        self.total_visits = 0

    def add(
        self,
        config: TTTConfig,
        reward: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new evaluation result or update existing entry."""
        # Check if this config already exists (by rough equality)
        config_dict = config.to_dict()
        for entry in self.entries:
            if entry.config.to_dict() == config_dict:
                # Update with running average
                entry.visit_count += 1
                entry.reward = (
                    entry.reward * (entry.visit_count - 1) + reward
                ) / entry.visit_count
                if metrics:
                    entry.metrics = metrics
                self.total_visits += 1
                return

        # New entry
        entry = BufferEntry(
            config=config,
            reward=reward,
            metrics=metrics or {},
        )
        if len(self.entries) >= self.capacity:
            # Remove lowest reward entry
            self.entries.sort(key=lambda e: e.reward)
            self.entries.pop(0)
        self.entries.append(entry)
        self.total_visits += 1

    def select_puct(self) -> Optional[BufferEntry]:
        """Select a configuration using PUCT-style upper confidence bound.

        UCB(i) = Q(i) + c_puct * sqrt(ln(N) / n_i)

        where Q(i) is the normalized reward, N is total visits, n_i is
        visit count for entry i.

        Returns the entry with highest UCB score.
        """
        if not self.entries:
            return None

        # Normalize rewards to [0, 1]
        rewards = [e.reward for e in self.entries]
        r_min = min(rewards)
        r_max = max(rewards)
        r_range = max(r_max - r_min, 1e-8)

        best_score = float("-inf")
        best_entry = None

        log_N = math.log(max(self.total_visits, 1))

        for entry in self.entries:
            q = (entry.reward - r_min) / r_range  # normalized reward
            exploration = self.c_puct * math.sqrt(log_N / max(entry.visit_count, 1))
            score = q + exploration

            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry

    def top_k(self, k: int = 5) -> List[BufferEntry]:
        """Return top-k entries by reward."""
        return sorted(self.entries, key=lambda e: e.reward, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "c_puct": self.c_puct,
            "total_visits": self.total_visits,
            "entries": [
                {
                    "config": e.config.to_dict(),
                    "reward": e.reward,
                    "metrics": e.metrics,
                    "visit_count": e.visit_count,
                }
                for e in self.entries
            ],
        }
