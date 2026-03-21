from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str = "constant"
    start: float = 0.0
    end: float = 0.0
    steps: int = 1

    def value(self, step: int) -> float:
        if self.kind == "linear":
            if self.steps <= 1:
                return float(self.end)
            alpha = min(1.0, max(0.0, float(step) / float(self.steps - 1)))
            return float(self.start + alpha * (self.end - self.start))
        return float(self.end)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeBiasScheduleIR:
    cycle: Optional[ScheduleSpec] = None
    window: Optional[ScheduleSpec] = None
    landmark: Optional[ScheduleSpec] = None
    rewire: Optional[ScheduleSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": None if self.cycle is None else self.cycle.to_dict(),
            "window": None if self.window is None else self.window.to_dict(),
            "landmark": None if self.landmark is None else self.landmark.to_dict(),
            "rewire": None if self.rewire is None else self.rewire.to_dict(),
        }


@dataclass(frozen=True)
class GraphIR:
    degree: int = 64
    strategy: str = "random"
    seed: int = 42
    num_cycles: int = 1
    window_size: int = 32
    window_schedule: Optional[ScheduleSpec] = None
    landmark_stride: Optional[int] = 64
    edge_bias: EdgeBiasScheduleIR = EdgeBiasScheduleIR()
    permute_window_enabled: bool = True
    permute_window_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degree": int(self.degree),
            "strategy": self.strategy,
            "seed": int(self.seed),
            "num_cycles": int(self.num_cycles),
            "window_size": int(self.window_size),
            "window_schedule": (
                None if self.window_schedule is None else self.window_schedule.to_dict()
            ),
            "landmark_stride": (
                None if self.landmark_stride is None else int(self.landmark_stride)
            ),
            "edge_bias": self.edge_bias.to_dict(),
            "permute_window_enabled": bool(self.permute_window_enabled),
            "permute_window_size": int(self.permute_window_size),
        }
