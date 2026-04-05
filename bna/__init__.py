"""Butterfly Network Attention (BNA): training-free sparse attention for long-context inference."""

from .model import GPT, GPTConfig
from .topology import Topology, TopologyGraph

__all__ = ["GPT", "GPTConfig", "Topology", "TopologyGraph"]
__version__ = "0.3.0"
