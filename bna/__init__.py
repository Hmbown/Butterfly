"""HCSA: Hamiltonian Cycle Sparse Attention for language models."""

from .model import GPT, GPTConfig
from .topology import Topology, TopologyGraph

__all__ = ["GPT", "GPTConfig", "Topology", "TopologyGraph"]
__version__ = "0.3.0"
