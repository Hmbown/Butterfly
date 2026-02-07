"""Device-aware optimization for TTT-Discover.

Profiles the current device (GPU type, memory, bandwidth) and adapts
the TTT search space to device characteristics.  For example:
- On high-memory GPUs: allow larger D, more cycles, bigger batches
- On memory-constrained devices: prefer lower D, smaller windows
- On Apple Silicon: prefer contiguous memory patterns (permute-to-cycle-order)
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DeviceProfile:
    """Hardware profile for device-aware optimization."""

    device_type: str = "cpu"  # "cpu", "cuda", "mps"
    device_name: str = "unknown"
    total_memory_bytes: int = 0
    compute_capability: Optional[str] = None

    # Inferred characteristics
    memory_tier: str = "low"  # "low" (<8GB), "medium" (8-24GB), "high" (>24GB)
    supports_flash_attn: bool = False
    supports_bf16: bool = False
    prefers_contiguous: bool = False  # True for Apple Silicon / MPS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_capability": self.compute_capability,
            "memory_tier": self.memory_tier,
            "supports_flash_attn": self.supports_flash_attn,
            "supports_bf16": self.supports_bf16,
            "prefers_contiguous": self.prefers_contiguous,
        }


def profile_device(device: Optional[torch.device] = None) -> DeviceProfile:
    """Profile the given or default device.

    Parameters
    ----------
    device : torch.device, optional
        Device to profile.  If None, auto-detect best available.

    Returns
    -------
    DeviceProfile
        Hardware characteristics.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    profile = DeviceProfile(device_type=device.type)

    if device.type == "cuda":
        idx = device.index or 0
        profile.device_name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        profile.total_memory_bytes = props.total_mem
        profile.compute_capability = f"{props.major}.{props.minor}"

        # Memory tier
        gb = profile.total_memory_bytes / (1024 ** 3)
        if gb >= 24:
            profile.memory_tier = "high"
        elif gb >= 8:
            profile.memory_tier = "medium"
        else:
            profile.memory_tier = "low"

        # FlashAttention requires compute capability >= 8.0 (Ampere+)
        profile.supports_flash_attn = props.major >= 8
        profile.supports_bf16 = props.major >= 8
        profile.prefers_contiguous = False

    elif device.type == "mps":
        profile.device_name = f"Apple Silicon ({platform.processor() or 'unknown'})"
        # MPS doesn't expose total memory directly; estimate from system
        try:
            import os
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            # Unified memory: GPU can use most of system RAM
            profile.total_memory_bytes = int(mem_bytes * 0.75)
        except (ValueError, OSError):
            profile.total_memory_bytes = 8 * 1024 ** 3  # assume 8GB

        gb = profile.total_memory_bytes / (1024 ** 3)
        if gb >= 24:
            profile.memory_tier = "high"
        elif gb >= 8:
            profile.memory_tier = "medium"
        else:
            profile.memory_tier = "low"

        profile.supports_flash_attn = False
        profile.supports_bf16 = False
        profile.prefers_contiguous = True  # Metal prefers contiguous access

    else:
        profile.device_name = f"CPU ({platform.processor() or platform.machine()})"
        try:
            import os
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            profile.total_memory_bytes = mem_bytes
        except (ValueError, OSError, AttributeError):
            profile.total_memory_bytes = 4 * 1024 ** 3

        profile.memory_tier = "low"
        profile.prefers_contiguous = True

    return profile


@dataclass
class DeviceAwareSearchBounds:
    """Search space bounds adapted to device capabilities."""

    # Window sizes to explore
    window_options: list[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Landmark stride options
    landmark_options: list[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Number of cycles
    num_cycles_range: tuple[int, int] = (1, 2)

    # Batch sizes
    batch_size_options: list[int] = field(default_factory=lambda: [8, 16, 32])

    # Sequence lengths
    seq_len_options: list[int] = field(default_factory=lambda: [128, 256])

    # Whether to use AMP
    use_amp: bool = False

    # Prefer permute-to-cycle-order attention
    prefer_permuted: bool = False


def adapt_search_bounds(profile: DeviceProfile) -> DeviceAwareSearchBounds:
    """Create device-appropriate search bounds.

    Parameters
    ----------
    profile : DeviceProfile
        Hardware profile.

    Returns
    -------
    DeviceAwareSearchBounds
        Adapted search space for TTT-Discover.
    """
    bounds = DeviceAwareSearchBounds()

    if profile.memory_tier == "high":
        bounds.window_options = [16, 32, 64, 128, 256]
        bounds.landmark_options = [16, 32, 64, 128]
        bounds.num_cycles_range = (1, 4)
        bounds.batch_size_options = [16, 32, 64, 128]
        bounds.seq_len_options = [128, 256, 512, 1024]
    elif profile.memory_tier == "medium":
        bounds.window_options = [8, 16, 32, 64, 128]
        bounds.landmark_options = [8, 16, 32, 64]
        bounds.num_cycles_range = (1, 3)
        bounds.batch_size_options = [8, 16, 32, 64]
        bounds.seq_len_options = [128, 256, 512]
    else:  # low
        bounds.window_options = [4, 8, 16, 32]
        bounds.landmark_options = [4, 8, 16, 32]
        bounds.num_cycles_range = (1, 2)
        bounds.batch_size_options = [4, 8, 16]
        bounds.seq_len_options = [64, 128, 256]

    # AMP on CUDA with sufficient compute capability
    bounds.use_amp = profile.supports_bf16 or (
        profile.device_type == "cuda" and profile.supports_flash_attn
    )

    # Apple Silicon prefers contiguous memory patterns
    bounds.prefer_permuted = profile.prefers_contiguous

    return bounds
