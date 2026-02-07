"""Integrations for external model runtimes (MLX, HF, etc.)."""

from .qwen_mlx import (
    QwenHHAConfig,
    QwenWayfinderAttention,
    extract_qkv_from_qwen_attention,
    swap_qwen_attention_with_wayfinder,
)

__all__ = [
    "QwenHHAConfig",
    "QwenWayfinderAttention",
    "extract_qkv_from_qwen_attention",
    "swap_qwen_attention_with_wayfinder",
]

