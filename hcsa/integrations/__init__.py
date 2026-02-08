"""Integrations for external model runtimes (MLX, HF, etc.)."""

from .qwen_mlx import (
    QwenWayfinderConfig,
    QwenWayfinderAttention,
    extract_qkv_from_qwen_attention,
    swap_qwen_attention_with_wayfinder,
)
from .gpt2_mlx import (
    GPT2WayfinderConfig,
    GPT2WayfinderAttention,
    extract_qkv_from_gpt2_attention,
    swap_gpt2_attention_with_wayfinder,
)
from .glm_mlx import (
    GLMWayfinderConfig,
    GLMWayfinderAttention,
    extract_qkv_from_glm_attention,
    swap_glm_attention_with_wayfinder,
)

__all__ = [
    "QwenWayfinderConfig",
    "QwenWayfinderAttention",
    "extract_qkv_from_qwen_attention",
    "swap_qwen_attention_with_wayfinder",
    "GPT2WayfinderConfig",
    "GPT2WayfinderAttention",
    "extract_qkv_from_gpt2_attention",
    "swap_gpt2_attention_with_wayfinder",
    "GLMWayfinderConfig",
    "GLMWayfinderAttention",
    "extract_qkv_from_glm_attention",
    "swap_glm_attention_with_wayfinder",
]
