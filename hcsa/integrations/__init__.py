"""Integrations for external model runtimes (MLX, HF, etc.)."""

__all__: list[str] = []

# MLX integrations (Apple Silicon only)
try:
    from .qwen_mlx import (
        QwenWayfinderConfig,
        QwenWayfinderAttention,
        extract_qkv_from_qwen_attention,
        swap_qwen_attention_with_wayfinder,
    )
    __all__ += [
        "QwenWayfinderConfig",
        "QwenWayfinderAttention",
        "extract_qkv_from_qwen_attention",
        "swap_qwen_attention_with_wayfinder",
    ]
except ImportError:
    pass

try:
    from .gpt2_mlx import (
        GPT2WayfinderConfig,
        GPT2WayfinderAttention,
        extract_qkv_from_gpt2_attention,
        swap_gpt2_attention_with_wayfinder,
    )
    __all__ += [
        "GPT2WayfinderConfig",
        "GPT2WayfinderAttention",
        "extract_qkv_from_gpt2_attention",
        "swap_gpt2_attention_with_wayfinder",
    ]
except ImportError:
    pass

try:
    from .glm_mlx import (
        GLMWayfinderConfig,
        GLMWayfinderAttention,
        extract_qkv_from_glm_attention,
        swap_glm_attention_with_wayfinder,
    )
    __all__ += [
        "GLMWayfinderConfig",
        "GLMWayfinderAttention",
        "extract_qkv_from_glm_attention",
        "swap_glm_attention_with_wayfinder",
    ]
except ImportError:
    pass

# PyTorch / CUDA integrations
try:
    from .nemotron_h_torch import (
        NemotronHWayfinderAttention,
        NemotronHWayfinderConfig,
        clear_nemotron_h_wayfinder_runtime_controls,
        extract_qkv_from_nemotron_h_attention,
        iter_nemotron_h_wayfinder_layers,
        set_nemotron_h_wayfinder_runtime_controls,
        swap_nemotron_h_attention_with_wayfinder,
    )
    __all__ += [
        "NemotronHWayfinderConfig",
        "NemotronHWayfinderAttention",
        "extract_qkv_from_nemotron_h_attention",
        "iter_nemotron_h_wayfinder_layers",
        "set_nemotron_h_wayfinder_runtime_controls",
        "clear_nemotron_h_wayfinder_runtime_controls",
        "swap_nemotron_h_attention_with_wayfinder",
    ]
except ImportError:
    pass

try:
    from .qwen_torch import (
        QwenCUDAWayfinderAttention,
        QwenCUDAWayfinderConfig,
        extract_qkv_from_qwen_attention as extract_qkv_from_qwen_attention_cuda,
        iter_qwen_wayfinder_layers as iter_qwen_wayfinder_layers_cuda,
        swap_qwen_attention_with_wayfinder_cuda,
    )
    __all__ += [
        "QwenCUDAWayfinderConfig",
        "QwenCUDAWayfinderAttention",
        "extract_qkv_from_qwen_attention_cuda",
        "iter_qwen_wayfinder_layers_cuda",
        "swap_qwen_attention_with_wayfinder_cuda",
    ]
except ImportError:
    pass
