from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, MutableSequence, Optional, Sequence

from mlx_lm.models.cache import QuantizedKVCache


SUPPORTED_KV_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)


@dataclass(frozen=True)
class MLXKVQuantizationConfig:
    bits: Optional[int] = None
    group_size: int = 64
    quantized_kv_start: int = 0

    @property
    def enabled(self) -> bool:
        return self.bits is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "bits": (None if self.bits is None else int(self.bits)),
            "group_size": int(self.group_size),
            "quantized_kv_start": int(self.quantized_kv_start),
            "backend": "mlx_lm",
            "applies_to": "full_attention_kv_cache_only",
        }


def validate_mlx_kv_quantization_config(config: MLXKVQuantizationConfig) -> None:
    if not config.enabled:
        return
    bits = int(config.bits or 0)
    if bits <= 0:
        raise ValueError("--kv-bits must be a positive integer when KV quantization is enabled.")
    if int(config.group_size) not in SUPPORTED_KV_GROUP_SIZES:
        raise ValueError(
            "--kv-group-size must be one of "
            f"{list(SUPPORTED_KV_GROUP_SIZES)} for MLX KV quantization."
        )
    if int(config.quantized_kv_start) < 0:
        raise ValueError("--quantized-kv-start must be >= 0.")


def summarize_mlx_prompt_cache_quantization(
    prompt_cache: Sequence[Any],
    *,
    config: MLXKVQuantizationConfig,
) -> Dict[str, Any]:
    full_attention_cache_entries = 0
    eligible_entries = 0
    quantized_entries = 0
    max_offset = 0

    for entry in prompt_cache:
        has_dense_quantizer = hasattr(entry, "to_quantized")
        is_quantized = isinstance(entry, QuantizedKVCache)
        if not has_dense_quantizer and not is_quantized:
            continue
        full_attention_cache_entries += 1
        offset = int(getattr(entry, "offset", 0) or 0)
        max_offset = max(max_offset, offset)
        if config.enabled and offset >= int(config.quantized_kv_start):
            eligible_entries += 1
        if is_quantized:
            quantized_entries += 1

    return {
        **config.to_dict(),
        "full_attention_cache_entries": int(full_attention_cache_entries),
        "eligible_entries": int(eligible_entries),
        "quantized_entries": int(quantized_entries),
        "active": bool(quantized_entries > 0),
        "max_offset": int(max_offset),
    }


def maybe_quantize_mlx_prompt_cache(
    prompt_cache: MutableSequence[Any],
    *,
    config: MLXKVQuantizationConfig,
) -> Dict[str, Any]:
    if not config.enabled:
        return {
            **summarize_mlx_prompt_cache_quantization(prompt_cache, config=config),
            "converted_entries": 0,
        }

    converted_entries = 0
    bits = int(config.bits or 0)
    group_size = int(config.group_size)
    quantized_kv_start = int(config.quantized_kv_start)

    for idx, entry in enumerate(list(prompt_cache)):
        if isinstance(entry, QuantizedKVCache):
            continue
        if not hasattr(entry, "to_quantized"):
            continue
        offset = int(getattr(entry, "offset", 0) or 0)
        if offset < quantized_kv_start:
            continue
        prompt_cache[idx] = entry.to_quantized(group_size=group_size, bits=bits)
        converted_entries += 1

    return {
        **summarize_mlx_prompt_cache_quantization(prompt_cache, config=config),
        "converted_entries": int(converted_entries),
    }
