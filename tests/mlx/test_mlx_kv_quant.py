from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402

from bna.integrations.mlx_kv_quant import (  # noqa: E402
    MLXKVQuantizationConfig,
    maybe_quantize_mlx_prompt_cache,
    summarize_mlx_prompt_cache_quantization,
    validate_mlx_kv_quantization_config,
)


def _build_kv_cache(tokens: int) -> KVCache:
    cache = KVCache()
    keys = mx.ones((1, 2, tokens, 64), dtype=mx.float16)
    values = mx.ones((1, 2, tokens, 64), dtype=mx.float16)
    cache.update_and_fetch(keys, values)
    return cache


def test_mlx_kv_quantization_only_converts_eligible_kv_entries() -> None:
    prompt_cache = [_build_kv_cache(4), _build_kv_cache(2), object()]
    config = MLXKVQuantizationConfig(bits=4, group_size=64, quantized_kv_start=4)

    before = summarize_mlx_prompt_cache_quantization(prompt_cache, config=config)
    assert before["full_attention_cache_entries"] == 2
    assert before["eligible_entries"] == 1
    assert before["quantized_entries"] == 0

    after = maybe_quantize_mlx_prompt_cache(prompt_cache, config=config)
    assert after["converted_entries"] == 1
    assert after["quantized_entries"] == 1
    assert isinstance(prompt_cache[0], QuantizedKVCache)
    assert isinstance(prompt_cache[1], KVCache)


def test_validate_mlx_kv_quantization_config_rejects_bad_group_size() -> None:
    config = MLXKVQuantizationConfig(bits=4, group_size=16, quantized_kv_start=0)
    with pytest.raises(ValueError, match="--kv-group-size"):
        validate_mlx_kv_quantization_config(config)
