"""Smoke tests for NF4 4-bit Qwen load + butterfly swap.

Inference-only path. These tests require CUDA + bitsandbytes and will skip
on any other host (e.g. the CI Mac that has no GPU).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

if importlib.util.find_spec("bitsandbytes") is None:
    pytest.skip("requires bitsandbytes", allow_module_level=True)

from bna.integrations.qwen_torch import (
    QwenCUDAButterflyAttention,
    load_qwen_4bit_cuda,
)


_SMALL_MODEL_ID = "Qwen/Qwen3-0.6B"


def _count_butterfly_layers(model) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, QwenCUDAButterflyAttention):
            count += 1
    return count


def test_load_qwen4b_nf4_and_butterfly_swap() -> None:
    model, tokenizer = load_qwen_4bit_cuda(
        model_id=_SMALL_MODEL_ID,
        butterfly_config={
            "path": "block_sparse",
            "strategy": "random",
            "block_size": 128,
            "block_local_window_blocks": 2,
            "block_partner_rule": "xor",
            "block_sink_blocks": 1,
        },
    )

    # Any parameter should be on CUDA.
    devices = {p.device.type for p in model.parameters()}
    assert "cuda" in devices, f"expected model on CUDA, got devices={devices}"

    assert _count_butterfly_layers(model) >= 1, (
        "expected at least one QwenCUDAButterflyAttention layer after swap"
    )

    input_ids = torch.tensor(
        [tokenizer.encode("Hello world")], dtype=torch.long, device="cuda"
    )
    input_len = int(input_ids.shape[1])
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=4, do_sample=False
        )
    assert tuple(out.shape) == (1, input_len + 4)


def test_load_qwen4b_nf4_baseline_no_swap() -> None:
    model, tokenizer = load_qwen_4bit_cuda(
        model_id=_SMALL_MODEL_ID,
        swap_butterfly=False,
    )

    assert _count_butterfly_layers(model) == 0, (
        "expected no butterfly-wrapped layers when swap_butterfly=False"
    )

    input_ids = torch.tensor(
        [tokenizer.encode("Hello world")], dtype=torch.long, device="cuda"
    )
    input_len = int(input_ids.shape[1])
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=4, do_sample=False
        )
    assert tuple(out.shape) == (1, input_len + 4)
