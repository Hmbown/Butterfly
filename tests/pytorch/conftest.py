from __future__ import annotations

import os
import random

import numpy as np
import pytest

# Make tests faster and less noisy on CPU.
try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass

from hcsa.integrations.qwen_torch import _parse_cuda_arch_token


def _cuda_exact_arch_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if str(os.environ.get("WAYFINDER_TEST_ALLOW_UNSUPPORTED_CUDA", "")).strip() in {"1", "true", "TRUE"}:
        return True
    try:
        raw_cap = torch.cuda.get_device_capability()
        capability = (int(raw_cap[0]), int(raw_cap[1]))
        supported_caps = {
            parsed
            for token in torch.cuda.get_arch_list()
            if (parsed := _parse_cuda_arch_token(token)) is not None
        }
    except Exception:
        return False
    return capability in supported_caps


def _available_devices() -> list["torch.device"]:
    out = [torch.device("cpu")]
    if _cuda_exact_arch_supported():
        out.append(torch.device("cuda"))
    return out


@pytest.fixture(params=_available_devices(), ids=lambda d: d.type)
def device(request) -> "torch.device":
    return request.param


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
