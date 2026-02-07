from __future__ import annotations

import random

import numpy as np
import pytest
import torch


def _available_devices() -> list[torch.device]:
    out = [torch.device("cpu")]
    if torch.cuda.is_available():
        out.append(torch.device("cuda"))
    return out


@pytest.fixture(params=_available_devices(), ids=lambda d: d.type)
def device(request) -> torch.device:
    return request.param


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
