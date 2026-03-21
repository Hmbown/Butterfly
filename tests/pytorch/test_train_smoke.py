from __future__ import annotations

import math

import torch

from bna.data import build_datasets, get_batch
from bna.model import GPT, GPTConfig
from bna.tokenizers import CharTokenizer
from bna.utils import set_seed


def _one_step(attn: str) -> float:
    torch.set_num_threads(1)
    set_seed(123)

    # Tiny in-memory dataset (keeps tests fast and offline)
    text = ("To be, or not to be: that is the question.\n" * 64) + ("And by opposing end them.\n" * 64)
    tok = CharTokenizer.from_text(text)
    data = build_datasets(text, tok, val_fraction=0.1)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        seq_len=16,
        n_layers=1,
        n_heads=1,
        n_embd=16,
        dropout=0.0,
        attn=attn,  # dense or hcsa
        cycle="random",
        window=2,
        landmark_stride=4,
        seed=123,
    )

    model = GPT(cfg)
    xb, yb = get_batch(data.train, batch_size=1, seq_len=16, device=torch.device("cpu"))
    out = model(xb, yb)
    loss = out["loss"]
    assert torch.isfinite(loss)

    # Note: We avoid backward() in unit tests to keep CI stable/fast.
    # Full training (with backward) is exercised by the training script.
    return float(loss.item())


def test_train_dense_one_step_smoke():
    loss = _one_step("dense")
    assert math.isfinite(loss)


def test_train_hcsa_one_step_smoke():
    loss = _one_step("hcsa")
    assert math.isfinite(loss)
