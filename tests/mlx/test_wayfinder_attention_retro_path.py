from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import WayfinderAttentionMLX
from bna.mlx.model import GPTConfigMLX, GPTMLX


def _make_attn() -> WayfinderAttentionMLX:
    return WayfinderAttentionMLX(
        n_embd=32,
        n_heads=2,
        window=4,
        landmark_stride=8,
        strategy="random",
        path="permute",
        seed=42,
        dropout=0.0,
    )


def test_permute_retro_training_only_guard_in_eval_mode() -> None:
    """Retro should stay off in eval mode when training_only=True."""
    mx.random.seed(123)
    x = mx.random.normal((1, 16, 32), dtype=mx.float16)

    attn = _make_attn()
    attn.eval()

    y_base = attn(x)
    mx.eval(y_base)

    attn.retro_backfill_enabled = True
    attn.retro_backfill_alpha = 0.5
    attn.retro_backfill_training_only = True
    attn.retro_backfill_causal_only = False

    y_eval = attn(x)
    mx.eval(y_eval)

    np.testing.assert_allclose(
        np.asarray(y_base, dtype=np.float32),
        np.asarray(y_eval, dtype=np.float32),
        atol=1e-5,
    )


def test_permute_retro_changes_training_output_when_enabled() -> None:
    """Retro should change permute outputs in training when enabled."""
    mx.random.seed(456)
    x = mx.random.normal((1, 16, 32), dtype=mx.float16)

    attn = _make_attn()
    attn.train()

    y_base = attn(x)
    mx.eval(y_base)

    attn.retro_backfill_enabled = True
    attn.retro_backfill_alpha = 0.3
    attn.retro_backfill_training_only = True
    attn.retro_backfill_causal_only = False

    y_retro = attn(x)
    mx.eval(y_retro)

    diff = np.max(np.abs(np.asarray(y_retro, dtype=np.float32) - np.asarray(y_base, dtype=np.float32)))
    assert diff > 1e-6


def test_gpt_config_wires_retro_to_wayfinder_attention() -> None:
    cfg = GPTConfigMLX(
        vocab_size=256,
        seq_len=32,
        n_layers=1,
        n_heads=2,
        n_embd=32,
        attn="wayfinder_permute",
        window=4,
        landmark_stride=8,
        retro_backfill_enabled=True,
        retro_backfill_alpha=0.2,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
    )
    model = GPTMLX(cfg)
    attn = model.blocks[0].attn

    assert isinstance(attn, WayfinderAttentionMLX)
    assert attn.retro_backfill_enabled is True
    assert attn.retro_backfill_alpha == pytest.approx(0.2)
    assert attn.retro_backfill_training_only is True
    assert attn.retro_backfill_causal_only is True
