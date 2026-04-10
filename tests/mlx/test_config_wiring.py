"""Tests that circular and multi_cycle_mode config flags wire through Butterfly modules."""
import mlx.core as mx

from bna.mlx.attention import ButterflyAttentionMLX
from bna.mlx.model import GPTConfigMLX, GPTMLX


class TestButterflyAttentionMLXWiring:
    """ButterflyAttentionMLX stores and forwards circular/multi_cycle_mode."""

    def test_defaults(self):
        attn = ButterflyAttentionMLX(n_embd=32, n_heads=2, window=4)
        assert attn.circular is False
        assert attn.multi_cycle_mode == "average"

    def test_circular_true_stored(self):
        attn = ButterflyAttentionMLX(
            n_embd=32, n_heads=2, window=4, circular=True,
        )
        assert attn.circular is True

    def test_multi_cycle_mode_union_stored(self):
        attn = ButterflyAttentionMLX(
            n_embd=32, n_heads=2, window=4, multi_cycle_mode="union",
        )
        assert attn.multi_cycle_mode == "union"


class TestGPTConfigMLXWiring:
    """GPTConfigMLX passes circular/multi_cycle_mode through to attention."""

    def test_config_defaults(self):
        cfg = GPTConfigMLX()
        assert cfg.circular is False
        assert cfg.multi_cycle_mode == "average"

    def test_config_circular_propagates(self):
        cfg = GPTConfigMLX(
            n_embd=32, n_heads=2, n_layers=1, seq_len=16,
            attn="butterfly_permute", window=4, circular=True,
        )
        model = GPTMLX(cfg)
        block = model.blocks[0]
        assert isinstance(block.attn, ButterflyAttentionMLX)
        assert block.attn.circular is True

    def test_config_multi_cycle_mode_propagates(self):
        cfg = GPTConfigMLX(
            n_embd=32, n_heads=2, n_layers=1, seq_len=16,
            attn="butterfly_permute", window=4,
            multi_cycle_mode="union", num_cycles=2,
        )
        model = GPTMLX(cfg)
        block = model.blocks[0]
        assert isinstance(block.attn, ButterflyAttentionMLX)
        assert block.attn.multi_cycle_mode == "union"


class TestCircularForwardPass:
    """Circular=True produces valid output through full model."""

    def test_forward_with_circular(self):
        cfg = GPTConfigMLX(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1,
            seq_len=16, attn="butterfly_permute", window=4,
            circular=True,
        )
        model = GPTMLX(cfg)
        idx = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        out = model(idx)
        logits = out["logits"]
        assert logits.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(logits)).item()

    def test_forward_with_union_multicycle(self):
        cfg = GPTConfigMLX(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1,
            seq_len=16, attn="butterfly_permute", window=4,
            num_cycles=2, multi_cycle_mode="union",
        )
        model = GPTMLX(cfg)
        idx = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        out = model(idx)
        logits = out["logits"]
        assert logits.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(logits)).item()

    def test_circular_union_combined(self):
        cfg = GPTConfigMLX(
            vocab_size=32, n_embd=32, n_heads=2, n_layers=1,
            seq_len=16, attn="butterfly_permute", window=4,
            num_cycles=2, circular=True, multi_cycle_mode="union",
        )
        model = GPTMLX(cfg)
        idx = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        out = model(idx)
        logits = out["logits"]
        assert logits.shape == (1, 16, 32)
        assert not mx.any(mx.isnan(logits)).item()
