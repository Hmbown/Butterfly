"""Integration tests: train -> checkpoint -> load -> verify -> generate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from hcsa.data import build_datasets, get_batch
from hcsa.model import GPT, GPTConfig
from hcsa.tokenizers import CharTokenizer, tokenizer_from_state_dict
from hcsa.utils import set_seed, save_json, load_json


def _make_tiny_setup():
    """Create tiny model, data, and tokenizer for integration testing."""
    set_seed(42)
    text = "Hello world! " * 100
    tok = CharTokenizer.from_text(text)
    data = build_datasets(text, tok, val_fraction=0.1)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, seq_len=16, n_layers=1, n_heads=1,
        n_embd=16, attn="dense", seed=42,
    )
    return cfg, tok, data


def test_checkpoint_save_load_roundtrip() -> None:
    """Train a few steps, save checkpoint, load, verify identical."""
    cfg, tok, data = _make_tiny_setup()
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train a few steps
    model.train()
    for _ in range(3):
        xb, yb = get_batch(data.train, 2, cfg.seq_len, torch.device("cpu"))
        out = model(xb, yb)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        from dataclasses import asdict
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": asdict(cfg),
            "tokenizer": tok.state_dict(),
            "step": 3,
        }, ckpt_path)

        # Load
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg2 = GPTConfig(**ckpt["cfg"])
        model2 = GPT(cfg2)
        model2.load_state_dict(ckpt["model"], strict=True)

        # Verify identical outputs
        model.eval()
        model2.eval()
        xb, yb = get_batch(data.train, 2, cfg.seq_len, torch.device("cpu"))
        with torch.no_grad():
            out1 = model(xb)["logits"]
            out2 = model2(xb)["logits"]
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model differs from original"


def test_tokenizer_roundtrip() -> None:
    """Tokenizer state_dict -> from_state_dict preserves encode/decode."""
    text = "Hello world! Test 123."
    tok = CharTokenizer.from_text(text)
    state = tok.state_dict()

    tok2 = tokenizer_from_state_dict(state)
    assert tok.encode(text) == tok2.encode(text)
    assert tok.decode(tok.encode(text)) == tok2.decode(tok2.encode(text))


def test_generate_produces_valid_tokens() -> None:
    """Generation should produce valid token indices."""
    cfg, tok, data = _make_tiny_setup()
    model = GPT(cfg)
    model.eval()

    prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

    assert generated.shape[1] == 13  # 3 prompt + 10 generated
    assert (generated >= 0).all()
    assert (generated < cfg.vocab_size).all()


def test_config_json_roundtrip() -> None:
    """Config save/load via JSON preserves values."""
    from dataclasses import asdict
    cfg = GPTConfig(vocab_size=100, seq_len=64, n_layers=2, n_heads=2, n_embd=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.json"
        save_json(path, {"model": asdict(cfg)})
        loaded = load_json(path)
        cfg2 = GPTConfig(**loaded["model"])
        assert asdict(cfg) == asdict(cfg2)
