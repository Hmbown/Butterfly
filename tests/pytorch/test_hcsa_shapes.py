import torch

from hcsa.attention_dense import DenseCausalSelfAttention
from hcsa.attention_hcsa import HCSASelfAttention
from hcsa.model import GPT, GPTConfig


def test_attention_output_shapes_match():
    torch.manual_seed(0)
    B, T, C, H = 1, 8, 16, 1
    x = torch.randn(B, T, C)

    dense = DenseCausalSelfAttention(C, H)
    hcsa = HCSASelfAttention(C, H, cycle="random", window=2, landmark_stride=4)

    y_dense = dense(x)
    y_hcsa = hcsa(x)

    assert y_dense.shape == x.shape
    assert y_hcsa.shape == x.shape


def test_gpt_forward_shapes():
    torch.manual_seed(0)
    cfg_d = GPTConfig(vocab_size=64, seq_len=16, n_layers=1, n_heads=1, n_embd=16, attn="dense")
    cfg_h = GPTConfig(vocab_size=64, seq_len=16, n_layers=1, n_heads=1, n_embd=16, attn="hcsa", cycle="random", window=2, landmark_stride=4)

    m_d = GPT(cfg_d)
    m_h = GPT(cfg_h)

    idx = torch.randint(0, 64, (1, 16))
    out_d = m_d(idx)
    out_h = m_h(idx)

    assert out_d["logits"].shape == (1, 16, 64)
    assert out_h["logits"].shape == (1, 16, 64)
