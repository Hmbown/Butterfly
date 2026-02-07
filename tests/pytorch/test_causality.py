import torch

from hcsa.attention_hcsa import HCSASelfAttention


def test_hcsa_masks_future_neighbors():
    torch.manual_seed(0)
    B, T, C, H = 1, 16, 16, 1
    x = torch.randn(B, T, C)
    attn = HCSASelfAttention(C, H, cycle="random", window=2, landmark_stride=4, seed=123)

    _, dbg = attn(x, return_debug=True)
    neigh = dbg["neigh_idx"]  # [T,D] on CPU
    causal_ok = dbg["causal_ok"]
    valid = dbg["valid_mask"]

    assert neigh.ndim == 2
    # For every valid neighbor j, if j>i it must be masked out.
    for i in range(T):
        for d in range(neigh.shape[1]):
            j = int(neigh[i, d])
            if j < 0:
                continue
            if j > i:
                assert bool(causal_ok[i, d]) is False
            else:
                # valid entries that are not future should be causally allowed
                assert bool(causal_ok[i, d]) is True
            assert bool(valid[i, d]) is True
