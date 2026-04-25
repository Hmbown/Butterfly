from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from bna.mlx.attention import _online_softmax_merge  # noqa: E402


def test_online_softmax_merge_two_streams_matches_dense_softmax() -> None:
    rng = np.random.default_rng(7)
    B, H, Tq, dh = 1, 2, 4, 8
    Ka, Kb = 5, 7
    q = rng.standard_normal((B, H, Tq, dh)).astype(np.float32)
    ka = rng.standard_normal((B, H, Ka, dh)).astype(np.float32)
    va = rng.standard_normal((B, H, Ka, dh)).astype(np.float32)
    kb = rng.standard_normal((B, H, Kb, dh)).astype(np.float32)
    vb = rng.standard_normal((B, H, Kb, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    sa = (q @ ka.transpose(0, 1, 3, 2)) * scale  # [B,H,Tq,Ka]
    sb = (q @ kb.transpose(0, 1, 3, 2)) * scale  # [B,H,Tq,Kb]
    ma = sa.max(axis=-1, keepdims=True)
    mb = sb.max(axis=-1, keepdims=True)
    la = np.exp(sa - ma).sum(axis=-1, keepdims=True)
    lb = np.exp(sb - mb).sum(axis=-1, keepdims=True)
    oa = (np.exp(sa - ma) @ va) / la  # [B,H,Tq,dh]
    ob = (np.exp(sb - mb) @ vb) / lb  # [B,H,Tq,dh]

    out_mx = _online_softmax_merge(
        (mx.array(oa), mx.array(la), mx.array(ma)),
        (mx.array(ob), mx.array(lb), mx.array(mb)),
    )
    mx.eval(out_mx)

    s_full = np.concatenate([sa, sb], axis=-1)
    v_full = np.concatenate([va, vb], axis=-2)
    w = np.exp(s_full - s_full.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    ref = w @ v_full  # [B,H,Tq,dh]

    assert np.allclose(np.asarray(out_mx, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)


def test_online_softmax_merge_three_way_via_associativity() -> None:
    """Merging three streams (A,(B,C)) must equal merging ((A,B),C)."""
    rng = np.random.default_rng(11)
    B, H, Tq, dh = 1, 2, 4, 8
    Ka, Kb, Kc = 3, 5, 4
    q = rng.standard_normal((B, H, Tq, dh)).astype(np.float32)
    ka, va = rng.standard_normal((B, H, Ka, dh)).astype(np.float32), rng.standard_normal((B, H, Ka, dh)).astype(np.float32)
    kb, vb = rng.standard_normal((B, H, Kb, dh)).astype(np.float32), rng.standard_normal((B, H, Kb, dh)).astype(np.float32)
    kc, vc = rng.standard_normal((B, H, Kc, dh)).astype(np.float32), rng.standard_normal((B, H, Kc, dh)).astype(np.float32)
    scale = 1.0 / math.sqrt(dh)

    def stream(k, v):
        s = (q @ k.transpose(0, 1, 3, 2)) * scale
        m = s.max(axis=-1, keepdims=True)
        l = np.exp(s - m).sum(axis=-1, keepdims=True)
        o = (np.exp(s - m) @ v) / l
        return mx.array(o), mx.array(l), mx.array(m)

    a, b, c = stream(ka, va), stream(kb, vb), stream(kc, vc)

    # Reference: full softmax over all three concatenated.
    sa_full = np.concatenate(
        [
            (q @ ka.transpose(0, 1, 3, 2)) * scale,
            (q @ kb.transpose(0, 1, 3, 2)) * scale,
            (q @ kc.transpose(0, 1, 3, 2)) * scale,
        ],
        axis=-1,
    )
    v_full = np.concatenate([va, vb, vc], axis=-2)
    w_full = np.exp(sa_full - sa_full.max(axis=-1, keepdims=True))
    w_full = w_full / w_full.sum(axis=-1, keepdims=True)
    ref = w_full @ v_full

    # Sequential merge: ((a, b), c) — but _online_softmax_merge returns just `out`,
    # so we need a variant that returns (out, l, m) for chaining.  Verify final
    # output matches full reference via the public API by computing ((a, b)) first
    # then merging with c using the same _online_softmax_merge primitive.
    # For this we need a helper that packs the merged (out, l, m). Define inline.
    o_a, l_a, m_a = a
    o_b, l_b, m_b = b
    m_ab = mx.maximum(m_a, m_b)
    coeff_a = mx.exp(m_a - m_ab)
    coeff_b = mx.exp(m_b - m_ab)
    l_ab = coeff_a * l_a + coeff_b * l_b
    o_ab = (coeff_a * l_a * o_a + coeff_b * l_b * o_b) / l_ab
    out_final = _online_softmax_merge((o_ab, l_ab, m_ab), c)
    mx.eval(out_final)

    assert np.allclose(np.asarray(out_final, dtype=np.float32), ref, atol=3e-4, rtol=3e-4)
