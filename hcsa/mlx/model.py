from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from hcsa.mlx.attention import WayfinderAttentionMLX, dense_causal_attention


AttnMode = Literal["dense", "wayfinder_sparse", "wayfinder_permute"]


@dataclass
class GPTConfigMLX:
    vocab_size: int = 256
    seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    attn: AttnMode = "dense"

    # Wayfinder-specific
    strategy: str = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int = 1
    routing_dim: Optional[int] = None
    seed: int = 0
    window_drop: float = 0.0
    edge_bias: bool = False
    graph_spec: Optional[str] = None
    compiled_graph_dir: Optional[str] = None


class DenseCausalAttentionMLX(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
        self.n_embd = int(n_embd)
        self.n_heads = int(n_heads)
        self.head_dim = self.n_embd // self.n_heads

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, *, return_debug: bool = False):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        y_h, w = dense_causal_attention(q, k, v, return_weights=return_debug)
        y = y_h.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.out(y)
        y = self.dropout(y)

        if return_debug:
            return y, {"attn_weights": w}
        return y


class MLPMLX(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        x = nn.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class BlockMLX(nn.Module):
    def __init__(self, cfg: GPTConfigMLX, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        if cfg.attn == "dense":
            self.attn = DenseCausalAttentionMLX(cfg.n_embd, cfg.n_heads, dropout=cfg.dropout)
        elif cfg.attn in ("wayfinder_sparse", "wayfinder_permute"):
            path = "sparse" if cfg.attn == "wayfinder_sparse" else "permute"
            self.attn = WayfinderAttentionMLX(
                cfg.n_embd,
                cfg.n_heads,
                routing_dim=cfg.routing_dim,
                dropout=cfg.dropout,
                window=cfg.window,
                landmark_stride=cfg.landmark_stride,
                strategy=cfg.strategy,
                num_cycles=cfg.num_cycles,
                seed=cfg.seed + 1337 * layer_idx,
                path=path,
                edge_bias=cfg.edge_bias,
                window_drop=cfg.window_drop,
                compiled_graph_dir=cfg.compiled_graph_dir,
            )
        else:
            raise ValueError(f"Unknown attention type: {cfg.attn}")

        self.mlp = MLPMLX(cfg.n_embd, dropout=cfg.dropout)

    def __call__(self, x: mx.array, *, return_debug: bool = False):
        if return_debug:
            attn_out, attn_dbg = self.attn(self.ln1(x), return_debug=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, {"attn": attn_dbg}

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTMLX(nn.Module):
    def __init__(self, cfg: GPTConfigMLX):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = [BlockMLX(cfg, i) for i in range(cfg.n_layers)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def set_wayfinder_runtime_controls(
        self,
        *,
        window_drop: Optional[float] = None,
        schedule_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, WayfinderAttentionMLX):
                attn.set_runtime_controls(window_drop=window_drop, schedule_bias=schedule_bias)

    def clear_wayfinder_runtime_controls(self) -> None:
        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, WayfinderAttentionMLX):
                attn.clear_runtime_controls()

    def __call__(
        self,
        idx: mx.array,
        targets: Optional[mx.array] = None,
        *,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        _B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"T={T} exceeds seq_len={self.cfg.seq_len}")

        tok = self.token_emb(idx)
        pos = self.pos_emb(mx.arange(T, dtype=mx.int32))
        x = self.drop(tok + pos)

        block_debug: List[Dict[str, Any]] = []
        for block in self.blocks:
            if return_debug:
                x, dbg = block(x, return_debug=True)
                block_debug.append(dbg)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, Any] = {"logits": logits}
        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
            out["loss"] = loss

        if return_debug:
            out["debug"] = {"blocks": block_debug}

        return out

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> mx.array:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / max(1e-8, float(temperature))
            probs = mx.softmax(logits, axis=-1)
            next_id = mx.random.categorical(mx.log(probs + 1e-8))
            next_id = next_id.reshape(-1, 1)
            idx = mx.concatenate([idx, next_id], axis=1)
        return idx
