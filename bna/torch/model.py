from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bna.torch.attention_dense import DenseCausalAttentionTorch
from bna.torch.attention_wayfinder_sparse import WayfinderAttentionTorch


AttnMode = Literal["dense", "wayfinder_sparse", "wayfinder_permute"]


@dataclass
class GPTConfigTorch:
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
    compiled_graph_dir: Optional[str] = None


class MLPTorch(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class BlockTorch(nn.Module):
    def __init__(self, cfg: GPTConfigTorch, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        if cfg.attn == "dense":
            self.attn = DenseCausalAttentionTorch(cfg.n_embd, cfg.n_heads, dropout=cfg.dropout)
        elif cfg.attn in ("wayfinder_sparse", "wayfinder_permute"):
            path = "sparse" if cfg.attn == "wayfinder_sparse" else "permute"
            self.attn = WayfinderAttentionTorch(
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

        self.mlp = MLPTorch(cfg.n_embd, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, *, return_debug: bool = False):
        if return_debug:
            attn_out, attn_dbg = self.attn(self.ln1(x), return_debug=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, {"attn": attn_dbg}

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTorch(nn.Module):
    def __init__(self, cfg: GPTConfigTorch):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([BlockTorch(cfg, i) for i in range(cfg.n_layers)])
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
            if isinstance(attn, WayfinderAttentionTorch):
                attn.set_runtime_controls(window_drop=window_drop, schedule_bias=schedule_bias)

    def clear_wayfinder_runtime_controls(self) -> None:
        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, WayfinderAttentionTorch):
                attn.clear_runtime_controls()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        _b, t = idx.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"T={t} exceeds seq_len={self.cfg.seq_len}")

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(t, dtype=torch.long, device=idx.device))
        x = self.drop(tok + pos)

        block_debug = []
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
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            out["loss"] = loss

        if return_debug:
            out["debug"] = {"blocks": block_debug}

        return out

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / max(1e-8, float(temperature))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
