from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_dense import DenseCausalSelfAttention
from .attention_hcsa import HCSASelfAttention


AttnType = Literal["dense", "hcsa"]


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    attn: AttnType = "dense"

    # HCSA-specific
    cycle: str = "random"
    window: int = 64
    landmark_stride: int | None = 64
    num_cycles: int = 1
    routing_dim: int | None = None
    seed: int = 0


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        if cfg.attn == "dense":
            self.attn = DenseCausalSelfAttention(cfg.n_embd, cfg.n_heads, dropout=cfg.dropout)
        elif cfg.attn == "hcsa":
            self.attn = HCSASelfAttention(
                cfg.n_embd,
                cfg.n_heads,
                routing_dim=cfg.routing_dim,
                dropout=cfg.dropout,
                window=cfg.window,
                landmark_stride=cfg.landmark_stride,
                cycle=cfg.cycle,
                num_cycles=cfg.num_cycles,
                seed=cfg.seed + 1337 * layer_idx,
            )
        else:
            raise ValueError(f"Unknown attention type: {cfg.attn}")

        self.mlp = MLP(cfg.n_embd, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            idx: [B,T] token indices
            targets: optional [B,T]

        Returns dict with 'logits' and optionally 'loss'.
        """
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"Sequence length T={T} exceeds model cfg.seq_len={self.cfg.seq_len}")

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / max(1e-8, float(temperature))

            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
