"""TTT-Discover evaluation loop.

For each proposed config:
1. Instantiate model
2. Train for N steps
3. Measure reward (throughput * quality)
4. Update buffer

Supports parallel evaluation via process pool.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .environment import TTTConfig, TTTEnvironment, TTTAction
from .reward import compute_reward, RewardConfig
from .buffer import ReplayBuffer
from .proposer import ConfigProposer

from hcsa.model import GPT, GPTConfig
from hcsa.data import build_datasets, get_batch
from hcsa.train import estimate_loss
from hcsa.utils import (
    auto_device,
    format_bytes,
    peak_memory_bytes,
    reset_peak_memory_stats,
    set_seed,
)


def _ttt_config_to_gpt_config(ttt_cfg: TTTConfig, vocab_size: int) -> GPTConfig:
    """Convert TTT config to GPT model config."""
    landmark = ttt_cfg.graph.landmark_stride
    if landmark <= 0:
        landmark = None
    return GPTConfig(
        vocab_size=vocab_size,
        seq_len=ttt_cfg.model.seq_len,
        n_layers=ttt_cfg.model.n_layers,
        n_heads=ttt_cfg.model.n_heads,
        n_embd=ttt_cfg.model.n_embd,
        dropout=ttt_cfg.model.dropout,
        attn="hcsa",
        cycle=ttt_cfg.graph.cycle,
        window=ttt_cfg.graph.window,
        landmark_stride=landmark,
        num_cycles=ttt_cfg.graph.num_cycles,
    )


def evaluate_config(
    ttt_cfg: TTTConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single TTT configuration.

    Returns metrics dict with tokens/sec, val_ppl, peak_memory, etc.
    """
    import math

    cfg = _ttt_config_to_gpt_config(ttt_cfg, vocab_size)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=ttt_cfg.lr,
        weight_decay=ttt_cfg.weight_decay,
    )

    reset_peak_memory_stats(device)
    model.train()
    t0 = time.perf_counter()
    all_finite = True

    for step in range(1, ttt_cfg.eval_steps + 1):
        xb, yb = get_batch(train_data, ttt_cfg.batch_size, cfg.seq_len, device)
        out = model(xb, yb)
        loss = out["loss"]

        if not torch.isfinite(loss):
            all_finite = False
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ttt_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), ttt_cfg.grad_clip)
        opt.step()

    elapsed = time.perf_counter() - t0
    total_tokens = ttt_cfg.eval_steps * ttt_cfg.batch_size * cfg.seq_len
    tokens_per_sec = total_tokens / max(elapsed, 1e-8)

    # Evaluation
    model.eval()
    val_loss = estimate_loss(
        model, val_data, batch_size=ttt_cfg.batch_size,
        seq_len=cfg.seq_len, device=device,
    )
    val_ppl = math.exp(min(val_loss, 20.0))

    mem = peak_memory_bytes(device)

    return {
        "tokens_per_sec": tokens_per_sec,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "peak_memory_bytes": mem,
        "peak_memory": format_bytes(mem),
        "n_params": n_params,
        "elapsed_s": elapsed,
        "loss_finite": all_finite,
    }


class TTTLoop:
    """Main TTT-Discover evaluation loop.

    Parameters
    ----------
    train_data, val_data : Tensor
        Token streams for training and validation.
    vocab_size : int
        Vocabulary size.
    device : torch.device
        Device for training.
    reward_config : RewardConfig
        Reward function configuration.
    n_iterations : int
        Number of TTT iterations.
    """

    def __init__(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        vocab_size: int,
        device: torch.device,
        reward_config: Optional[RewardConfig] = None,
        n_iterations: int = 20,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.device = device
        self.reward_config = reward_config or RewardConfig()
        self.n_iterations = n_iterations

        self.env = TTTEnvironment()
        self.buffer = ReplayBuffer()
        self.proposer = ConfigProposer()

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full TTT-Discover loop."""
        for i in range(self.n_iterations):
            # Propose config
            config = self.proposer.propose(self.buffer)

            if verbose:
                g = config.graph
                print(
                    f"[{i+1}/{self.n_iterations}] "
                    f"cycle={g.cycle} w={g.window} lm={g.landmark_stride} "
                    f"nc={g.num_cycles}"
                )

            # Evaluate
            metrics = evaluate_config(
                config, self.train_data, self.val_data,
                self.vocab_size, self.device,
            )

            # Compute reward
            reward = compute_reward(
                tokens_per_sec=metrics["tokens_per_sec"],
                val_ppl=metrics["val_ppl"],
                peak_memory_bytes=metrics["peak_memory_bytes"],
                loss_is_finite=metrics["loss_finite"],
                config=self.reward_config,
            )

            if verbose:
                print(
                    f"  ppl={metrics['val_ppl']:.1f} "
                    f"tok/s={metrics['tokens_per_sec']:,.0f} "
                    f"reward={reward:.4f}"
                )

            # Update
            action = TTTAction(proposed_config=config)
            self.env.step(action, reward, metrics)
            self.buffer.add(config, reward, metrics)

        # Results
        best = self.env.state.best_config
        return {
            "best_config": best.to_dict() if best else None,
            "best_reward": self.env.state.best_reward,
            "history": self.env.state.history,
            "buffer_size": len(self.buffer),
        }
