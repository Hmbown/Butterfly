from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask


class CompressedKVCache:
    def __init__(self, *, block_size: int = 128, local_window_tokens: int = 128):
        self.block_size = int(block_size)
        self.local_window_tokens = int(local_window_tokens)
        self.keys = None
        self.values = None
        self.k_summary = None
        self.v_summary = None
        self.offset = 0
        self.summary_offset = 0

    def _empty_summary_like(self, x: mx.array) -> mx.array:
        return mx.zeros((*x.shape[:2], 0, x.shape[-1]), dtype=x.dtype)

    def _append_summary(self, k_block: mx.array, v_block: mx.array) -> None:
        k_sum = mx.mean(k_block.astype(mx.float32), axis=2, keepdims=True).astype(k_block.dtype)
        v_sum = mx.mean(v_block.astype(mx.float32), axis=2, keepdims=True).astype(v_block.dtype)
        if self.k_summary is None:
            self.k_summary = k_sum
            self.v_summary = v_sum
        else:
            self.k_summary = mx.concatenate([self.k_summary, k_sum], axis=2)
            self.v_summary = mx.concatenate([self.v_summary, v_sum], axis=2)
        self.summary_offset += self.block_size

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        step = int(keys.shape[2])
        if self.keys is None:
            self.keys = keys
            self.values = values
            self.k_summary = self._empty_summary_like(keys)
            self.v_summary = self._empty_summary_like(values)
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += step

        target_tail = max(int(self.local_window_tokens), int(step) + int(self.local_window_tokens) - 1)
        while int(self.summary_offset) + int(self.block_size) <= int(self.offset) - int(self.local_window_tokens):
            tail_start = int(self.offset) - int(self.keys.shape[2])
            block_start = int(self.summary_offset) - tail_start
            if block_start < 0 or block_start + int(self.block_size) > int(self.keys.shape[2]):
                break
            k_block = self.keys[:, :, block_start : block_start + self.block_size, :]
            v_block = self.values[:, :, block_start : block_start + self.block_size, :]
            self._append_summary(k_block, v_block)
        while int(self.keys.shape[2]) - int(self.block_size) >= int(target_tail):
            self.keys = self.keys[:, :, self.block_size :, :]
            self.values = self.values[:, :, self.block_size :, :]
        mx.eval(self.keys, self.values, self.k_summary, self.v_summary)
        return self.keys, self.values

    def get_compressed_state(self) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int]:
        if self.keys is None or self.values is None or self.k_summary is None or self.v_summary is None:
            raise ValueError("CompressedKVCache is empty")
        tail_start = int(self.offset) - int(self.keys.shape[2])
        return self.keys, self.values, self.k_summary, self.v_summary, int(tail_start), int(self.offset)

    def size(self) -> int:
        return int(self.offset)

    def is_trimmable(self) -> bool:
        return False

    def trim(self, n: int) -> int:
        return 0

    def make_mask(self, N: int, window_size: Optional[int] = None, return_array: bool = False):
        offset = int(self.offset)
        if N == 1 and window_size is None:
            return None
        if return_array or N > 1:
            return create_causal_mask(N, offset, window_size=window_size)
        return None

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        total = 0
        for x in (self.keys, self.values, self.k_summary, self.v_summary):
            if x is not None:
                total += int(x.nbytes)
        return total

    @property
    def state(self) -> tuple[Any, Any, Any, Any]:
        return self.keys, self.values, self.k_summary, self.v_summary

    @state.setter
    def state(self, v: tuple[Any, Any, Any, Any]) -> None:
        self.keys, self.values, self.k_summary, self.v_summary = v

    @property
    def meta_state(self) -> tuple[str, str, str, str]:
        return tuple(map(str, (self.block_size, self.local_window_tokens, self.offset, self.summary_offset)))

    @meta_state.setter
    def meta_state(self, v: tuple[str, str, str, str]) -> None:
        self.block_size, self.local_window_tokens, self.offset, self.summary_offset = map(int, v)
