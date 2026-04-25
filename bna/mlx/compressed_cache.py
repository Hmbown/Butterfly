from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask


class CompressedKVCache:
    """Two-pool KV cache for compressed Butterfly attention.

    Maintains two pre-allocated, slice-updated buffers (matching DeepSeek V4's
    "state cache + classical cache" layout, paper §3.6.1, Figure 6):

    * **Tail buffer** ``[B, H, tail_capacity, dh]`` — holds the most recent
      raw tokens kept around for the SWA stream. Slide-and-trim is done in
      place via `mx.slice_update` rather than `mx.concatenate`, so the lazy
      graph does not accumulate a chain of intermediate tensors.
    * **Summary buffer** ``[B, H, summary_capacity, dh]`` — holds mean-pooled
      summary vectors for completed compression blocks. One slot is written
      per completed block.

    Logic invariants are unchanged from the prior `mx.concatenate` version:
        * `summary_offset` advances when a block fully predates the SWA
          window (`summary_offset + block_size <= offset - local_window_tokens`).
        * Tail size is trimmed when it exceeds the working window.
        * Summary and tail can overlap (a recently summarized block may still
          be in the tail); attention's routed-indices logic filters that.

    Public attribute access (`keys`, `values`, `k_summary`, `v_summary`,
    `state`, `meta_state`, `nbytes`, `make_mask`) preserves the original API
    so `mlx_lm.utils.maybe_quantize_kv_cache` and the bench harness keep
    working unchanged.

    Args:
        block_size: Compression block size.
        local_window_tokens: SWA window for the recent raw tokens.
        max_kv_size: If known, sizes the summary buffer exactly. Else uses a
            conservative default of 2048 summary blocks.
        max_chunk_size: Largest single chunk fed to `update_and_fetch`. Used
            to size the tail buffer. Default 512.
    """

    def __init__(
        self,
        *,
        block_size: int = 128,
        local_window_tokens: int = 128,
        max_kv_size: Optional[int] = None,
        max_chunk_size: int = 512,
    ) -> None:
        self.block_size = int(block_size)
        self.local_window_tokens = int(local_window_tokens)
        self.max_kv_size = int(max_kv_size) if max_kv_size else None
        self.max_chunk_size = int(max_chunk_size)

        self._tail_buf_keys: Optional[mx.array] = None
        self._tail_buf_values: Optional[mx.array] = None
        self._summary_buf_keys: Optional[mx.array] = None
        self._summary_buf_values: Optional[mx.array] = None

        self.tail_capacity = 0
        self.summary_capacity = 0
        self.tail_size = 0
        self.summary_count = 0
        self.offset = 0
        self.summary_offset = 0

    def _allocate(self, B: int, H: int, dh: int, dtype: Any) -> None:
        # Tail size can transiently reach roughly
        # `previous_trimmed_tail + step` where the trimmed tail can be up to
        # `target_tail + block_size - 1 = (step + local_window - 1) + block_size - 1`.
        # Worst case: `2 * step + local_window + block_size`. Round up.
        self.tail_capacity = (
            int(self.local_window_tokens)
            + 2 * int(self.max_chunk_size)
            + 2 * int(self.block_size)
        )
        if self.max_kv_size is not None:
            self.summary_capacity = (int(self.max_kv_size) // int(self.block_size)) + 4
        else:
            self.summary_capacity = 2048
        self._tail_buf_keys = mx.zeros((B, H, self.tail_capacity, dh), dtype=dtype)
        self._tail_buf_values = mx.zeros((B, H, self.tail_capacity, dh), dtype=dtype)
        self._summary_buf_keys = mx.zeros((B, H, self.summary_capacity, dh), dtype=dtype)
        self._summary_buf_values = mx.zeros((B, H, self.summary_capacity, dh), dtype=dtype)
        mx.eval(
            self._tail_buf_keys,
            self._tail_buf_values,
            self._summary_buf_keys,
            self._summary_buf_values,
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        step = int(keys.shape[2])
        B = int(keys.shape[0])
        H = int(keys.shape[1])
        dh = int(keys.shape[3])
        if self._tail_buf_keys is None:
            self._allocate(B, H, dh, keys.dtype)
        bs = int(self.block_size)
        lw = int(self.local_window_tokens)

        # Append new tokens at [tail_size : tail_size + step] of the tail buffer.
        if self.tail_size + step > self.tail_capacity:
            raise RuntimeError(
                f"CompressedKVCache.tail_capacity={self.tail_capacity} too small to fit "
                f"tail_size={self.tail_size} + step={step}; raise max_chunk_size or "
                f"local_window_tokens to make room."
            )
        self._tail_buf_keys = mx.slice_update(
            self._tail_buf_keys, keys, mx.array([self.tail_size], dtype=mx.int32), axes=[2]
        )
        self._tail_buf_values = mx.slice_update(
            self._tail_buf_values, values, mx.array([self.tail_size], dtype=mx.int32), axes=[2]
        )
        self.tail_size += step
        self.offset += step

        # Summarize blocks whose end is older than the SWA window. The block
        # offset within the tail buffer is `summary_offset - tail_start`, where
        # `tail_start = offset - tail_size`.
        while self.summary_offset + bs <= self.offset - lw:
            tail_start = self.offset - self.tail_size
            block_start = self.summary_offset - tail_start
            if block_start < 0 or block_start + bs > self.tail_size:
                break
            if self.summary_count >= self.summary_capacity:
                raise RuntimeError(
                    f"CompressedKVCache.summary_capacity={self.summary_capacity} exhausted "
                    f"at offset={self.offset}; raise max_kv_size at construction time."
                )
            k_block = self._tail_buf_keys[:, :, block_start : block_start + bs, :]
            v_block = self._tail_buf_values[:, :, block_start : block_start + bs, :]
            k_sum = mx.mean(k_block.astype(mx.float32), axis=2, keepdims=True).astype(
                self._tail_buf_keys.dtype
            )
            v_sum = mx.mean(v_block.astype(mx.float32), axis=2, keepdims=True).astype(
                self._tail_buf_values.dtype
            )
            slot = self.summary_count
            self._summary_buf_keys = mx.slice_update(
                self._summary_buf_keys, k_sum, mx.array([slot], dtype=mx.int32), axes=[2]
            )
            self._summary_buf_values = mx.slice_update(
                self._summary_buf_values, v_sum, mx.array([slot], dtype=mx.int32), axes=[2]
            )
            self.summary_count += 1
            self.summary_offset += bs

        # Trim the tail when oversized: shift tokens [bs : tail_size] down to
        # [0 : tail_size - bs] in place using slice_update.
        target_tail = max(lw, step + lw - 1)
        while self.tail_size - bs >= target_tail:
            keep = self.tail_size - bs
            moved_k = self._tail_buf_keys[:, :, bs : bs + keep, :]
            moved_v = self._tail_buf_values[:, :, bs : bs + keep, :]
            self._tail_buf_keys = mx.slice_update(
                self._tail_buf_keys, moved_k, mx.array([0], dtype=mx.int32), axes=[2]
            )
            self._tail_buf_values = mx.slice_update(
                self._tail_buf_values, moved_v, mx.array([0], dtype=mx.int32), axes=[2]
            )
            self.tail_size = keep

        mx.eval(
            self._tail_buf_keys,
            self._tail_buf_values,
            self._summary_buf_keys,
            self._summary_buf_values,
        )
        return self.keys, self.values

    @property
    def keys(self) -> Optional[mx.array]:
        if self._tail_buf_keys is None:
            return None
        return self._tail_buf_keys[:, :, : int(self.tail_size), :]

    @property
    def values(self) -> Optional[mx.array]:
        if self._tail_buf_values is None:
            return None
        return self._tail_buf_values[:, :, : int(self.tail_size), :]

    @property
    def k_summary(self) -> Optional[mx.array]:
        if self._summary_buf_keys is None:
            return None
        return self._summary_buf_keys[:, :, : int(self.summary_count), :]

    @property
    def v_summary(self) -> Optional[mx.array]:
        if self._summary_buf_values is None:
            return None
        return self._summary_buf_values[:, :, : int(self.summary_count), :]

    def get_compressed_state(self) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int]:
        if self._tail_buf_keys is None:
            raise ValueError("CompressedKVCache is empty")
        tail_start = int(self.offset) - int(self.tail_size)
        return (
            self.keys,
            self.values,
            self.k_summary,
            self.v_summary,
            int(tail_start),
            int(self.offset),
        )

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
        return self._tail_buf_keys is None or int(self.tail_size) == 0

    @property
    def nbytes(self) -> int:
        # Report only the *valid* slice size to keep apples-to-apples vs the
        # standard KV cache's `nbytes` (which similarly reports valid prefix,
        # not pre-allocated capacity).
        total = 0
        if self._tail_buf_keys is not None:
            elem = int(self._tail_buf_keys.dtype.size)
            B, H, _, dh = self._tail_buf_keys.shape
            total += int(B) * int(H) * int(self.tail_size) * int(dh) * elem  # keys
            total += int(B) * int(H) * int(self.tail_size) * int(dh) * elem  # values
            total += int(B) * int(H) * int(self.summary_count) * int(dh) * elem  # k_summary
            total += int(B) * int(H) * int(self.summary_count) * int(dh) * elem  # v_summary
        return total

    @property
    def state(self) -> tuple[Any, Any, Any, Any]:
        return self.keys, self.values, self.k_summary, self.v_summary

    @state.setter
    def state(self, v: tuple[Any, Any, Any, Any]) -> None:
        keys, values, k_summary, v_summary = v
        if keys is None or values is None:
            self._tail_buf_keys = None
            self._tail_buf_values = None
            self.tail_size = 0
        else:
            B, H, T, dh = keys.shape
            if self._tail_buf_keys is None or int(self._tail_buf_keys.shape[2]) < T:
                self._allocate(int(B), int(H), int(dh), keys.dtype)
            self._tail_buf_keys = mx.slice_update(
                self._tail_buf_keys, keys, mx.array([0], dtype=mx.int32), axes=[2]
            )
            self._tail_buf_values = mx.slice_update(
                self._tail_buf_values, values, mx.array([0], dtype=mx.int32), axes=[2]
            )
            self.tail_size = int(T)
        if k_summary is None or v_summary is None:
            self.summary_count = 0
        else:
            B, H, S, dh = k_summary.shape
            if self._summary_buf_keys is None or int(self._summary_buf_keys.shape[2]) < S:
                self._summary_buf_keys = mx.zeros(
                    (B, H, max(S, self.summary_capacity), dh), dtype=k_summary.dtype
                )
                self._summary_buf_values = mx.zeros(
                    (B, H, max(S, self.summary_capacity), dh), dtype=v_summary.dtype
                )
                self.summary_capacity = int(self._summary_buf_keys.shape[2])
            self._summary_buf_keys = mx.slice_update(
                self._summary_buf_keys,
                k_summary,
                mx.array([0], dtype=mx.int32),
                axes=[2],
            )
            self._summary_buf_values = mx.slice_update(
                self._summary_buf_values,
                v_summary,
                mx.array([0], dtype=mx.int32),
                axes=[2],
            )
            self.summary_count = int(S)

    @property
    def meta_state(self) -> tuple[str, str, str, str]:
        return tuple(
            map(str, (self.block_size, self.local_window_tokens, self.offset, self.summary_offset))
        )

    @meta_state.setter
    def meta_state(self, v: tuple[str, str, str, str]) -> None:
        self.block_size, self.local_window_tokens, self.offset, self.summary_offset = map(int, v)
