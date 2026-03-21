# Wayfinder Block-Sparse Attention

This note describes the new staged `block_sparse_topology=wayfinder` path used by the Qwen CUDA integration.

## Goal

The objective is to replace irregular or routing-heavy sparse attention with a fixed communication network that is:

- causal
- block-structured
- compile-time predictable
- KV-cache-aware
- low-degree per block
- globally mixing across layers through a deterministic stage schedule

## Topology

The sequence is partitioned into fixed-size blocks (`block_size`, typically `64` or `128` tokens). For each query block, the support is:

1. the query block itself
2. a fixed number of immediately preceding local blocks (`block_local_window_blocks`)
3. one or more deterministic long-range partner blocks (`block_partner_count`)
4. a small fixed set of sink blocks (`block_sink_blocks`)

The partner blocks are stage-dependent. For a layer `l`, the stage is derived from the layer index and the number of active block stages:

- `xor` / `bit_reversal`: `stage = l mod ceil(log2(num_blocks))`
- `benes`: `stage = l mod (2 * ceil(log2(num_blocks)) - 2)`

The current implementation supports partner rules:

- `xor`
- `bit_reversal`
- `benes`

All support is filtered causally at token level.

## Execution Paths

### Prefill

Prefill uses the existing flex-attention block mask path. The block mask is compiled from the static Wayfinder block layout and reused through the shared cache.

### Cached Decode / Cached Prefill

When `k_len > q_len`, the Wayfinder block path switches to a sparse gather backend. It precomputes exact token indices from the selected block support and runs exact softmax over that sparse support via grouped-query sparse attention.

This is what removes the old “prefill-only” limitation for the new Wayfinder block topology.

## Qwen Integration Surface

`bna/integrations/qwen_torch.py` exposes the following knobs:

- `path="block_sparse"`
- `block_size`
- `block_local_window_blocks`
- `block_partner_count`
- `block_sink_blocks`
- `block_partner_rule="xor" | "bit_reversal" | "benes"`

The block_sparse path always uses the Wayfinder staged topology. The older Hamiltonian random-graph mechanism has been archived.

## Benchmark CLI

The CUDA benchmark and serve entrypoints expose the same Wayfinder block controls:

```bash
python scripts/bench_qwen35_cuda_wayfinder.py \
  --model-path /home/hmbown/HF_Models/Qwen3.5-9B \
  --path block_sparse \
  --block-size 128 \
  --block-local-window-blocks 1 \
  --block-partner-count 1 \
  --block-sink-blocks 1 \
  --block-partner-rule xor \
  --forward-target backbone \
  --seq-lens 512 4096
```

## Status

- Implementation cutover: complete
- Targeted compile/test validation: complete
- Real-model benchmark evidence for the new Wayfinder block topology: not run yet

Until real measurements exist, treat this as an implementation-ready experimental path rather than a performance claim.
