# Butterfly Overnight Log — 2026-04-25

## Phase 0: Baselines (locked)

Apple Silicon, MLX, Qwen3.5-0.8B-4bit, decode_len=8, repeats=1.

| Variant | prefill_sec | e2e_sec | peak_memory_bytes | cache_after_prefill (bytes) |
|---|---:|---:|---:|---:|
| stock 32768 | 9.472026957999333 | 9.516935833002208 | 1774884158 | 414326784 |
| compressed_butterfly w64 qc64 32768 | 15.240854582996690 | 15.287026457997854 | 6273298171 | 16379904 |

**Floor confirmation gate (15% drift threshold):**
- stock e2e drift vs 9.34s: +1.89%
- compressed e2e drift vs 14.64s: +4.42%
- decision: PROCEED

Pre-flight free memory snapshots:
- before stock: 10.32 GB reclaimable (free=169481 + inactive=472606 + speculative=34011 pages × 16 KiB)
- before compressed: 10.48 GB reclaimable (free=167719 + inactive=487465 + speculative=31576 pages × 16 KiB)

## Phase 1: (pending)
## Phase 2: (pending)
## Phase 3: (pending)
## Phase 4: (pending)
## Phase 5: (pending)
## Phase 6: (pending)
## Phase 7: (pending)
