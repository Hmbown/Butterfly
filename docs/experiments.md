# Experiments & ablations

This repo is intentionally small; the goal is to compare **dense** vs **HCSA** fairly under identical settings.

## Suggested experiment plan

### 1) Sanity check
- Dataset: `data/tinyshakespeare.txt`
- Tokenizer: `char`
- Model: `--layers 4 --heads 4 --embd 256 --seq-len 256`
- Train 1k-5k steps
- Compare:
  - training loss / validation ppl
  - tokens/sec
  - peak memory (best-effort)

### 2) HCSA variants
Keep model and dataset fixed and sweep:
- `--cycle random` vs `--cycle greedy`
- `--window {32,64,128}`
- `--landmark-stride {0,128,64,32}` (0 disables)
- `--num-cycles {1,2,4}` for random/greedy union-of-cycles

### 3) Context scaling
Use `scripts/bench.py` and sweep `T` (128..1024). Expect:
- Dense time/memory grows ~O(T^2)
- HCSA grows closer to ~O(T*D) where D is the sparse degree

## What to expect
- **Random cycles** can help long-range mixing but may be noisy.
- **Greedy cycles** often create semantically coherent long-range edges but cost O(T^2) per forward to build.
- **Landmarks** provide cheap global communication and tend to stabilize training.

## Metrics to report
- Validation perplexity at fixed compute budget
- Throughput (tokens/sec)
- Peak memory
- Degree statistics of neighbor graph (optional; via debug mode)

