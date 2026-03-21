# BNA — Butterfly Network Attention

Training-free block-sparse attention for long-context inference. Works on existing dense-attention models without retraining.

![Dense causal vs BNA block-sparse attention](docs/assets/bna_block_topology.png)

## How it works

In dense attention, every token attends to every earlier token — O(T²) work per layer. As context gets longer, throughput drops.

BNA splits the sequence into fixed-size blocks (128 tokens) and restricts which blocks can attend to each other. Each block sees:

- **Itself and its immediate neighbors** (local context)
- **One long-range partner block** chosen by a [butterfly network](https://en.wikipedia.org/wiki/Butterfly_network) schedule — at layer `l`, block `b` partners with block `b XOR (1 << (l mod log₂ N))`
- **Block 0** (handles the [attention sink](https://arxiv.org/abs/2309.17453) effect)

The partner changes every layer. After log₂(N) layers, every block can transitively reach every other block. Per-layer work is O(T) instead of O(T²).

## Results

### Qwen 3.5 9B — DGX Spark GB10

Triton block-sparse kernel, `block_size=128`. 8 of 32 layers replaced (24 [DeltaNet](https://arxiv.org/abs/2412.06464) layers untouched). BF16 through 131K, FP8 weight-only at 262K.

| Context | Dense tok/s | BNA tok/s | Top-1 agreement |
|--------:|------------:|----------:|----------------:|
| 4,096   | —           | —         | 99.88%          |
| 8,192   | 1,651       | 1,698     | —               |
| 16,384  | —           | —         | 94.44%          |
| 32,768  | 1,585       | 1,688     | —               |
| 65,536  | 1,475       | 1,724     | —               |
| 98,304  | 1,413       | 1,660     | —               |
| 131,072 | 1,365       | 1,667     | —               |
| 262,144 | 1,257       | 1,712     | —               |

Dense throughput drops 24% from 8K to 262K. BNA stays flat around 1,700 tok/s. At 262K that's 1.36x faster. Memory usage is the same.

### Qwen 3.5 35B A3B FP8

10 of 40 layers replaced. 30 linear-attention / MoE layers untouched.

| Context | Dense tok/s | BNA tok/s |
|--------:|------------:|----------:|
| 8,192   | 931         | 954       |
| 32,768  | 1,280       | 1,301     |
| 65,536  | 1,241       | 1,326     |
| 131,072 | 1,131       | 1,331     |
| 163,840 | —           | 1,306     |
| 196,608 | —           | 1,364     |
| 229,376 | —           | 1,233     |

### Output quality

BNA is an approximation. "Top-1 agreement" is the percentage of positions where BNA and dense attention predict the same next token. At 4K tokens it's 99.88% — nearly identical. At 16K it's 94.44%.

Perplexity and downstream task evaluations haven't been done yet.

## Try it

```bash
git clone https://github.com/Hmbown/Butterfly && cd Butterfly
pip install -e ".[dev]"

python scripts/bench_qwen35_cuda_wayfinder.py \
    --model-path <path-to-Qwen3.5-9B> \
    --path block_sparse --engine triton --block-size 128 \
    --seq-lens 4096 8192 16384 32768
```

## Related work

BNA is training-free — it works on existing models at inference time.

- [BigBird](https://arxiv.org/abs/2007.14062) — random + window + global tokens; BNA replaces the random blocks with a deterministic butterfly schedule
- [Longformer](https://arxiv.org/abs/2004.05150) — sliding window + global attention
- [Monarch](https://arxiv.org/abs/2204.00595) — butterfly matrices for structured computation
- [FlexPrefill](https://arxiv.org/abs/2502.20766) — content-aware per-head sparsity budgets
- [NSA](https://arxiv.org/abs/2502.11089) / [MoBA](https://arxiv.org/abs/2502.13189) — trained sparse attention

## License

MIT
