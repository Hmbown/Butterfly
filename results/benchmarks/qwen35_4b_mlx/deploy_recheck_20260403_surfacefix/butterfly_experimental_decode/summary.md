# Qwen MLX Summary

- mode: `butterfly`
- butterfly decode backend: `experimental`
- butterfly prefill scope: `qwen35_full_attention_layers`
- butterfly prefill target layers: `[3, 7, 11, 15, 19, 23, 27, 31]`
- butterfly prefill active layers: `[3, 7, 11, 15, 19, 23, 27, 31]`

| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.8538 | 0.0262 | 64.53 | 1.9778 | 0.00 |
| 8192 | 7.8905 | 0.2169 | 22.17 | 8.2514 | 0.10 |
