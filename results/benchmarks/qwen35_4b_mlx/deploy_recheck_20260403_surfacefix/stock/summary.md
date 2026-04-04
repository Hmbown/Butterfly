# Qwen MLX Summary

- mode: `stock`
- butterfly decode backend: `stock`
- butterfly prefill scope: `qwen35_full_attention_layers`
- butterfly prefill target layers: `[3, 7, 11, 15, 19, 23, 27, 31]`
- butterfly prefill active layers: `[]`

| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.8768 | 0.0325 | 75.02 | 1.9834 | 0.00 |
| 8192 | 8.1508 | 0.0890 | 47.90 | 8.3178 | 0.00 |
