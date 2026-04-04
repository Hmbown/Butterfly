# Qwen MLX Summary

- mode: `butterfly`
- butterfly decode backend: `stock`
- butterfly prefill scope: `qwen35_full_attention_layers`
- butterfly prefill target layers: `[3, 7, 11, 15, 19, 23, 27, 31]`
- butterfly prefill active layers: `[3, 7, 11, 15, 19, 23, 27, 31]`

| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.8767 | 0.0290 | 76.09 | 1.9819 | 0.89 |
| 2048 | 1.8644 | 0.0344 | 70.97 | 1.9771 | 0.89 |
| 2048 | 1.8466 | 0.0299 | 76.02 | 1.9519 | 0.89 |
