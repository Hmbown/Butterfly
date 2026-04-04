# Qwen MLX Summary

- mode: `stock`
- butterfly decode backend: `stock`
- butterfly prefill scope: `qwen35_full_attention_layers`
- butterfly prefill target layers: `[3, 7, 11, 15, 19, 23, 27, 31]`
- butterfly prefill active layers: `[]`

| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1.8771 | 0.0208 | 85.62 | 1.9705 | 0.00 |
| 2048 | 1.8732 | 0.0231 | 83.53 | 1.9689 | 0.00 |
| 2048 | 1.8723 | 0.0239 | 81.34 | 1.9707 | 0.00 |
