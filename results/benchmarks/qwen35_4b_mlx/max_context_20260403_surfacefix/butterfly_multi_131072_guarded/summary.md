# Qwen MLX Summary

- mode: `butterfly`
- butterfly decode backend: `stock`
- butterfly prefill scope: `qwen35_full_attention_layers`
- butterfly prefill target layers: `[3, 7, 11, 15, 19, 23, 27, 31]`
- butterfly prefill active layers: `[3, 7, 11, 15, 19, 23, 27, 31]`

| Prompt | Prefill s | TTFT s | Decode tok/s | E2E s | Stock fallback share |
| ---: | ---: | ---: | ---: | ---: | ---: |
