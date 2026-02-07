#!/usr/bin/env bash
set -euo pipefail

# Dense baseline
python -m hcsa.train \
  --data data/tinyshakespeare.txt \
  --tokenizer char \
  --attn dense \
  --steps 200 \
  --seq-len 256 \
  --layers 6 \
  --heads 8 \
  --embd 512

# Hamiltonian Cycle Sparse Attention (greedy cycle)
python -m hcsa.train \
  --data data/tinyshakespeare.txt \
  --tokenizer char \
  --attn hcsa \
  --cycle greedy \
  --window 64 \
  --landmark-stride 64 \
  --steps 200 \
  --seq-len 256 \
  --layers 6 \
  --heads 8 \
  --embd 512
