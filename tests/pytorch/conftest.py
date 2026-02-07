from __future__ import annotations

# Make tests faster and less noisy on CPU.
try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass
