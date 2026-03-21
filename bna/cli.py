"""Optional CLI entrypoint helpers.

This repo primarily supports module invocation:

- python -m hcsa.train ...
- python -m hcsa.generate ...

This module exists as a convenience if you want to add console_scripts.
"""

from __future__ import annotations

from .train import main as train_main
from .generate import main as generate_main

__all__ = ["train_main", "generate_main"]
