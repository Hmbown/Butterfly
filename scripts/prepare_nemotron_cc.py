"""Prepare an NVIDIA open dataset (Nemotron-CC) into a packed mmap .bin file.

Default ``--dataset-id`` is ``nvidia/Nemotron-CC``. If that gated repo errors,
fall back to the freely-available pretraining sample with:

    --dataset-id nvidia/Nemotron-Pretraining-Dataset-sample-v1

Default ``--tokenizer`` is ``gpt2`` (~50k vocab): with ``tiny_150m`` this gives
a ~170M-param model that fits an RTX 3080 10GB budget. Using ``qwen3`` (152k
vocab) inflates the model to ~327M params, which is too large for a 3080
"150M" run — pick that only if you know the memory budget.

Example (RTX 3080 box, 100M tokens):

    python scripts/prepare_nemotron_cc.py \\
        --dataset-id nvidia/Nemotron-CC \\
        --tokenizer gpt2 \\
        --output data/nemotron_cc_train.bin \\
        --max-tokens 100000000

For real training runs, raise --max-tokens (e.g. 5_000_000_000 for 5B tokens).
"""

from __future__ import annotations

import argparse
import json
import os


def _build_tokenizer(kind: str):
    from bna.tokenizers import Qwen3TokenizerAdapter, build_tokenizer
    kind = kind.lower()
    if kind in {"qwen", "qwen3"}:
        return Qwen3TokenizerAdapter()
    return build_tokenizer(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", default="nvidia/Nemotron-CC")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", default="data/nemotron_cc_train.bin")
    parser.add_argument("--tokenizer", default="gpt2", help="gpt2|qwen3|char — gpt2 keeps the model near 150M; qwen3 balloons to ~327M")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100_000_000,
        help="Stop after this many tokens. Bump for real training (e.g. 5_000_000_000).",
    )
    parser.add_argument("--dtype", default="uint32", choices=["uint16", "uint32"])
    parser.add_argument("--no-append-eos", action="store_true")
    trust = parser.add_mutually_exclusive_group()
    trust.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    trust.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    tokenizer = _build_tokenizer(args.tokenizer)

    from bna.data_hf import stream_hf_to_mmap_bin
    meta = stream_hf_to_mmap_bin(
        args.dataset_id,
        output_path=args.output,
        tokenizer=tokenizer,
        text_field=args.text_field,
        split=args.split,
        max_tokens=args.max_tokens,
        dataset_config=args.dataset_config,
        trust_remote_code=args.trust_remote_code,
        append_eos=not args.no_append_eos,
        dtype=args.dtype,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
