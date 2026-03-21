#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bna.integrations.nemotron_h_torch import (
    NemotronHWayfinderConfig,
    swap_nemotron_h_attention_with_wayfinder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap Nemotron-H attention layers for Wayfinder prefill attention on CUDA."
    )
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype used for model weights.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Transformers attention backend for fallback decode/masked paths.",
    )
    parser.add_argument(
        "--path",
        default="permute",
        choices=("permute", "sparse"),
        help="Wayfinder prefill path.",
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=("random", "regular_partition", "greedy", "online_insertion"),
        help="Wayfinder graph construction strategy.",
    )
    parser.add_argument("--window", type=int, default=64, help="Local causal window.")
    parser.add_argument("--landmark-stride", type=int, default=64, help="Landmark stride.")
    parser.add_argument("--num-cycles", type=int, default=1, help="Number of cycle backbones.")
    parser.add_argument(
        "--compiled-graph-dir",
        default=None,
        help="Optional path to a saved Wayfinder graph artifact for the prompt length.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a CUDA kernel that computes row-wise softmax and explain the launch configuration.",
        help="Prompt to run after swapping the layers.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Value passed to AutoModelForCausalLM.from_pretrained(..., device_map=...). '
        'Use "auto" for sharded multi-GPU loading.',
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def maybe_move_inputs(inputs: dict[str, torch.Tensor], model: torch.nn.Module, device_map: str) -> dict[str, torch.Tensor]:
    if str(device_map) == "auto":
        return inputs
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return inputs
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    cfg = NemotronHWayfinderConfig(
        path=args.path,
        strategy=args.strategy,
        window=args.window,
        landmark_stride=args.landmark_stride,
        num_cycles=args.num_cycles,
        compiled_graph_dir=args.compiled_graph_dir,
    )
    replaced = swap_nemotron_h_attention_with_wayfinder(model, cfg)
    print(f"Replaced Nemotron attention layers: {replaced}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_inputs = tokenizer(args.prompt, return_tensors="pt")
    model_inputs = maybe_move_inputs(model_inputs, model, args.device_map)

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))

    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", model)
    for layer_idx in replaced[:4]:
        profile: Any = decoder.layers[layer_idx].mixer.last_profile
        print(f"Layer {layer_idx} profile: {profile}")


if __name__ == "__main__":
    main()
