#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import convert as mlx_convert
from mlx_lm import load


DEFAULT_MLX_CANDIDATES = [
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-4B-Instruct-4bit",
]
DEFAULT_HF_CANDIDATES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct",
]


def _slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def _validate_forward(model, config: Dict[str, Any]) -> Dict[str, Any]:
    vocab = int(config.get("vocab_size", 32000))
    test_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    test_ids = [min(vocab - 1, max(0, t)) for t in test_ids]
    x = mx.array([test_ids], dtype=mx.int32)
    y = model(x)
    mx.eval(y)
    return {
        "forward_ok": True,
        "output_shape": list(y.shape),
        "output_dtype": str(y.dtype),
    }


def _try_load(candidate: str, trust_remote_code: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"candidate": candidate}
    try:
        model, tokenizer, config = load(
            candidate,
            return_config=True,
            lazy=True,
            tokenizer_config={"trust_remote_code": trust_remote_code},
        )
        out["load_ok"] = True
        out["config"] = {
            "model_type": config.get("model_type"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
            "rope_scaling": config.get("rope_scaling"),
        }
        out["validation"] = _validate_forward(model, config)
        return out
    except Exception as exc:  # pragma: no cover - depends on runtime/network
        out["load_ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch or convert Qwen3-4B model for MLX")
    p.add_argument("--model-id", type=str, default="", help="Single preferred model id/path")
    p.add_argument(
        "--mlx-candidates",
        nargs="*",
        default=DEFAULT_MLX_CANDIDATES,
        help="MLX-native candidates to try first.",
    )
    p.add_argument(
        "--hf-candidates",
        nargs="*",
        default=DEFAULT_HF_CANDIDATES,
        help="HF model ids used for conversion fallback.",
    )
    p.add_argument("--out-root", type=Path, default=Path("mlx_models"))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--q-bits", type=int, default=4)
    p.add_argument("--q-group-size", type=int, default=64)
    p.add_argument("--dtype", type=str, default="")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    attempts: List[Dict[str, Any]] = []

    ordered_candidates: List[str] = []
    if args.model_id:
        ordered_candidates.append(args.model_id)
    for cid in args.mlx_candidates:
        if cid not in ordered_candidates:
            ordered_candidates.append(cid)

    selected: Dict[str, Any] | None = None
    for candidate in ordered_candidates:
        attempt = _try_load(candidate, trust_remote_code=args.trust_remote_code)
        attempts.append(attempt)
        if attempt.get("load_ok"):
            selected = {
                "source": "mlx_native_or_local",
                "model_id": candidate,
                "local_model_path": candidate if Path(candidate).exists() else None,
                "config": attempt.get("config"),
                "validation": attempt.get("validation"),
            }
            break

    if selected is None:
        args.out_root.mkdir(parents=True, exist_ok=True)
        for hf_candidate in args.hf_candidates:
            local_dir = args.out_root / _slugify_model_id(hf_candidate)
            if local_dir.exists():
                attempt = _try_load(str(local_dir), trust_remote_code=args.trust_remote_code)
                attempt["candidate"] = f"{hf_candidate} -> {local_dir}"
                attempt["from_cache"] = True
                attempts.append(attempt)
                if attempt.get("load_ok"):
                    selected = {
                        "source": "converted_cached",
                        "model_id": hf_candidate,
                        "local_model_path": str(local_dir),
                        "config": attempt.get("config"),
                        "validation": attempt.get("validation"),
                    }
                    break
            if selected is not None:
                break

            convert_attempt: Dict[str, Any] = {"candidate": hf_candidate, "conversion": True}
            try:
                mlx_convert(
                    hf_path=hf_candidate,
                    mlx_path=str(local_dir),
                    quantize=bool(args.quantize),
                    q_bits=int(args.q_bits),
                    q_group_size=int(args.q_group_size),
                    dtype=args.dtype or None,
                    trust_remote_code=args.trust_remote_code,
                )
                convert_attempt["convert_ok"] = True
                load_attempt = _try_load(str(local_dir), trust_remote_code=args.trust_remote_code)
                load_attempt["candidate"] = f"{hf_candidate} -> {local_dir}"
                attempts.append(convert_attempt)
                attempts.append(load_attempt)
                if load_attempt.get("load_ok"):
                    selected = {
                        "source": "converted_now",
                        "model_id": hf_candidate,
                        "local_model_path": str(local_dir),
                        "config": load_attempt.get("config"),
                        "validation": load_attempt.get("validation"),
                    }
                    break
            except Exception as exc:  # pragma: no cover - depends on network/hub
                convert_attempt["convert_ok"] = False
                convert_attempt["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(convert_attempt)

    payload: Dict[str, Any] = {
        "status": "ok" if selected is not None else "failed",
        "selected": selected,
        "attempts": attempts,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

