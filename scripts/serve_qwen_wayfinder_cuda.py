#!/usr/bin/env python3
"""OpenAI-compatible server for Qwen3.5 + Butterfly on CUDA.

Loads the model once, optionally swaps in Butterfly attention, and serves
an OpenAI chat-completions endpoint. Dense decode is always used for
autoregressive token generation (Butterfly only activates during prefill).

Usage:
    python scripts/serve_qwen_wayfinder_cuda.py \
        --model-path ~/HF_Models/Qwen3.5-9B \
        --mode butterfly \
        --port 8012

    # Then query it:
    curl http://localhost:8012/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3.5-9b-butterfly","messages":[{"role":"user","content":"Hello"}]}'
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import transformers as transformers_module
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.qwen_torch import (  # noqa: E402
    QwenCUDAWayfinderConfig,
    get_cuda_arch_support_diagnostics,
    iter_qwen_wayfinder_layers,
    swap_qwen_attention_with_wayfinder_cuda,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clear_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        ipc_collect()


def _unsupported_flex_message(diag: Dict[str, Any]) -> str:
    cap = diag.get("capability")
    cap_str = "unknown" if cap is None else f"{cap[0]}.{cap[1]}"
    return (
        "Requested `--engine flex` on a CUDA device that this PyTorch build does not "
        f"explicitly support (device capability={cap_str}; torch arch list={diag.get('supported_arch_list')}). "
        "Use `--engine batched`, or pass `--allow-unsupported-arch` if you intentionally "
        "want to force flex on this machine."
    )


def _quantization_config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {"repr": repr(config)}


def _checkpoint_native_quantization(config: Any) -> Optional[Dict[str, Any]]:
    native = _quantization_config_to_dict(getattr(config, "quantization_config", None))
    if native is not None:
        return native
    text_config = getattr(config, "text_config", None)
    return _quantization_config_to_dict(getattr(text_config, "quantization_config", None))


def _choose_auto_model_loader(config: Any) -> Any:
    architectures = tuple(getattr(config, "architectures", []) or ())
    if any("ConditionalGeneration" in arch for arch in architectures):
        loader = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if loader is None:
            raise RuntimeError(
                "This checkpoint requires `AutoModelForImageTextToText`, "
                "but the active transformers install does not expose it."
            )
        return loader
    return transformers_module.AutoModelForCausalLM


_UNSUPPORTED_FLEX_VALIDATED_SEQ_CAPS = {
    "permute": 8192,
    "block_sparse": 4096,
}


def _unsupported_flex_longrun_message(
    *,
    path: str,
    max_input_tokens: Optional[int],
    validated_cap: int,
) -> str:
    requested = "unbounded" if max_input_tokens is None else f"{int(max_input_tokens):,}"
    return (
        "Refusing unsafe-longrun unsupported-arch flex service configuration. "
        f"`--path {path}` on this machine is only validated through {int(validated_cap):,} input tokens, "
        f"but the server is configured for {requested}. "
        f"Lower `--max-input-tokens` to <= {int(validated_cap):,}, use a supported CUDA arch, "
        "switch to a non-flex path, or pass `--unsafe-longrun` if you "
        "intentionally want to risk driver faults or host instability."
    )


def _guard_unsupported_flex_longrun(
    *,
    path: str,
    engine: str,
    arch_diag: Dict[str, Any],
    max_input_tokens: Optional[int],
    allow_unsafe_longrun: bool,
) -> None:
    if not torch.cuda.is_available():
        return
    if arch_diag.get("exact_match"):
        return
    if engine != "flex":
        return
    validated_cap = _UNSUPPORTED_FLEX_VALIDATED_SEQ_CAPS.get(path)
    if validated_cap is None:
        return
    if max_input_tokens is not None and int(max_input_tokens) <= int(validated_cap):
        return
    if allow_unsafe_longrun:
        _log(
            "WARNING: "
            + _unsupported_flex_longrun_message(
                path=path,
                max_input_tokens=max_input_tokens,
                validated_cap=int(validated_cap),
            )
        )
        return
    raise SystemExit(
        _unsupported_flex_longrun_message(
            path=path,
            max_input_tokens=max_input_tokens,
            validated_cap=int(validated_cap),
        )
    )


@dataclass
class ServerState:
    model: Any
    tokenizer: Any
    model_id: str
    mode: str
    lock: threading.Lock
    wayfinder_cfg: Optional[QwenCUDAWayfinderConfig] = None
    replaced_layers: Optional[List[int]] = None
    max_input_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Butterfly CUDA OpenAI Bridge", version="0.1.0")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        gpu_mem = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_bytes = getattr(props, "total_memory", None)
            if total_bytes is None:
                total_bytes = getattr(props, "total_mem")
            gpu_mem = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "total_gb": round(total_bytes / (1024**3), 2),
            }
        return {
            "ok": True,
            "model": state.model_id,
            "mode": state.mode,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_mem": gpu_mem,
            "replaced_layers": state.replaced_layers,
            "busy": bool(state.lock.locked()),
            "max_input_tokens": state.max_input_tokens,
            "max_total_tokens": state.max_total_tokens,
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "butterfly-local",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")

        messages: List[Dict[str, str]] = []
        for m in messages_raw:
            if isinstance(m, dict):
                messages.append({
                    "role": str(m.get("role", "user")),
                    "content": str(m.get("content", "")),
                })
        if not messages:
            raise HTTPException(status_code=400, detail="No valid messages found.")

        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 256)
        max_tokens = max(1, min(max_tokens, 4096))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        stream = bool(payload.get("stream", False))

        if not state.lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Server busy — one request at a time.")

        try:
            # Build prompt via chat template
            prompt_text = state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = state.tokenizer(prompt_text, return_tensors="pt")
            prompt_len = int(inputs["input_ids"].shape[1])
            if state.max_input_tokens is not None and prompt_len > state.max_input_tokens:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt too long: {prompt_len} tokens exceeds server cap "
                        f"{state.max_input_tokens}. Start the server with a higher "
                        "`--max-input-tokens` if you want to allow this."
                    ),
                )
            if (
                state.max_total_tokens is not None
                and prompt_len + max_tokens > state.max_total_tokens
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Requested total tokens {prompt_len + max_tokens} exceeds server cap "
                        f"{state.max_total_tokens}. Reduce `max_tokens` or raise "
                        "`--max-total-tokens`."
                    ),
                )
            device = next(state.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            with torch.inference_mode():
                output_ids = state.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-7) if temperature > 0 else 1.0,
                    top_p=top_p if temperature > 0 else 1.0,
                )
            elapsed = time.perf_counter() - t0

            new_ids = output_ids[0][prompt_len:]
            text = state.tokenizer.decode(new_ids, skip_special_tokens=True)
            completion_tokens = len(new_ids)

            # Collect Butterfly profiles
            butterfly_profiles = []
            for layer in iter_qwen_wayfinder_layers(state.model):
                p = layer.last_profile
                butterfly_profiles.append({
                    "layer_idx": p.get("layer_idx"),
                    "mode": p.get("mode"),
                    "reason": p.get("reason"),
                    "elapsed_ms": p.get("elapsed_ms"),
                })

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        finally:
            _clear_cuda_memory()
            state.lock.release()

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        usage = {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        }
        meta = {
            "latency_sec": round(elapsed, 3),
            "tokens_per_sec": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0,
            "mode": state.mode,
            "butterfly_profiles": butterfly_profiles,
            "wayfinder_profiles": butterfly_profiles,
        }

        if not stream:
            return JSONResponse({
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": state.model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
                "butterfly_meta": meta,
                "wayfinder_meta": meta,
            })

        def _stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': state.model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': state.model_id, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Serve Qwen3.5 + Butterfly on CUDA (OpenAI-compatible)")
    p.add_argument("--model-path", type=str, required=True,
                    help="Local path or HF model ID")
    p.add_argument("--model-id", type=str, default="",
                    help="Model ID exposed in API (default: auto from path)")
    p.add_argument("--mode", type=str, default="butterfly",
                    help="Attention mode: butterfly (sparse prefill, default) or dense. Legacy alias: wayfinder.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--quantize",
        type=str,
        default="none",
        choices=["none", "fp8-weight-only", "fp8-dynamic"],
        help=(
            "Post-training quantization: none (BF16), "
            "fp8-weight-only (FP8 weights, BF16 compute), "
            "fp8-dynamic (FP8 weights + activations, tensor core compute)"
        ),
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--log-level", type=str, default="info")

    # Butterfly config
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--strategy", type=str, default="random",
                    choices=["random", "regular_partition", "greedy", "online_insertion"])
    p.add_argument("--path", type=str, default="sparse", choices=["permute", "sparse", "block_sparse"],
                    help="Butterfly path: `sparse` uses the full HCSA graph; "
                         "`permute` uses the cycle-window surrogate; "
                         "`block_sparse` uses flex-attention over a static block layout.")
    p.add_argument("--engine", type=str, default="auto",
                    choices=["auto", "flex", "batched", "legacy"],
                    help="Butterfly engine for the permute path: auto, flex, batched, or legacy.")
    p.add_argument("--block-size", type=int, default=128,
                    help="Block size for the `block_sparse` path.")
    # Block-sparse topology is always "butterfly" now; the old "wayfinder"/"hamiltonian"
    # option has been archived.  Hidden arg for backwards compat.
    p.add_argument("--block-sparse-topology", type=str, default="butterfly",
                    help=argparse.SUPPRESS)
    p.add_argument("--block-local-window-blocks", type=int, default=1,
                    help="Number of causal predecessor blocks kept per query block for Butterfly block topology.")
    p.add_argument("--block-partner-count", type=int, default=1,
                    help="Number of deterministic partner blocks kept per query block for Butterfly block topology.")
    p.add_argument("--block-sink-blocks", type=int, default=1,
                    help="Number of fixed sink blocks exposed to every later block for Butterfly block topology.")
    p.add_argument("--block-partner-rule", type=str, default="xor",
                    choices=["xor", "bit_reversal", "benes"],
                    help="Deterministic partner rule used by the Butterfly block topology.")
    p.add_argument("--allow-unsupported-arch", action="store_true",
                    help="Allow `--engine flex` even when the current CUDA capability is not "
                         "explicitly supported by this PyTorch build.")
    p.add_argument("--allow-unsafe-unsupported-flex-longrun", "--unsafe-longrun",
                    dest="allow_unsafe_unsupported_flex_longrun", action="store_true",
                    help="Allow unsupported-arch flex service configs above the validated smoke cap "
                         "(permute: 8k, block_sparse: 4k). Unsafe and may fault the driver.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=16384,
                    help="Reject prompts longer than this many tokens. Set <=0 to disable.")
    p.add_argument("--max-total-tokens", type=int, default=20480,
                    help="Reject requests where prompt + completion tokens exceed this cap. Set <=0 to disable.")
    args = p.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    arch_diag = get_cuda_arch_support_diagnostics(0 if torch.cuda.is_available() else None)
    if args.path == "permute" and args.engine == "flex" and torch.cuda.is_available() and not arch_diag["exact_match"]:
        message = _unsupported_flex_message(arch_diag)
        if not args.allow_unsupported_arch:
            raise SystemExit(message)
        _log(f"WARNING: {message}")
    if args.path == "block_sparse" and args.engine not in {"auto", "flex"}:
        raise SystemExit(f"`--path {args.path}` supports only `--engine auto` or `--engine flex`.")
    if args.path == "block_sparse" and torch.cuda.is_available() and not arch_diag["exact_match"]:
        message = _unsupported_flex_message(arch_diag)
        if not args.allow_unsupported_arch:
            raise SystemExit(message)
        _log(f"WARNING: {message}")
        if args.engine == "auto":
            _log(f"Forcing `--engine flex` for `--path {args.path}` on an unsupported arch.")
            args.engine = "flex"
    if args.path == "sparse" and args.engine != "auto":
        _log("Note: --engine is ignored for --path sparse.")
    _guard_unsupported_flex_longrun(
        path=args.path,
        engine=args.engine,
        arch_diag=arch_diag,
        max_input_tokens=args.max_input_tokens if int(args.max_input_tokens) > 0 else None,
        allow_unsafe_longrun=bool(args.allow_unsafe_unsupported_flex_longrun),
    )

    _log(f"Loading model: {args.model_path}")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    native_quantization = _checkpoint_native_quantization(model_config)
    native_quant_method = (
        None if native_quantization is None else native_quantization.get("quant_method")
    )
    quantization_config = None
    if args.quantize != "none":
        if native_quantization is not None:
            raise SystemExit(
                "Checkpoint already declares native quantization "
                f"(quant_method={native_quant_method!r}); refusing to stack "
                f"`--quantize {args.quantize}` on top. Use `--quantize none`."
            )
        from transformers import TorchAoConfig

        if args.quantize == "fp8-weight-only":
            from torchao.quantization import Float8WeightOnlyConfig

            quantization_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
            _log("Quantization: FP8 weight-only (memory savings, BF16 compute)")
        elif args.quantize == "fp8-dynamic":
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

            quantization_config = TorchAoConfig(
                quant_type=Float8DynamicActivationFloat8WeightConfig()
            )
            _log("Quantization: FP8 dynamic (FP8 weights + activations, tensor core compute)")

    load_dtype: torch.dtype | str = dtype_map[args.dtype]
    if native_quantization is not None:
        _log(
            "Checkpoint declares native quantization "
            f"(quant_method={native_quant_method!r}); keeping requested compute dtype {load_dtype}. "
            "On this checkpoint path, `dtype='auto'` attempts to set the default dtype to "
            "float8 and fails before model load."
        )

    model_loader = _choose_auto_model_loader(model_config)
    model = model_loader.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        config=model_config,
        dtype=load_dtype,
        device_map=target_device,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _log(f"Model loader: {getattr(model_loader, '__name__', repr(model_loader))}")
    _log(f"Model loaded on {next(model.parameters()).device}")

    replaced_layers = None
    wf_cfg = None
    mode = args.mode.strip().lower()
    if mode == "wayfinder":
        mode = "butterfly"
    elif mode not in {"butterfly", "dense"}:
        raise SystemExit(
            "`--mode` must be one of ['butterfly', 'dense'] plus legacy alias ['wayfinder']."
        )

    if mode == "butterfly":
        wf_cfg = QwenCUDAWayfinderConfig(
            path=args.path,
            strategy=args.strategy,
            window=args.window,
            landmark_stride=args.landmark_stride if args.landmark_stride > 0 else None,
            num_cycles=args.num_cycles,
            engine=args.engine,
            seed=args.seed,
            block_size=int(args.block_size),
            block_local_window_blocks=int(args.block_local_window_blocks),
            block_partner_count=int(args.block_partner_count),
            block_sink_blocks=int(args.block_sink_blocks),
            block_partner_rule=args.block_partner_rule,
        )
        replaced_layers = swap_qwen_attention_with_wayfinder_cuda(model, wf_cfg)
        _log(f"Butterfly swap: {len(replaced_layers)} layers replaced {replaced_layers}")
        _log(f"  path={wf_cfg.path} strategy={wf_cfg.strategy} "
             f"window={wf_cfg.window} landmark_stride={wf_cfg.landmark_stride} "
             f"block_topology=butterfly "
             f"block_local={wf_cfg.block_local_window_blocks} "
             f"block_partners={wf_cfg.block_partner_count} "
             f"block_sinks={wf_cfg.block_sink_blocks} "
             f"block_partner_rule={wf_cfg.block_partner_rule}")
    else:
        _log("Dense mode — no Butterfly swap.")

    model_id = args.model_id.strip() or f"{Path(args.model_path).name}-{mode}"

    state = ServerState(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        mode=mode,
        lock=threading.Lock(),
        wayfinder_cfg=wf_cfg,
        replaced_layers=replaced_layers,
        max_input_tokens=args.max_input_tokens if args.max_input_tokens > 0 else None,
        max_total_tokens=args.max_total_tokens if args.max_total_tokens > 0 else None,
    )
    app = create_app(state)
    _log(f"Serving on http://{args.host}:{args.port}/v1 (model_id={model_id})")
    _log(f"  POST /v1/chat/completions  |  GET /health  |  GET /v1/models")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
