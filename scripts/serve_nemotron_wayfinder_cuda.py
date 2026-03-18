#!/usr/bin/env python3
"""OpenAI-compatible server for Nemotron 3 Super + Wayfinder on CUDA.

Loads the model once, optionally swaps in Wayfinder attention, and serves
an OpenAI chat-completions endpoint. Dense decode is always used for
autoregressive token generation (Wayfinder only activates during prefill).

Usage:
    python scripts/serve_nemotron_wayfinder_cuda.py \
        --model-path ~/HF_Models/nvidia/Nemotron-3-Super-49B-v1 \
        --mode wayfinder \
        --port 8013

    # Then query it:
    curl http://localhost:8013/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"nemotron-wayfinder","messages":[{"role":"user","content":"Hello"}]}'
"""
from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcsa.integrations.nemotron_h_torch import (  # noqa: E402
    NemotronHWayfinderConfig,
    iter_nemotron_h_wayfinder_layers,
    swap_nemotron_h_attention_with_wayfinder,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class ServerState:
    model: Any
    tokenizer: Any
    model_id: str
    mode: str
    lock: threading.Lock
    wayfinder_cfg: Optional[NemotronHWayfinderConfig] = None
    replaced_layers: Optional[List[int]] = None


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Wayfinder CUDA OpenAI Bridge (Nemotron)", version="0.1.0")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        gpu_mem = {}
        if torch.cuda.is_available():
            gpu_mem = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2),
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
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "wayfinder-local",
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
            prompt_text = state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = state.tokenizer(prompt_text, return_tensors="pt")
            device = next(state.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

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

            wayfinder_profiles = []
            for layer in iter_nemotron_h_wayfinder_layers(state.model):
                p = layer.last_profile
                wayfinder_profiles.append({
                    "layer_idx": p.get("layer_idx"),
                    "mode": p.get("mode"),
                    "reason": p.get("reason"),
                    "elapsed_ms": p.get("elapsed_ms"),
                })

        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        finally:
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
            "wayfinder_profiles": wayfinder_profiles,
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
                "wayfinder_meta": meta,
            })

        def _stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': state.model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': state.model_id, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Serve Nemotron 3 Super + Wayfinder on CUDA (OpenAI-compatible)")
    p.add_argument("--model-path", type=str, required=True,
                    help="Local path or HF model ID")
    p.add_argument("--model-id", type=str, default="",
                    help="Model ID exposed in API (default: auto from path)")
    p.add_argument("--mode", type=str, default="wayfinder", choices=["wayfinder", "dense"],
                    help="Attention mode: wayfinder (sparse prefill) or dense")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--log-level", type=str, default="info")

    # Wayfinder config
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=int, default=1)
    p.add_argument("--strategy", type=str, default="random",
                    choices=["random", "regular_partition", "greedy", "online_insertion"])
    p.add_argument("--path", type=str, default="permute", choices=["permute", "sparse"])
    p.add_argument("--engine", type=str, default="auto",
                    choices=["auto", "flex", "batched", "legacy"],
                    help="Wayfinder engine: auto, flex, batched, or legacy.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    _log(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _log(f"Model loaded on {next(model.parameters()).device}")

    replaced_layers = None
    wf_cfg = None
    mode = args.mode.strip().lower()

    if mode == "wayfinder":
        wf_cfg = NemotronHWayfinderConfig(
            path=args.path,
            strategy=args.strategy,
            window=args.window,
            landmark_stride=args.landmark_stride if args.landmark_stride > 0 else None,
            num_cycles=args.num_cycles,
            engine=args.engine,
            seed=args.seed,
        )
        replaced_layers = swap_nemotron_h_attention_with_wayfinder(model, wf_cfg)
        _log(f"Wayfinder swap: {len(replaced_layers)} layers replaced {replaced_layers}")
        _log(f"  path={wf_cfg.path} strategy={wf_cfg.strategy} engine={wf_cfg.engine} "
             f"window={wf_cfg.window} landmark_stride={wf_cfg.landmark_stride}")
    else:
        _log("Dense mode — no Wayfinder swap.")

    model_id = args.model_id.strip() or f"{Path(args.model_path).name}-{mode}"

    state = ServerState(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        mode=mode,
        lock=threading.Lock(),
        wayfinder_cfg=wf_cfg,
        replaced_layers=replaced_layers,
    )
    app = create_app(state)
    _log(f"Serving on http://{args.host}:{args.port}/v1 (model_id={model_id})")
    _log(f"  POST /v1/chat/completions  |  GET /health  |  GET /v1/models")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
