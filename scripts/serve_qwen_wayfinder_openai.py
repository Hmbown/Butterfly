#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bna.integrations.qwen_mlx import (  # noqa: E402
    QwenWayfinderConfig,
    swap_qwen_attention_with_wayfinder,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _configure_hf_cache(
    *,
    hf_home: Optional[str],
    hf_hub_cache: Optional[str],
    hf_offline: bool,
) -> Dict[str, Optional[str]]:
    default_home = Path("/Volumes/VIXinSSD/hf_cache")
    resolved_home = str(default_home) if default_home.exists() else None
    if hf_home:
        resolved_home = str(Path(hf_home).expanduser())
    elif os.environ.get("HF_HOME"):
        resolved_home = str(Path(os.environ["HF_HOME"]).expanduser())

    resolved_hub = None
    if hf_hub_cache:
        resolved_hub = str(Path(hf_hub_cache).expanduser())
    elif os.environ.get("HF_HUB_CACHE"):
        resolved_hub = str(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    elif resolved_home:
        resolved_hub = str(Path(resolved_home) / "hub")

    if resolved_home:
        os.environ["HF_HOME"] = resolved_home
    if resolved_hub:
        os.environ["HF_HUB_CACHE"] = resolved_hub
    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    return {
        "hf_home": resolved_home,
        "hf_hub_cache": resolved_hub,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def _encode_text(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            out = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(out, list):
                return [int(x) for x in out]
        except TypeError:
            out = tokenizer.encode(text)
            if isinstance(out, list):
                return [int(x) for x in out]
    return []


def _build_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except TypeError:
            try:
                return str(tokenizer.apply_chat_template(messages, tokenize=False))
            except Exception:
                pass
        except Exception:
            pass

    lines: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content", ""))
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _normalize_num_cycles(raw: str) -> int | str:
    value = str(raw).strip().lower()
    if value == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--num-cycles must be >= 0 or 'auto'")
    return parsed


def _make_openai_sampler(temperature: float, top_p: float):
    temp = float(max(0.0, temperature))
    p = float(max(0.0, min(1.0, top_p)))
    # mlx-lm sampler treats top_p=0 as disabled.
    if p >= 1.0:
        p = 0.0
    return make_sampler(temp=temp, top_p=p)


def _clear_workspace() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


@dataclass
class ServerState:
    model: Any
    tokenizer: Any
    model_id: str
    mode: str
    lock: threading.Lock


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Wayfinder OpenAI Bridge", version="0.1.0")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model": state.model_id,
            "mode": state.mode,
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
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        messages_raw = payload.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")

        messages: List[Dict[str, str]] = []
        for m in messages_raw:
            if not isinstance(m, dict):
                continue
            messages.append(
                {
                    "role": str(m.get("role", "user")),
                    "content": str(m.get("content", "")),
                }
            )
        if not messages:
            raise HTTPException(status_code=400, detail="No valid chat messages found.")

        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 256)
        if max_tokens <= 0:
            max_tokens = 1
        max_tokens = int(min(max_tokens, 4096))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        stream = bool(payload.get("stream", False))

        # Enforce single inference at a time as an explicit server-side gate.
        if not state.lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Server is busy. Only one inference request is allowed at a time.",
            )

        try:
            prompt = _build_prompt(state.tokenizer, messages)
            prompt_ids = _encode_text(state.tokenizer, prompt)
            sampler = _make_openai_sampler(temperature, top_p)
            t0 = time.perf_counter()
            text = str(
                generate(
                    state.model,
                    state.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            )
            elapsed = time.perf_counter() - t0
            completion_ids = _encode_text(state.tokenizer, text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        finally:
            _clear_workspace()
            state.lock.release()

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        usage = {
            "prompt_tokens": int(len(prompt_ids)),
            "completion_tokens": int(len(completion_ids)),
            "total_tokens": int(len(prompt_ids) + len(completion_ids)),
        }
        meta = {
            "latency_sec": float(elapsed),
            "mode": state.mode,
            "single_inference_gate": True,
        }

        if not stream:
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "wayfinder_meta": meta,
            }
            return JSONResponse(response)

        def _stream_iter():
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_iter(), media_type="text/event-stream")

    return app


def _build_wayfinder_config(args: argparse.Namespace) -> QwenWayfinderConfig:
    return QwenWayfinderConfig(
        path="permute",
        strategy="random",
        window=int(args.window),
        landmark_stride=None if int(args.landmark_stride) <= 0 else int(args.landmark_stride),
        num_cycles=_normalize_num_cycles(str(args.num_cycles)),
        edge_disjoint=not bool(args.disable_edge_disjoint),
        enforce_hamiltonian=not bool(args.allow_non_hamiltonian),
        seed=int(args.seed),
        edge_bias=True,
        window_drop=0.0,
        compute_edge_utilization_proxy=False,
        compute_graph_metrics=False,
        permute_head_chunk_size=int(max(1, args.head_chunk_size)),
        query_chunk_size=int(max(1, args.query_chunk_size)),
        use_fused_dispatch=not bool(args.disable_fused_dispatch),
        wayfinder_decode_backend="dense",
        retro_backfill_enabled=False,
        retro_backfill_alpha=0.0,
        retro_backfill_training_only=True,
        retro_backfill_causal_only=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Serve Qwen + Wayfinder as an OpenAI-compatible endpoint.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-id", type=str, default="")
    p.add_argument("--mode", type=str, default="wayfinder", choices=["wayfinder", "dense"])
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--log-level", type=str, default="info")
    p.add_argument("--hf-home", type=str, default="/Volumes/VIXinSSD/hf_cache")
    p.add_argument("--hf-hub-cache", type=str, default="/Volumes/VIXinSSD/hf_cache/hub")
    p.add_argument("--hf-offline", action="store_true", default=False)

    # Wayfinder controls (decode is always forced dense in this server).
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--landmark-stride", type=int, default=64)
    p.add_argument("--num-cycles", type=str, default="1")
    p.add_argument("--head-chunk-size", type=int, default=2)
    p.add_argument("--query-chunk-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-fused-dispatch", action="store_true", default=False)
    p.add_argument("--allow-non-hamiltonian", action="store_true", default=False)
    p.add_argument("--disable-edge-disjoint", action="store_true", default=False)
    args = p.parse_args()

    cache_cfg = _configure_hf_cache(
        hf_home=(str(args.hf_home).strip() or None),
        hf_hub_cache=(str(args.hf_hub_cache).strip() or None),
        hf_offline=bool(args.hf_offline),
    )
    _log(
        "HF cache config: "
        f"HF_HOME={cache_cfg['hf_home']} HF_HUB_CACHE={cache_cfg['hf_hub_cache']} "
        f"HF_HUB_OFFLINE={cache_cfg['hf_hub_offline']}"
    )
    _log(f"Loading model: {args.model_path}")
    model, tokenizer, _cfg = load(
        args.model_path,
        return_config=True,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )

    mode = str(args.mode).strip().lower()
    if mode == "wayfinder":
        wf_cfg = _build_wayfinder_config(args)
        replaced = swap_qwen_attention_with_wayfinder(model, cfg=wf_cfg)
        _log(
            "Wayfinder swap complete: "
            f"layers_replaced={len(replaced)} decode_backend=dense window={wf_cfg.window}"
        )
    else:
        _log("Dense mode selected: no Wayfinder swap applied.")

    model_id = str(args.model_id).strip() or f"{Path(args.model_path).name}-{mode}"
    state = ServerState(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        mode=mode,
        lock=threading.Lock(),
    )
    app = create_app(state)
    _log(f"Serving OpenAI endpoint on http://{args.host}:{args.port}/v1 (model_id={model_id})")
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level))


if __name__ == "__main__":
    main()
