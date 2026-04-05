from __future__ import annotations

import json
from functools import partial
import importlib
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Tuple


_QWEN35_MLX_COMPAT_PATCHED = False
_QWEN35_ORIGINAL_SANITIZE = None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_config(path: Path) -> Dict[str, Any]:
    return _read_json(path / "config.json")


def _load_weight_index(path: Path) -> Dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map")
    if isinstance(weight_map, dict):
        return {str(k): str(v) for k, v in weight_map.items()}
    return {}


def _detect_detokenizer_class(model_path: Path) -> Any:
    from mlx_lm.tokenizer_utils import (
        BPEStreamingDetokenizer,
        NaiveStreamingDetokenizer,
        SPMStreamingDetokenizer,
        _is_bpe_decoder,
        _is_spm_decoder,
        _is_spm_decoder_no_space,
    )

    detokenizer_class = NaiveStreamingDetokenizer
    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.exists():
        return detokenizer_class

    with tokenizer_file.open("r", encoding="utf-8") as fid:
        try:
            tokenizer_content = json.load(fid)
        except JSONDecodeError as exc:
            raise JSONDecodeError("Failed to parse tokenizer.json", exc.doc, exc.pos)

    decoder = tokenizer_content.get("decoder")
    if _is_spm_decoder(decoder):
        detokenizer_class = SPMStreamingDetokenizer
    elif _is_spm_decoder_no_space(decoder):
        detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
    elif _is_bpe_decoder(decoder):
        detokenizer_class = BPEStreamingDetokenizer
    return detokenizer_class


def _load_qwen35_tokenizer(
    model_path: Path,
    *,
    tokenizer_config_extra: Dict[str, Any] | None = None,
    eos_token_ids: Any = None,
) -> Any:
    from mlx_lm.tokenizer_utils import TokenizerWrapper, _infer_tool_parser
    from transformers import PreTrainedTokenizerFast

    detokenizer_class = _detect_detokenizer_class(model_path)

    tokenizer_kwargs = dict(tokenizer_config_extra or {})
    tokenizer_kwargs.pop("trust_remote_code", None)
    tokenizer_kwargs.setdefault("fix_mistral_regex", True)

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path), **tokenizer_kwargs)
    except AttributeError as exc:
        # Newer tokenizers builds can reject the regex patch path used by some
        # transformers releases. Retry without that patch to keep local MLX
        # loading working under the current Apple Silicon environment.
        if tokenizer_kwargs.get("fix_mistral_regex") and "backend_tokenizer" in str(exc):
            retry_kwargs = dict(tokenizer_kwargs)
            retry_kwargs["fix_mistral_regex"] = False
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path), **retry_kwargs)
        else:
            raise
    tokenizer_config = getattr(tokenizer, "init_kwargs", {})
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    chat_template = None
    if chat_template_type := tokenizer_config.get("chat_template_type", False):
        chat_template = importlib.import_module(
            f"mlx_lm.chat_templates.{chat_template_type}"
        ).apply_chat_template

    tool_parser_type = tokenizer_config.get(
        "tool_parser_type",
        _infer_tool_parser(getattr(tokenizer, "chat_template", None)),
    )
    if tool_parser_type is not None:
        tool_module = importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
        tool_parser = tool_module.parse_tool_call
        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end
    else:
        tool_parser = None
        tool_call_start = None
        tool_call_end = None

    return TokenizerWrapper(
        tokenizer,
        detokenizer_class,
        eos_token_ids=eos_token_ids,
        chat_template=chat_template,
        tool_parser=tool_parser,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
    )


def resolve_qwen_mlx_model_path(model_path: str | Path, *, prefer_text: bool = True) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not prefer_text or path.name.endswith("-text"):
        return path
    text_sibling = path.with_name(f"{path.name}-text")
    if (text_sibling / "config.json").exists():
        return text_sibling
    return path


def _infer_tie_word_embeddings(model_path: Path, config: Dict[str, Any]) -> bool:
    tie_word_embeddings = config.get("tie_word_embeddings")
    if tie_word_embeddings is not None:
        return bool(tie_word_embeddings)
    weight_index = _load_weight_index(model_path)
    return "lm_head.weight" not in weight_index


def build_qwen35_model_config(model_path: str | Path) -> Dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    config = _load_config(path)
    source = dict(config.get("text_config") or config)

    rope_parameters = source.get("rope_parameters") or config.get("rope_parameters") or {}
    attention_bias = source.get("attention_bias")
    if attention_bias is None:
        attention_bias = config.get("attention_bias")

    overrides: Dict[str, Any] = {
        "attention_bias": bool(attention_bias) if attention_bias is not None else False,
        "decoder_sparse_step": int(source.get("decoder_sparse_step") or 1),
        "moe_intermediate_size": int(source.get("moe_intermediate_size") or 0),
        "norm_topk_prob": bool(source.get("norm_topk_prob") or False),
        "num_experts": int(source.get("num_experts") or 0),
        "num_experts_per_tok": int(source.get("num_experts_per_tok") or 0),
        "partial_rotary_factor": float(
            source.get("partial_rotary_factor")
            or rope_parameters.get("partial_rotary_factor")
            or 0.25
        ),
        "rope_scaling": source.get("rope_scaling"),
        "rope_theta": float(
            source.get("rope_theta") or rope_parameters.get("rope_theta") or 10_000_000.0
        ),
        "shared_expert_intermediate_size": int(
            source.get("shared_expert_intermediate_size") or 0
        ),
        "tie_word_embeddings": _infer_tie_word_embeddings(path, source),
    }

    return overrides


def _normalize_qwen35_weight_keys(weights: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in weights.items():
        new_key = str(key)
        if new_key.startswith("language_model.model."):
            new_key = f"model.{new_key[len('language_model.model.'):]}"
        elif new_key.startswith("language_model.lm_head."):
            new_key = f"lm_head.{new_key[len('language_model.lm_head.'):]}"
        elif new_key.startswith("language_model."):
            new_key = new_key[len("language_model.") :]

        if (
            new_key.startswith("model.visual.")
            or new_key.startswith("visual.")
            or new_key.startswith("vision.")
            or new_key.startswith("model.vision_tower.")
            or new_key.startswith("vision_tower.")
            or new_key.startswith("multi_modal_projector.")
        ):
            continue
        normalized[new_key] = value
    return normalized


def _join_split_projection_weights(weights: Dict[str, Any]) -> Dict[str, Any]:
    import mlx.core as mx

    merged = dict(weights)
    join_specs = (
        ("in_proj_qkv", "in_proj_z", "in_proj_qkvz"),
        ("in_proj_b", "in_proj_a", "in_proj_ba"),
    )
    for left_name, right_name, merged_name in join_specs:
        token = f".{left_name}."
        for key in list(merged.keys()):
            if token not in key:
                continue
            prefix, suffix = key.split(token, 1)
            left_key = f"{prefix}.{left_name}.{suffix}"
            right_key = f"{prefix}.{right_name}.{suffix}"
            merged_key = f"{prefix}.{merged_name}.{suffix}"
            if merged_key in merged:
                merged.pop(left_key, None)
                merged.pop(right_key, None)
                continue
            if left_key not in merged or right_key not in merged:
                continue
            merged[merged_key] = mx.concatenate([merged.pop(left_key), merged.pop(right_key)], axis=0)
    return merged


def ensure_qwen35_mlx_compat() -> None:
    global _QWEN35_MLX_COMPAT_PATCHED
    global _QWEN35_ORIGINAL_SANITIZE

    if _QWEN35_MLX_COMPAT_PATCHED:
        return

    import mlx_lm.utils as mlx_utils
    from mlx_lm.models import qwen3_next

    mlx_utils.MODEL_REMAPPING.setdefault("qwen3_5", "qwen3_next")
    mlx_utils.MODEL_REMAPPING.setdefault("qwen3_5_text", "qwen3_next")

    if _QWEN35_ORIGINAL_SANITIZE is None:
        _QWEN35_ORIGINAL_SANITIZE = qwen3_next.Model.sanitize

    def _compat_sanitize(self: Any, weights: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _normalize_qwen35_weight_keys(weights)
        normalized = _join_split_projection_weights(normalized)
        return _QWEN35_ORIGINAL_SANITIZE(self, normalized)

    qwen3_next.Model.sanitize = _compat_sanitize
    _QWEN35_MLX_COMPAT_PATCHED = True


def load_qwen_mlx_model(
    model_path: str | Path,
    *,
    prefer_text: bool = True,
    tokenizer_config: Dict[str, Any] | None = None,
    model_config: Dict[str, Any] | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: str | None = None,
) -> Tuple[Any, Any] | Tuple[Any, Any, Dict[str, Any]]:
    import mlx_lm
    from mlx_lm.utils import load_adapters, load_model

    ensure_qwen35_mlx_compat()
    path = Path(model_path).expanduser()
    if not path.exists():
        return mlx_lm.load(
            str(model_path),
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=return_config,
            revision=revision,
        )

    resolved_path = resolve_qwen_mlx_model_path(path, prefer_text=prefer_text)
    merged_model_config = build_qwen35_model_config(resolved_path)
    if model_config:
        merged_model_config.update(model_config)
    model, config = load_model(resolved_path, lazy=lazy, model_config=merged_model_config)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = _load_qwen35_tokenizer(
        resolved_path,
        tokenizer_config_extra=tokenizer_config,
        eos_token_ids=config.get("eos_token_id", None),
    )
    if return_config:
        return model, tokenizer, config
    return model, tokenizer
