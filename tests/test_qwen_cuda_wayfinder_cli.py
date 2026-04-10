from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    vocab_size = 32000

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text: str, add_special_tokens: bool = False):
        del text, add_special_tokens
        return [1, 2, 3, 4]

    def __len__(self) -> int:
        return self.vocab_size


class _FakeModel:
    last_from_pretrained_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = dict(kwargs)
        return cls()

    def eval(self):
        return self

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))


class _FakeConfig:
    def __init__(
        self,
        *,
        model_type: str = "qwen3_5",
        architectures=None,
        quantization_config=None,
    ) -> None:
        self.model_type = model_type
        self.architectures = list(architectures or ["FakeForCausalLM"])
        self.quantization_config = quantization_config
        self.text_config = None


def _fake_transformers_module(
    *,
    config: _FakeConfig | None = None,
    model_cls: type[_FakeModel] | None = None,
) -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.AutoModelForCausalLM = model_cls or _FakeModel
    module.AutoModelForImageTextToText = model_cls or _FakeModel
    module.AutoTokenizer = _FakeTokenizer

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return config or _FakeConfig()

    module.AutoConfig = _FakeAutoConfig
    return module


class _StopAfterConfig(RuntimeError):
    pass


def test_bench_qwen_cli_wires_butterfly_sparse_path(monkeypatch, tmp_path: Path) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )
    captured = {}

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        raise _StopAfterConfig

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-model",
            "--seq-lens",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--phases",
            "butterfly",
            "--skip-divergence",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
            "--path",
            "sparse",
            "--strategy",
            "regular_partition",
            "--compute-graph-metrics",
            "--sparse-query-chunk-size",
            "512",
            "--sparse-kv-head-chunk-size",
            "2",
            "--sparse-degree-chunk-size",
            "24",
            "--sparse-chunk-temp-budget-mib",
            "192",
            "--sparse-compute-dtype",
            "model",
            "--dump-sparse-trace-dir",
            str(tmp_path / "traces"),
            "--dump-sparse-trace-max-per-layer",
            "2",
            "--dump-sparse-trace-layers",
            "1",
            "3",
        ],
    )

    with pytest.raises(_StopAfterConfig):
        module.main()

    cfg = captured["cfg"]
    assert cfg.path == "sparse"
    assert cfg.strategy == "regular_partition"
    assert cfg.compute_graph_metrics is True
    assert cfg.sparse_query_chunk_size == 512
    assert cfg.sparse_kv_head_chunk_size == 2
    assert cfg.sparse_degree_chunk_size == 24
    assert cfg.sparse_chunk_temp_budget_mib == pytest.approx(192.0)
    assert cfg.sparse_compute_dtype == "model"
    assert cfg.sparse_trace_dir == str(tmp_path / "traces")
    assert cfg.sparse_trace_max_per_layer == 2
    assert cfg.sparse_trace_layer_indices == (1, 3)


def test_bench_qwen_cli_wires_butterfly_block_sparse_path(monkeypatch, tmp_path: Path) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_block_sparse_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )
    captured = {}

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        raise _StopAfterConfig

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-model",
            "--seq-lens",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--phases",
            "butterfly",
            "--skip-divergence",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
            "--path",
            "block_sparse",
            "--block-size",
            "256",
            "--allow-unsupported-arch",
        ],
    )

    with pytest.raises(_StopAfterConfig):
        module.main()

    cfg = captured["cfg"]
    assert cfg.path == "block_sparse"
    assert cfg.block_size == 256
    assert cfg.engine == "triton"


def test_bench_qwen_cli_accepts_legacy_wayfinder_phase_alias_for_block_topology(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_block_topology_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )
    captured = {}

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        raise _StopAfterConfig

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-model",
            "--seq-lens",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--phases",
            "butterfly",
            "--skip-divergence",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
            "--path",
            "block_sparse",
            "--block-size",
            "192",
            "--block-local-window-blocks",
            "2",
            "--block-partner-count",
            "2",
            "--block-sink-blocks",
            "1",
            "--block-partner-rule",
            "bit_reversal",
        ],
    )

    with pytest.raises(_StopAfterConfig):
        module.main()

    cfg = captured["cfg"]
    assert cfg.path == "block_sparse"
    assert cfg.block_size == 192
    assert cfg.block_local_window_blocks == 2
    assert cfg.block_partner_count == 2
    assert cfg.block_sink_blocks == 1
    assert cfg.block_partner_rule == "bit_reversal"


def test_bench_qwen_cli_rejects_butterfly_block_sparse_longrun_without_unsafe_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_block_sparse_longrun_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model, cfg
        raise _StopAfterConfig

    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-model",
            "--seq-lens",
            "32768",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--phases",
            "butterfly",
            "--skip-divergence",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
            "--path",
            "block_sparse",
            "--engine",
            "flex",
            "--allow-unsupported-arch",
        ],
    )

    with pytest.raises(SystemExit, match="unsafe-longrun"):
        module.main()


def test_bench_qwen_backbone_device_map_uses_single_cuda(monkeypatch) -> None:
    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_device_map_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert module._resolve_model_device_map(forward_target="backbone") == 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert module._resolve_model_device_map(forward_target="backbone") == "auto"
    assert module._resolve_model_device_map(forward_target="causal_lm") == "auto"


def test_bench_qwen_backbone_prefers_language_model() -> None:
    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_language_backbone_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    language_model = nn.Linear(4, 4)
    model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=language_model))

    assert module._get_forward_module(model, "backbone") is language_model


def test_bench_qwen_loader_uses_image_text_auto_class_for_conditional_generation() -> None:
    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_loader_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    fake_transformers = _fake_transformers_module(
        config=_FakeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"])
    )

    loader = module._choose_auto_model_loader(
        fake_transformers,
        fake_transformers.AutoConfig.from_pretrained("fake-model"),
    )

    assert loader is fake_transformers.AutoModelForImageTextToText


def test_bench_qwen_backbone_residency_rejects_cpu_or_meta(monkeypatch) -> None:
    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_residency_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    class _MixedResidency(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cpu_param = nn.Parameter(torch.zeros(1))
            self.meta_param = nn.Parameter(torch.empty(1, device="meta"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="dense backbone benchmark module residency is invalid"):
        module._validate_backbone_module_residency(_MixedResidency(), label="dense")


def test_collect_butterfly_profiles_preserves_sparse_backend_and_cuda_timing(monkeypatch) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_profiles_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    class _FakeLayer:
        last_profile = {"attn_kernel_ms": 1.0}

        def ensure_last_graph_metrics(self) -> None:
            return None

        def snapshot_last_profile(self, *, sync: bool = True):
            return {
                "layer_idx": 3,
                "mode": "butterfly",
                "reason": None,
                "elapsed_ms": 20.0,
                "graph_build_ms": 0.1,
                "attn_kernel_ms": 12.5,
                "attn_kernel_ms_host": 2.0,
                "path": "sparse",
                "engine": "batched",
                "strategy": "random",
                "graph_source": "runtime",
                "graph_cache_hit": True,
                "graph_metrics": None,
                "graph_cache_entries": 1,
                "sparse_chunk_mode": "auto",
                "sparse_compute_dtype": "bfloat16",
                "sparse_query_input_dtype": "bfloat16",
                "sparse_key_input_dtype": "bfloat16",
                "sparse_value_input_dtype": "bfloat16",
                "sparse_query_chunk_size": 1536,
                "sparse_kv_head_chunk_size": 1,
                "sparse_degree_chunk_size": 194,
                "sparse_num_query_chunks": 6,
                "sparse_num_head_blocks": 4,
                "sparse_num_degree_blocks": 1,
                "sparse_streamed_degree": False,
                "sparse_chunk_budget_exceeded": False,
                "sparse_estimated_temp_mib": 156.09,
                "sparse_contraction_backend": "sdpa",
                "sparse_contraction_cuda_ms": 12.5,
                "sparse_trace_path": "/tmp/trace.pt",
                "sparse_trace_error": None,
            }

    monkeypatch.setattr(qwen_torch, "iter_qwen_wayfinder_layers", lambda _model: [_FakeLayer()])

    profiles = module.collect_wayfinder_profiles(object())

    assert len(profiles) == 1
    assert profiles[0]["attn_kernel_ms"] == pytest.approx(12.5)
    assert profiles[0]["attn_kernel_ms_host"] == pytest.approx(2.0)
    assert profiles[0]["sparse_contraction_backend"] == "sdpa"
    assert profiles[0]["sparse_contraction_cuda_ms"] == pytest.approx(12.5)
    assert profiles[0]["sparse_trace_path"] == "/tmp/trace.pt"
    assert profiles[0]["sparse_trace_error"] is None


def test_collect_butterfly_profiles_preserves_block_sparse_butterfly_metadata(
    monkeypatch,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_block_profiles_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    class _FakeLayer:
        last_profile = {"attn_kernel_ms": 1.0}

        def ensure_last_graph_metrics(self) -> None:
            return None

        def snapshot_last_profile(self, *, sync: bool = True):
            return {
                "layer_idx": 7,
                "mode": "butterfly",
                "reason": None,
                "elapsed_ms": 14.0,
                "graph_build_ms": 0.2,
                "attn_kernel_ms": 9.5,
                "attn_kernel_ms_host": 1.7,
                "path": "block_sparse",
                "engine": "flex",
                "strategy": "random",
                "graph_source": "runtime_block_butterfly",
                "graph_cache_hit": True,
                "graph_metrics": None,
                "block_sparse_backend": "flex_attention",
                "block_sparse_topology": "butterfly",
                "block_sparse_block_size": 128,
                "block_sparse_num_blocks": 64,
                "block_sparse_neighbor_blocks": 5,
                "block_sparse_stage": 3,
                "block_sparse_stage_count": 6,
                "block_local_window_blocks": 2,
                "block_partner_count": 2,
                "block_sink_blocks": [0],
                "block_partner_rule": "xor",
                "block_sparse_cuda_ms": 9.5,
            }

    monkeypatch.setattr(qwen_torch, "iter_qwen_wayfinder_layers", lambda _model: [_FakeLayer()])

    profiles = module.collect_wayfinder_profiles(object())

    assert len(profiles) == 1
    assert profiles[0]["path"] == "block_sparse"
    assert profiles[0]["block_sparse_backend"] == "flex_attention"
    assert profiles[0]["block_sparse_topology"] == "butterfly"
    assert profiles[0]["block_sparse_stage"] == 3
    assert profiles[0]["block_sparse_stage_count"] == 6
    assert profiles[0]["block_local_window_blocks"] == 2
    assert profiles[0]["block_partner_count"] == 2
    assert profiles[0]["block_partner_rule"] == "xor"
    assert profiles[0]["block_sparse_cuda_ms"] == pytest.approx(9.5)


def test_bench_qwen_cli_divergence_emits_each_requested_seq_len(monkeypatch, tmp_path: Path) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_divergence_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )
    captured_cfgs = []

    class _FakeDivergenceModel(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.use_wayfinder = False

        def eval(self):
            return self

        def forward(self, input_ids, use_cache: bool = False):
            del use_cache
            vocab = torch.arange(4, device=input_ids.device, dtype=torch.float32)
            logits = input_ids.to(dtype=torch.float32).unsqueeze(-1) + vocab
            if self.use_wayfinder:
                logits = logits + 0.01
            return types.SimpleNamespace(logits=logits)

    def _fake_swap(model, cfg):
        model.use_wayfinder = True
        captured_cfgs.append(cfg)
        return [0]

    @contextmanager
    def _fake_lock(_path):
        yield

    fake_transformers = _fake_transformers_module(model_cls=_FakeDivergenceModel)

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(qwen_torch, "restore_qwen_dense_attention", lambda _model: [])
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _fake_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-model",
            "--seq-lens",
            "8",
            "16",
            "32",
            "--dtype",
            "bfloat16",
            "--forward-target",
            "causal_lm",
            "--phases",
            "divergence",
            "--output",
            str(tmp_path / "divergence.ndjson"),
            "--lock-file",
            str(tmp_path / "divergence.lock"),
            "--path",
            "sparse",
        ],
    )

    module.main()

    rows = module._load_ndjson_rows(tmp_path / "divergence.ndjson")
    divergence_rows = [row for row in rows if row.get("type") == "divergence"]
    assert [row["seq_len"] for row in divergence_rows] == [8, 16, 32]
    assert len(captured_cfgs) == 1
    assert all("cosine_similarity" in row for row in divergence_rows)
    assert all("top1_agreement" in row for row in divergence_rows)
    assert all("l2_relative" in row for row in divergence_rows)


def test_bench_qwen_cli_keeps_requested_compute_dtype_for_native_fp8_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import bna.integrations.qwen_torch as qwen_torch

    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_native_fp8_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model, cfg
        raise _StopAfterConfig

    _FakeModel.last_from_pretrained_kwargs = None
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(
            config=_FakeConfig(
                model_type="qwen3_5_moe",
                architectures=["Qwen3_5MoeForCausalLM"],
                quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
            )
        ),
    )
    monkeypatch.setattr(qwen_torch, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(qwen_torch, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module, "_exclusive_lock", _fake_lock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-native-fp8-model",
            "--seq-lens",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--phases",
            "butterfly",
            "--skip-divergence",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
        ],
    )

    with pytest.raises(_StopAfterConfig):
        module.main()

    kwargs = _FakeModel.last_from_pretrained_kwargs
    assert kwargs is not None
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["quantization_config"] is None


def test_bench_qwen_cli_rejects_runtime_quantization_on_native_fp8_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module(
        "bench_qwen35_cuda_wayfinder_native_fp8_reject_test",
        "scripts/bench_qwen35_cuda_wayfinder.py",
    )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(
            config=_FakeConfig(
                model_type="qwen3_5_moe",
                architectures=["Qwen3_5MoeForCausalLM"],
                quantization_config={"quant_method": "fp8"},
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_qwen35_cuda_wayfinder.py",
            "--model-path",
            "fake-native-fp8-model",
            "--seq-lens",
            "8",
            "--quantize",
            "fp8-weight-only",
            "--output",
            str(tmp_path / "bench.ndjson"),
            "--lock-file",
            str(tmp_path / "bench.lock"),
        ],
    )

    with pytest.raises(SystemExit, match="already declares native quantization"):
        module.main()


def test_serve_qwen_cli_defaults_to_butterfly_sparse(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )
    captured = {}

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        return [3, 7]

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-model",
            "--port",
            "9999",
        ],
    )

    module.main()

    cfg = captured["cfg"]
    assert cfg.path == "sparse"
    assert cfg.strategy == "random"


def test_serve_qwen_cli_keeps_requested_compute_dtype_for_native_fp8_checkpoint(
    monkeypatch,
) -> None:
    _FakeModel.last_from_pretrained_kwargs = None
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers_module(
            config=_FakeConfig(
                model_type="qwen3_5_moe",
                architectures=["Qwen3_5MoeForCausalLM"],
                quantization_config={"quant_method": "fp8"},
            )
        ),
    )
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_native_fp8_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )

    def _capture_swap(model, cfg):
        del model, cfg
        return [3, 7]

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": None,
        "supported_arch_list": [],
        "exact_match": True,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-native-fp8-model",
            "--mode",
            "butterfly",
            "--port",
            "9999",
        ],
    )

    module.main()

    kwargs = _FakeModel.last_from_pretrained_kwargs
    assert kwargs is not None
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["quantization_config"] is None


def test_serve_qwen_cli_wires_butterfly_block_sparse_smoke_on_unsupported_arch(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_block_sparse_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )
    captured = {}

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        return [3, 7]

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--port",
            "9999",
            "--path",
            "block_sparse",
            "--allow-unsupported-arch",
            "--max-input-tokens",
            "4096",
        ],
    )

    module.main()

    cfg = captured["cfg"]
    assert cfg.path == "block_sparse"
    assert cfg.engine == "flex"


def test_serve_qwen_cli_wires_butterfly_block_topology_on_unsupported_arch(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_block_topology_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )
    captured = {}

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        return [3, 7]

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--port",
            "9999",
            "--path",
            "block_sparse",
            "--allow-unsupported-arch",
            "--max-input-tokens",
            "4096",
            "--block-local-window-blocks",
            "2",
            "--block-partner-count",
            "2",
            "--block-sink-blocks",
            "1",
            "--block-partner-rule",
            "benes",
        ],
    )

    module.main()

    cfg = captured["cfg"]
    assert cfg.path == "block_sparse"
    assert cfg.engine == "flex"
    assert cfg.block_local_window_blocks == 2
    assert cfg.block_partner_count == 2
    assert cfg.block_sink_blocks == 1
    assert cfg.block_partner_rule == "benes"


def test_serve_qwen_cli_rejects_butterfly_block_sparse_longrun_without_unsafe_override(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_block_sparse_longrun_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )
    captured = {}

    @contextmanager
    def _fake_lock(_path):
        yield

    def _capture_swap(model, cfg):
        del model
        captured["cfg"] = cfg
        raise _StopAfterConfig

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-model",
            "--mode",
            "butterfly",
            "--port",
            "9999",
            "--path",
            "block_sparse",
            "--allow-unsupported-arch",
            "--max-input-tokens",
            "32768",
        ],
    )

    with pytest.raises(SystemExit, match="unsafe-longrun"):
        module.main()


def test_serve_qwen_cli_rejects_wayfinder_block_sparse_longrun_without_unsafe_override(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module())
    module = _load_script_module(
        "serve_qwen_wayfinder_cuda_wayfinder_block_sparse_longrun_test",
        "scripts/serve_qwen_wayfinder_cuda.py",
    )

    def _capture_swap(model, cfg):
        del model, cfg
        raise _StopAfterConfig

    monkeypatch.setattr(module, "get_cuda_arch_support_diagnostics", lambda *_: {
        "capability": (12, 1),
        "supported_arch_list": ["sm_80"],
        "exact_match": False,
    })
    monkeypatch.setattr(module, "swap_qwen_attention_with_wayfinder_cuda", _capture_swap)
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Fake GPU")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_qwen_wayfinder_cuda.py",
            "--model-path",
            "fake-model",
            "--mode",
            "wayfinder",
            "--port",
            "9999",
            "--path",
            "block_sparse",
            "--allow-unsupported-arch",
            "--max-input-tokens",
            "32768",
        ],
    )

    with pytest.raises(SystemExit, match="unsafe-longrun"):
        module.main()
