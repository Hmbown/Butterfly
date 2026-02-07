from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, List, Sequence

from hcsa.compiler.graph_ir import EdgeBiasScheduleIR, GraphIR, ScheduleSpec


_TOKEN_RE = re.compile(r'\s*(\(|\)|"(?:[^"\\]|\\.)*"|[^\s()]+)')


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text)
    return [t for t in tokens if t and not t.isspace()]


def parse_sexp(text: str) -> Any:
    tokens = _tokenize(text)
    pos = 0

    def parse_node() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out: list[Any] = []
            while pos < len(tokens) and tokens[pos] != ")":
                out.append(parse_node())
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Unbalanced parentheses")
            pos += 1
            return out
        if tok == ")":
            raise ValueError("Unexpected ')' token")
        if tok.startswith('"') and tok.endswith('"'):
            return tok[1:-1]
        return tok

    node = parse_node()
    if pos != len(tokens):
        raise ValueError("Unexpected trailing tokens")
    return node


def _coerce_atom(v: Any) -> Any:
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v)
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "nil", "off"}:
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _kv_tail(parts: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    while i < len(parts):
        k = str(parts[i])
        if not k.startswith(":"):
            i += 1
            continue
        if i + 1 >= len(parts):
            out[k[1:]] = True
            break
        out[k[1:]] = parts[i + 1]
        i += 2
    return out


def _parse_schedule(node: Any) -> ScheduleSpec:
    if not isinstance(node, list) or not node or str(node[0]) != "schedule":
        raise ValueError(f"Expected (schedule ...), got: {node!r}")
    tail = node[1:]
    if not tail:
        raise ValueError("schedule requires arguments")

    kind = str(tail[0]).lstrip(":")
    if kind == "linear":
        if len(tail) < 4:
            raise ValueError("linear schedule must provide start end :steps N")
        start = float(_coerce_atom(tail[1]))
        end = float(_coerce_atom(tail[2]))
        kv = _kv_tail(tail[3:])
        steps = int(_coerce_atom(kv.get("steps", 1)))
        return ScheduleSpec(kind="linear", start=start, end=end, steps=max(1, steps))
    if kind in {"const", "constant"}:
        val = float(_coerce_atom(tail[1] if len(tail) > 1 else 0.0))
        return ScheduleSpec(kind="constant", start=val, end=val, steps=1)

    raise ValueError(f"Unsupported schedule kind: {kind}")


def _parse_bool(v: Any) -> bool:
    x = _coerce_atom(v)
    if isinstance(x, bool):
        return x
    raise ValueError(f"Expected bool, got {v!r}")


def _iter_sections(root: list[Any]) -> Iterator[list[Any]]:
    for part in root[1:]:
        if isinstance(part, list) and part:
            yield part


def parse_graph_ir(root: Any) -> GraphIR:
    if not isinstance(root, list) or not root or str(root[0]) != "wayfinder":
        raise ValueError("Spec root must be (wayfinder ...)")

    degree = 64
    strategy = "random"
    seed = 42
    num_cycles = 1
    window_size = 32
    window_schedule = None
    landmark_stride: int | None = 64
    permute_window_enabled = True
    permute_window_size = 32
    edge_bias = EdgeBiasScheduleIR()

    for section in _iter_sections(root):
        tag = str(section[0])

        if tag == "degree" and len(section) >= 2:
            degree = int(_coerce_atom(section[1]))
            continue

        if tag == "backbone" and len(section) >= 2 and isinstance(section[1], list):
            cycle_cfg = section[1]
            if str(cycle_cfg[0]) != "cycle":
                raise ValueError("Only (backbone (cycle ...)) is supported")
            kv = _kv_tail(cycle_cfg[1:])
            strategy = str(kv.get("type", "random"))
            seed = int(_coerce_atom(kv.get("seed", seed)))
            num_cycles = int(_coerce_atom(kv.get("k", num_cycles)))
            continue

        if tag == "local" and len(section) >= 2 and isinstance(section[1], list):
            local_cfg = section[1]
            if str(local_cfg[0]) != "window":
                raise ValueError("Only (local (window ...)) is supported")
            kv = _kv_tail(local_cfg[1:])
            sz = kv.get("size")
            if isinstance(sz, list):
                window_schedule = _parse_schedule(sz)
                window_size = int(window_schedule.start)
            elif sz is not None:
                window_size = int(_coerce_atom(sz))
            continue

        if tag == "highways" and len(section) >= 2 and isinstance(section[1], list):
            h_cfg = section[1]
            if str(h_cfg[0]) != "landmarks":
                raise ValueError("Only (highways (landmarks ...)) is supported")
            kv = _kv_tail(h_cfg[1:])
            stride = _coerce_atom(kv.get("stride", landmark_stride))
            landmark_stride = None if stride is None else int(stride)
            continue

        if tag == "bias" and len(section) >= 2 and isinstance(section[1], list):
            b_cfg = section[1]
            if str(b_cfg[0]) != "edge_logit_bias":
                raise ValueError("Only (bias (edge_logit_bias ...)) is supported")
            kv = _kv_tail(b_cfg[1:])

            def to_sched(name: str) -> ScheduleSpec | None:
                raw = kv.get(name)
                if raw is None:
                    return None
                if isinstance(raw, list):
                    return _parse_schedule(raw)
                val = float(_coerce_atom(raw))
                return ScheduleSpec(kind="constant", start=val, end=val, steps=1)

            edge_bias = EdgeBiasScheduleIR(
                cycle=to_sched("cycle"),
                window=to_sched("window"),
                landmark=to_sched("landmark"),
                rewire=to_sched("rewire"),
            )
            continue

        if tag == "permute_window":
            kv = _kv_tail(section[1:])
            if "enabled" in kv:
                permute_window_enabled = _parse_bool(kv["enabled"])
            if "window" in kv:
                permute_window_size = int(_coerce_atom(kv["window"]))
            continue

    return GraphIR(
        degree=degree,
        strategy=strategy,
        seed=seed,
        num_cycles=num_cycles,
        window_size=window_size,
        window_schedule=window_schedule,
        landmark_stride=landmark_stride,
        edge_bias=edge_bias,
        permute_window_enabled=permute_window_enabled,
        permute_window_size=permute_window_size,
    )


def load_graph_ir(path: str | Path) -> GraphIR:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    root = parse_sexp(text)
    return parse_graph_ir(root)
