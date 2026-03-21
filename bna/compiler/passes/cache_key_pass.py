from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from bna.compiler.graph_ir import GraphIR


def cache_key_pass(
    ir: GraphIR,
    *,
    T: int,
    H: int,
    dtype: str = "float16",
) -> Dict[str, Any]:
    payload = {
        "ir": ir.to_dict(),
        "T": int(T),
        "H": int(H),
        "dtype": dtype,
        "schema": "wayfinder-cache-v1",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {
        "hash": digest,
        "payload": payload,
        "raw": raw,
    }
