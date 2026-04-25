from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, ids: List[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    def state_dict(self) -> Dict:
        ...

    @classmethod
    def from_state_dict(cls, state: Dict) -> "Tokenizer":
        ...


@dataclass
class CharTokenizer:
    """A simple char-level tokenizer derived from an observed alphabet.

    - Always works (no downloads)
    - Deterministic: vocabulary is sorted by codepoint
    """

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(
                f"Character {missing!r} not in vocabulary (vocab_size={self.vocab_size})."
            ) from None

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def state_dict(self) -> Dict:
        return {"type": "char", "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "CharTokenizer":
        assert state.get("type") == "char"
        itos = list(state["itos"])
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class GPT2BPETokenizer:
    """Optional GPT-2 BPE tokenizer wrapper.

    Requires `transformers`.
    """

    def __init__(self):
        try:
            from transformers import GPT2TokenizerFast
        except Exception as e:
            raise ImportError(
                "GPT2BPETokenizer requires `transformers`. Install with: pip install -e .[bpe]"
            ) from e
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def state_dict(self) -> Dict:
        # GPT-2 vocab is standard; store type only.
        return {"type": "gpt2_bpe"}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "GPT2BPETokenizer":
        assert state.get("type") == "gpt2_bpe"
        return cls()


class HFAutoTokenizer:
    """Wrapper around `transformers.AutoTokenizer` for HF-hosted tokenizers.

    Used for Qwen3 and any other HF checkpoint where we want to share the
    pretrained vocab between inference (4-bit Qwen swap) and training
    (from-scratch 150M runs).
    """

    def __init__(self, model_id: str, trust_remote_code: bool = True):
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError(
                "HFAutoTokenizer requires `transformers`. Install with: pip install -e .[bpe]"
            ) from e
        self.model_id = str(model_id)
        self._tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=bool(trust_remote_code)
        )

    @property
    def vocab_size(self) -> int:
        inner = getattr(self._tok, "vocab_size", None)
        if inner is None:
            return int(len(self._tok))
        return int(inner)

    @property
    def eos_id(self) -> int | None:
        return getattr(self._tok, "eos_token_id", None)

    @property
    def pad_id(self) -> int | None:
        pad = getattr(self._tok, "pad_token_id", None)
        if pad is None:
            return self.eos_id
        return int(pad)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def state_dict(self) -> Dict:
        return {"type": "hf_auto", "model_id": self.model_id}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "HFAutoTokenizer":
        assert state.get("type") == "hf_auto"
        return cls(model_id=str(state["model_id"]))


QWEN3_DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"


def Qwen3TokenizerAdapter(model_id: str = QWEN3_DEFAULT_MODEL_ID) -> HFAutoTokenizer:
    return HFAutoTokenizer(model_id=model_id, trust_remote_code=True)


def build_tokenizer(kind: str, text_for_char_vocab: str | None = None) -> Tokenizer:
    kind = kind.lower()
    if kind == "char":
        if text_for_char_vocab is None:
            raise ValueError("Char tokenizer requires text to build the vocabulary.")
        return CharTokenizer.from_text(text_for_char_vocab)
    if kind in {"bpe", "gpt2"}:
        return GPT2BPETokenizer()
    if kind in {"qwen", "qwen3"}:
        return Qwen3TokenizerAdapter()
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def tokenizer_from_state_dict(state: Dict) -> Tokenizer:
    t = state.get("type")
    if t == "char":
        return CharTokenizer.from_state_dict(state)
    if t == "gpt2_bpe":
        return GPT2BPETokenizer.from_state_dict(state)
    if t == "hf_auto":
        return HFAutoTokenizer.from_state_dict(state)
    raise ValueError(f"Unknown tokenizer state type: {t}")
