"""Test doubles for the cascade pipeline. No CUDA, no real models, no network."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple

import torch


@contextmanager
def autocast_noop(*args, **kwargs):
    """Replacement for torch.amp.autocast — does nothing on CPU."""
    yield


class FakeMbTokenizer:
    """Mimics the ModernBERT tokenizer's __call__ contract used by pipeline.py.

    Always returns 1 token per character so offset bookkeeping is trivial:
    the offset_mapping is [(0,1),(1,2),...,(n-1,n)] for an n-char input.
    """

    def __call__(self, text: str, *, add_special_tokens=False,
                 return_offsets_mapping=True, truncation=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert truncation is False
        n = len(text)
        return {
            "input_ids": list(range(n)),
            "offset_mapping": [(i, i + 1) for i in range(n)],
        }


@dataclass
class FakeQwenTokenizer:
    """Minimal stand-in for the Qwen tokenizer's chat-template + decode methods."""
    eos_token_id: int = 0
    captured_messages: List[List[dict]] = None

    def __post_init__(self):
        self.captured_messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        self.captured_messages.append(messages)
        # Pretend the rendered prompt is 10 tokens of zeros.
        return torch.zeros((1, 10), dtype=torch.long)

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        # The fake cleaner stuffs the desired output into ids as raw bytes.
        return getattr(self, "_canned_output", "")


class FakeChunker:
    """Returns a fixed logit per call, sequenced from `scalars` (one per chunk)."""

    def __init__(self, scalars: List[float]):
        self._scalars = list(scalars)
        self.calls = 0

    def __call__(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        s = self._scalars[self.calls]
        self.calls += 1
        return torch.tensor(s)


class FakeSpanner:
    """Returns canned (start_logits, end_logits, no_answer_logit) per call."""

    def __init__(self, calls: List[Tuple[List[float], List[float], float]]):
        self._calls = list(calls)
        self.idx = 0

    def __call__(self, ids: torch.Tensor, attn: torch.Tensor):
        s_log, e_log, na = self._calls[self.idx]
        self.idx += 1
        return (
            torch.tensor(s_log).unsqueeze(0),
            torch.tensor(e_log).unsqueeze(0),
            torch.tensor(na),
        )


class FakeCleaner:
    """Records the prompt ids and returns a tensor whose decoded value is `canned`."""

    def __init__(self, canned: str, qwen_tok: FakeQwenTokenizer):
        self._canned = canned
        self._qwen_tok = qwen_tok
        self.last_prompt = None

    def generate(self, prompt_ids: torch.Tensor, *, max_new_tokens: int,
                 do_sample: bool, pad_token_id: int) -> torch.Tensor:
        assert do_sample is False
        self.last_prompt = prompt_ids
        # Set up the fake tokenizer to return our canned string from decode().
        self._qwen_tok._canned_output = self._canned
        # Return prompt + a single synthetic generated token (the tokenizer ignores it).
        return torch.cat([prompt_ids, torch.tensor([[1]])], dim=1)
