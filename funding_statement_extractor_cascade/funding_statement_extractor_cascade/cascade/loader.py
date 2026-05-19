"""Model loader for the cascade. Wired so tests can monkeypatch heavyweight imports."""
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from funding_statement_extractor_cascade.cascade.types import CascadeConfig
from funding_statement_extractor_cascade.models.chunk_classifier.modeling import ChunkClassifier
from funding_statement_extractor_cascade.models.span_head.modeling import SpanHead


_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class CascadeModels:
    mb_tok: object
    chunker: object
    spanner: object
    qwen_tok: Optional[object]
    cleaner: Optional[object]


def _resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return torch.bfloat16
    return _DTYPE_MAP[dtype_str]


def load_models(cfg: CascadeConfig) -> CascadeModels:
    mb_tok = AutoTokenizer.from_pretrained(cfg.mb_base)

    chunker = ChunkClassifier(cfg.mb_base).to(cfg.device).eval()
    chunker.load_state_dict(torch.load(
        hf_hub_download(cfg.chunk_repo, "pytorch_model.bin"),
        map_location=cfg.device, weights_only=True,
    ))

    spanner = SpanHead(cfg.mb_base).to(cfg.device).eval()
    spanner.load_state_dict(torch.load(
        hf_hub_download(cfg.span_repo, "pytorch_model.bin"),
        map_location=cfg.device, weights_only=True,
    ))

    if cfg.skip_cleanup:
        return CascadeModels(mb_tok=mb_tok, chunker=chunker, spanner=spanner,
                             qwen_tok=None, cleaner=None)

    qwen_tok = AutoTokenizer.from_pretrained(cfg.qwen_base)
    cleaner_base = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_base, dtype=_resolve_dtype(cfg.dtype), device_map=cfg.device,
    )
    cleaner = PeftModel.from_pretrained(cleaner_base, cfg.clean_repo).eval()

    return CascadeModels(mb_tok=mb_tok, chunker=chunker, spanner=spanner,
                          qwen_tok=qwen_tok, cleaner=cleaner)
