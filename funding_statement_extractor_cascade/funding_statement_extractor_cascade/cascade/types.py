from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CascadeConfig:
    chunk_repo: str = "cometadata/funding-chunk-classifier-modernbert-base"
    span_repo:  str = "cometadata/funding-extraction-modernbert-base-spanhead"
    clean_repo: str = "cometadata/funding-cleaning-qwen3-4b-lora"
    mb_base:    str = "answerdotai/ModernBERT-base"
    qwen_base:  str = "Qwen/Qwen3-4B-Instruct-2507"

    max_tok: int = 8192
    stride:  int = 4096
    chunk_keep_threshold: float = 0.4
    chunk_top_k: int = 2

    span_no_answer_threshold: float = 0.5
    span_max_length: int = 300

    cleaner_context_chars: int = 400
    cleaner_max_new_tokens: int = 512
    skip_cleanup: bool = False

    device: str = "cuda"
    dtype:  str = "bf16"   # one of: auto, fp32, fp16, bf16


@dataclass
class CascadeResult:
    funding_statement: str            # "" if none
    rough_span: str                   # "" if none
    chunk_prob: Optional[float]       # max stage-1 prob across surviving chunks; None if no chunks
    no_answer_prob: Optional[float]   # min stage-2 no_answer; None if no chunk fed to spanner
    latency_ms: float
    error: Optional[str] = None


@dataclass
class DocPayload:
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    doc_id: str
    result: CascadeResult
    enqueue_ts: float
    yield_ts: float
    metadata: Optional[Dict[str, Any]] = None
