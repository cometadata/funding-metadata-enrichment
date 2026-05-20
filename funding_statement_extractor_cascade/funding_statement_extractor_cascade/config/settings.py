from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from funding_statement_extractor_cascade.exceptions import ConfigurationError


@dataclass
class InputSettings:
    path: Path
    input_format: Optional[str] = None
    parquet_text_column: Optional[str] = None
    parquet_id_column: Optional[str] = None
    parquet_batch_size: int = 64


@dataclass
class OutputSettings:
    output_path: Path
    checkpoint_path: Path


@dataclass
class CascadeRuntimeSettings:
    """Cascade hyperparameters and engine knobs that can be overridden from the CLI.

    Keeps CascadeConfig (cascade/types.py) immutable + opinionated; this layer
    is the mutable view used by the app / CLI.
    """
    chunk_repo: str = "cometadata/funding-chunk-classifier-modernbert-base"
    span_repo: str = "cometadata/funding-extraction-modernbert-base-spanhead"
    clean_repo: str = "cometadata/funding-cleaning-qwen3-4b-lora"
    max_tok: int = 8192
    stride: int = 4096
    chunk_keep_threshold: float = 0.4
    chunk_top_k: int = 2
    span_no_answer_threshold: float = 0.5
    span_max_length: int = 300
    cleaner_context_chars: int = 400
    cleaner_max_new_tokens: int = 512
    skip_cleanup: bool = False
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass
class ProcessingSettings:
    skip_extraction: bool = False


@dataclass
class RuntimeSettings:
    batch_size: int = 50            # flush interval
    workers: Optional[int] = None    # reserved for multiprocess input prep
    resume: bool = False
    force: bool = False
    verbose: bool = False
    retry_failed: bool = False


@dataclass
class ApplicationConfig:
    input: InputSettings
    output: OutputSettings
    cascade: CascadeRuntimeSettings
    processing: ProcessingSettings
    runtime: RuntimeSettings

    def validate(self) -> None:
        if not self.input.path.exists():
            raise ConfigurationError(f"Input path {self.input.path} does not exist.")
        if self.input.input_format and self.input.input_format not in {"markdown", "parquet"}:
            raise ConfigurationError("input_format must be one of: markdown, parquet.")
        if self.cascade.chunk_keep_threshold < 0 or self.cascade.chunk_keep_threshold > 1:
            raise ConfigurationError("chunk_keep_threshold must be in [0, 1].")
        if self.cascade.span_no_answer_threshold < 0 or self.cascade.span_no_answer_threshold > 1:
            raise ConfigurationError("span_no_answer_threshold must be in [0, 1].")
        if self.cascade.max_tok < self.cascade.stride:
            raise ConfigurationError("max_tok must be >= stride.")
        if self.cascade.dtype not in {"auto", "fp32", "fp16", "bf16"}:
            raise ConfigurationError("dtype must be one of: auto, fp32, fp16, bf16.")
