from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from funding_statement_extractor.exceptions import ConfigurationError


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
class ExtractionSettings:
    colbert_model: str = "lightonai/GTE-ModernColBERT-v1"
    threshold: float = 28.0
    top_k: int = 5
    semantic_batch_size: int = 32


@dataclass
class ProcessingSettings:
    normalize: bool = False
    skip_extraction: bool = False
    enable_pattern_rescue: bool = False
    enable_post_filter: bool = False
    enable_paragraph_prefilter: bool = False


@dataclass
class RuntimeSettings:
    batch_size: int = 50
    workers: Optional[int] = None
    resume: bool = False
    force: bool = False
    verbose: bool = False
    legacy_engine: bool = False
    paragraphs_per_batch: int = 4096
    encode_batch_size: int = 512
    queue_depth: int = 128
    retry_failed: bool = False
    dtype: str = "auto"


@dataclass
class ConfigPaths:
    queries_file: Optional[Path] = None
    config_dir: Optional[Path] = None
    patterns_file: Optional[Path] = None


@dataclass
class ApplicationConfig:
    input: InputSettings
    output: OutputSettings
    extraction: ExtractionSettings
    processing: ProcessingSettings
    runtime: RuntimeSettings
    config_paths: ConfigPaths

    def validate(self) -> None:
        if not self.input.path.exists():
            raise ConfigurationError(f"Input path {self.input.path} does not exist.")

        if self.input.input_format and self.input.input_format not in {"markdown", "parquet"}:
            raise ConfigurationError("input_format must be one of: markdown, parquet.")

        if self.extraction.top_k < 1:
            raise ConfigurationError("top_k must be at least 1.")
        if self.extraction.threshold < 0:
            raise ConfigurationError("threshold must be non-negative.")
