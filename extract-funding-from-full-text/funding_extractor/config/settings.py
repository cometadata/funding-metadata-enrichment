import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from funding_extractor.exceptions import ConfigurationError
from funding_extractor.providers.base import (
    ModelProvider,
    ProviderConfig,
    get_provider_config,
    validate_provider_requirements,
)


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
    heal_markdown: bool = False
    skip_extraction: bool = False
    skip_structured: bool = False
    enable_pattern_rescue: bool = False
    enable_post_filter: bool = False


@dataclass
class ProviderSettings:
    provider: ModelProvider = ModelProvider.GEMINI
    model_id: Optional[str] = None
    model_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60
    skip_model_validation: bool = False
    debug: bool = False


@dataclass
class RuntimeSettings:
    batch_size: int = 10
    workers: Optional[int] = None
    resume: bool = False
    force: bool = False
    verbose: bool = False


@dataclass
class ConfigPaths:
    queries_file: Optional[Path] = None
    config_dir: Optional[Path] = None
    patterns_file: Optional[Path] = None
    prompt_file: Optional[Path] = None
    examples_file: Optional[Path] = None


@dataclass
class ApplicationConfig:
    input: InputSettings
    output: OutputSettings
    extraction: ExtractionSettings
    processing: ProcessingSettings
    provider: ProviderSettings
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

        if not self.processing.skip_structured:
            self._apply_provider_defaults()

    def _apply_provider_defaults(self) -> None:
        provider_config: ProviderConfig = get_provider_config(self.provider.provider)

        if self.provider.model_id is None:
            self.provider.model_id = provider_config.default_model
        if self.provider.model_url is None:
            self.provider.model_url = provider_config.default_url

        if self.provider.api_key is None and provider_config.requires_api_key:
            env_name = "OPENAI_API_KEY" if self.provider.provider == ModelProvider.OPENAI else "GEMINI_API_KEY"
            self.provider.api_key = os.environ.get(env_name)

        validate_provider_requirements(
            provider=self.provider.provider,
            api_key=self.provider.api_key,
            model_url=self.provider.model_url,
            model_id=self.provider.model_id,
            skip_model_validation=self.provider.skip_model_validation,
        )
