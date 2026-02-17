"""Provider abstractions and shared validation."""

import concurrent.futures
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import langextract as lx
import requests

from funding_extractor.core.models import Award, ExtractionResult, FunderEntity
from funding_extractor.exceptions import ProviderConfigurationError, ProviderNotFoundError

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL_OPENAI = "local_openai"


@dataclass
class ProviderConfig:
    provider: ModelProvider
    default_model: Optional[str]
    requires_api_key: bool
    default_url: Optional[str] = None


_PROVIDER_CONFIGS = {
    ModelProvider.GEMINI: ProviderConfig(
        provider=ModelProvider.GEMINI,
        default_model="gemini-2.5-flash-lite",
        requires_api_key=True,
        default_url=None,
    ),
    ModelProvider.OLLAMA: ProviderConfig(
        provider=ModelProvider.OLLAMA,
        default_model=None,
        requires_api_key=False,
        default_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    ),
    ModelProvider.OPENAI: ProviderConfig(
        provider=ModelProvider.OPENAI,
        default_model="gpt-4o-mini",
        requires_api_key=True,
        default_url=None,
    ),
    ModelProvider.LOCAL_OPENAI: ProviderConfig(
        provider=ModelProvider.LOCAL_OPENAI,
        default_model=None,
        requires_api_key=False,
        default_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000"),
    ),
}


def get_provider_config(provider: ModelProvider) -> ProviderConfig:
    try:
        return _PROVIDER_CONFIGS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError(f"Unsupported provider '{provider}'.") from exc


class BaseProvider(ABC):
    """Abstract base class for structured extraction providers."""

    def __init__(
        self,
        model_id: Optional[str],
        model_url: Optional[str],
        api_key: Optional[str],
        timeout: int = 60,
        debug: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_url = model_url
        self.api_key = api_key
        self.timeout = timeout
        self.debug = debug

    @property
    @abstractmethod
    def provider(self) -> ModelProvider:
        """Provider type."""

    @abstractmethod
    def build_extract_params(self, statement: str, prompt: str, examples: List[Any]) -> Dict[str, Any]:
        """Build provider-specific parameters for langextract.extract."""

    def extract(self, statement: str, prompt: str, examples: List[Any]) -> ExtractionResult:
        params = self.build_extract_params(statement, prompt, examples)
        return self._execute_extract(params, statement)

    def _execute_extract(self, extract_params: Dict[str, Any], statement: str) -> ExtractionResult:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lx.extract, **extract_params)
                result = future.result(timeout=self.timeout)
                return self._convert_extractions_to_result(result.extractions, statement)
        except concurrent.futures.TimeoutError:
            logger.warning("Request timed out after %s seconds, returning empty result", self.timeout)
            return ExtractionResult(statement=statement, funders=[])

    @staticmethod
    def _convert_extractions_to_result(extractions: List[Any], funding_statement: str) -> ExtractionResult:
        funders_map: Dict[Optional[str], FunderEntity] = {}
        awards_index: Dict[Tuple[Optional[str], Optional[str]], Award] = {}

        def _get_or_create_award(funder_name: Optional[str], funding_scheme: Optional[str]) -> Award:
            key = (funder_name, funding_scheme)
            if key in awards_index:
                return awards_index[key]
            if funder_name not in funders_map:
                funders_map[funder_name] = FunderEntity(funder_name=funder_name, awards=[])
            award = Award()
            if funding_scheme:
                award.funding_scheme.append(funding_scheme)
            funders_map[funder_name].awards.append(award)
            awards_index[key] = award
            return award

        # Pass 1: Register all funder names
        for extraction in extractions:
            if extraction.extraction_class == "funder_name":
                funder_name = extraction.extraction_text
                if funder_name not in funders_map:
                    funders_map[funder_name] = FunderEntity(funder_name=funder_name, awards=[])

        # Pass 2: Attach award_ids, funding_scheme, and award_title
        for extraction in extractions:
            attrs = extraction.attributes or {}
            funder_name = attrs.get("funder_name")

            if extraction.extraction_class == "award_ids":
                if funder_name is not None:
                    scheme = attrs.get("funding_scheme")
                    award = _get_or_create_award(funder_name, scheme)
                    if extraction.extraction_text not in award.award_ids:
                        award.award_ids.append(extraction.extraction_text)

            elif extraction.extraction_class == "funding_scheme":
                if funder_name is not None:
                    _get_or_create_award(funder_name, extraction.extraction_text)

            elif extraction.extraction_class == "award_title":
                if funder_name is not None:
                    scheme = attrs.get("funding_scheme")
                    award = _get_or_create_award(funder_name, scheme)
                    if extraction.extraction_text not in award.award_title:
                        award.award_title.append(extraction.extraction_text)

        # Ensure every funder has at least one award
        for funder in funders_map.values():
            if not funder.awards:
                funder.awards.append(Award())

        return ExtractionResult(statement=funding_statement, funders=list(funders_map.values()))


def get_available_ollama_models(model_url: Optional[str] = None) -> List[str]:
    url = model_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Cannot connect to Ollama at {url}: {exc}") from exc


def validate_gemini_model(model_id: str) -> bool:
    valid_patterns = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]
    for pattern in valid_patterns:
        if model_id.startswith(pattern):
            return True

    pattern = r"^gemini-\d+\.\d+-[a-z]+(-[a-z]+)*$"
    return bool(re.match(pattern, model_id.lower()))


def validate_openai_model(model_id: str) -> bool:
    valid_patterns = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
        "o3-mini",
    ]
    for pattern in valid_patterns:
        if model_id == pattern or model_id.startswith(f"{pattern}-"):
            return True

    return False


def validate_ollama_model(model_id: str, model_url: Optional[str] = None) -> None:
    available_models = get_available_ollama_models(model_url)
    model_base = model_id.split(":")[0]
    model_found = any(model == model_id or model.startswith(f"{model_base}:") for model in available_models)
    if not model_found:
        available_str = ", ".join(sorted(available_models)) if available_models else "none"
        raise ProviderConfigurationError(
            f"Model '{model_id}' is not available in Ollama. "
            f"Available models: {available_str}. Install it with: ollama pull {model_id}"
        )


def validate_provider_requirements(
    provider: ModelProvider,
    api_key: Optional[str],
    model_url: Optional[str],
    model_id: Optional[str] = None,
    skip_model_validation: bool = False,
) -> None:
    config = get_provider_config(provider)

    if config.requires_api_key and not api_key:
        env_var_name = "OPENAI_API_KEY" if provider == ModelProvider.OPENAI else "GEMINI_API_KEY"
        raise ProviderConfigurationError(
            f"{provider.value} requires an API key. "
            f"Set {env_var_name} environment variable or use --api-key"
        )

    if model_id and not skip_model_validation:
        if provider == ModelProvider.OLLAMA:
            validate_ollama_model(model_id, model_url)
        elif provider == ModelProvider.GEMINI:
            if not validate_gemini_model(model_id):
                raise ProviderConfigurationError(
                    f"Model '{model_id}' does not appear to be a valid Gemini model. "
                    "Expected format: gemini-X.X-[flash|pro|lite] "
                    "(e.g., gemini-2.5-flash, gemini-2.5-pro)"
                )
        elif provider == ModelProvider.OPENAI:
            is_custom_endpoint = model_url and not model_url.startswith("https://api.openai.com")
            if not is_custom_endpoint and not validate_openai_model(model_id):
                raise ProviderConfigurationError(
                    f"Model '{model_id}' does not appear to be a valid OpenAI model. "
                    "Expected format: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc."
                )
        elif provider == ModelProvider.LOCAL_OPENAI:
            # Validation handled by the local endpoint; no-op here.
            return
        else:
            raise ProviderNotFoundError(f"Unsupported provider {provider}.")
