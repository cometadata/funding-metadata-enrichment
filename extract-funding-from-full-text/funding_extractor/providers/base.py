import concurrent.futures
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import langextract as lx

from funding_extractor.entities.models import Award, ExtractionResult, FunderEntity
from funding_extractor.exceptions import ProviderConfigurationError, ProviderNotFoundError

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"


@dataclass
class ProviderConfig:
    provider: ModelProvider
    default_model: Optional[str]
    requires_api_key: bool
    default_url: Optional[str] = None


_PROVIDER_CONFIGS = {
    ModelProvider.OPENAI: ProviderConfig(
        provider=ModelProvider.OPENAI,
        default_model=None,
        requires_api_key=False,
        default_url=os.environ.get("OPENAI_BASE_URL"),
    ),
    ModelProvider.VLLM: ProviderConfig(
        provider=ModelProvider.VLLM,
        default_model=None,
        requires_api_key=False,
    ),
}


def get_provider_config(provider: ModelProvider) -> ProviderConfig:
    try:
        return _PROVIDER_CONFIGS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError(f"Unsupported provider '{provider}'.") from exc


class BaseProvider(ABC):
    def __init__(
        self,
        model_id: Optional[str],
        model_url: Optional[str],
        api_key: Optional[str],
        timeout: int = 60,
        debug: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.model_url = model_url
        self.api_key = api_key
        self.timeout = timeout
        self.debug = debug
        self.reasoning_effort = reasoning_effort

    @property
    @abstractmethod
    def provider(self) -> ModelProvider: ...

    @abstractmethod
    def build_extract_params(self, statement: str, prompt: str, examples: List[Any]) -> Dict[str, Any]: ...

    def drain_reasoning(self) -> List[str]:
        """Return and clear accumulated reasoning traces. Default: no traces."""
        return []

    def extract(self, statement: str, prompt: str, examples: List[Any]) -> ExtractionResult:
        params = self.build_extract_params(statement, prompt, examples)
        return self._execute_extract(params, statement)

    def _execute_extract(self, extract_params: Dict[str, Any], statement: str) -> ExtractionResult:
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(lx.extract, **extract_params)
                result = future.result(timeout=self.timeout)
                return self._convert_extractions_to_result(result.extractions, statement)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Request timed out after %s seconds for statement (%.80s...), "
                    "returning empty result (background thread may still be running)",
                    self.timeout,
                    statement,
                )
                return ExtractionResult(statement=statement, funders=[])
            except ValueError as exc:
                if attempt < max_attempts:
                    logger.warning(
                        "Resolver rejected model output for statement (%.80s...), "
                        "retrying (attempt %d/%d): %s",
                        statement,
                        attempt,
                        max_attempts,
                        exc,
                    )
                else:
                    logger.warning(
                        "Resolver rejected model output for statement (%.80s...) "
                        "after %d attempts: %s",
                        statement,
                        max_attempts,
                        exc,
                    )
                    return ExtractionResult(statement=statement, funders=[])
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
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


def validate_provider_requirements(
    api_key: Optional[str],
    model_url: Optional[str],
    model_id: Optional[str] = None,
    skip_model_validation: bool = False,
) -> None:
    if skip_model_validation:
        return
    if not model_id and not model_url:
        raise ProviderConfigurationError(
            "Either --model or --model-url must be provided."
        )
