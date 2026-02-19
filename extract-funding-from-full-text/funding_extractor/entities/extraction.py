import os
from typing import List, Optional

import langextract as lx

from funding_extractor.config.loader import (
    load_extraction_examples,
    load_extraction_prompt,
)
from funding_extractor.config.settings import ProviderSettings
from funding_extractor.entities.models import ExtractionResult
from funding_extractor.providers.base import (
    ModelProvider,
    validate_provider_requirements,
)
from funding_extractor.providers.factory import ProviderFactory


def create_extraction_prompt(prompt_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> str:
    return load_extraction_prompt(prompt_file, custom_config_dir)


def create_funding_examples(
    examples_file: Optional[str] = None, custom_config_dir: Optional[str] = None
) -> List[lx.data.ExampleData]:
    examples_data = load_extraction_examples(examples_file, custom_config_dir)

    examples: List[lx.data.ExampleData] = []
    for example in examples_data:
        extractions = []
        for ext in example.get("extractions", []):
            extraction = lx.data.Extraction(extraction_class=ext["class"], extraction_text=ext["text"])
            if "attributes" in ext:
                extraction.attributes = ext["attributes"]
            extractions.append(extraction)

        examples.append(lx.data.ExampleData(text=example["text"], extractions=extractions))

    return examples


def build_provider_settings(
    model_id: Optional[str],
    model_url: Optional[str],
    api_key: Optional[str],
    timeout: int,
    skip_model_validation: bool,
    debug: bool,
    reasoning_effort: Optional[str] = None,
) -> ProviderSettings:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

    validate_provider_requirements(
        api_key=resolved_api_key,
        model_url=model_url,
        model_id=model_id,
        skip_model_validation=skip_model_validation,
    )

    return ProviderSettings(
        provider=ModelProvider.OPENAI,
        model_id=model_id,
        model_url=model_url,
        api_key=resolved_api_key,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        skip_model_validation=skip_model_validation,
        debug=debug,
    )


class StructuredExtractionService:
    def __init__(
        self,
        provider_settings: ProviderSettings,
        prompt_file: Optional[str] = None,
        examples_file: Optional[str] = None,
        custom_config_dir: Optional[str] = None,
    ) -> None:
        self.provider_settings = provider_settings
        self.prompt = create_extraction_prompt(prompt_file, custom_config_dir)
        self.examples = create_funding_examples(examples_file, custom_config_dir)

    def extract_entities(self, funding_statement: str) -> ExtractionResult:
        provider = ProviderFactory.create(self.provider_settings)
        return provider.extract(funding_statement, self.prompt, self.examples)


def extract_structured_entities(
    funding_statement: str,
    model_id: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    skip_model_validation: bool = False,
    timeout: int = 60,
    debug: bool = False,
    reasoning_effort: Optional[str] = None,
    prompt_file: Optional[str] = None,
    examples_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None,
) -> ExtractionResult:
    settings = build_provider_settings(
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        timeout=timeout,
        skip_model_validation=skip_model_validation,
        debug=debug,
        reasoning_effort=reasoning_effort,
    )
    service = StructuredExtractionService(
        provider_settings=settings,
        prompt_file=prompt_file,
        examples_file=examples_file,
        custom_config_dir=custom_config_dir,
    )
    return service.extract_entities(funding_statement)
