import logging
import os
from typing import Dict, List, Optional, Tuple

import langextract as lx

from funding_extractor.config.loader import (
    load_extraction_examples,
    load_extraction_prompt,
)
from funding_extractor.config.settings import ProviderSettings
from funding_extractor.entities.models import ExtractionResult
from funding_extractor.providers.base import (
    BaseProvider,
    ModelProvider,
    validate_provider_requirements,
)
from funding_extractor.providers.factory import ProviderFactory

logger = logging.getLogger(__name__)


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
    provider: str = "openai",
    model_id: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
    skip_model_validation: bool = False,
    debug: bool = False,
    reasoning_effort: Optional[str] = None,
    vllm_config_path: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> ProviderSettings:
    provider_enum = ModelProvider(provider)
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if provider_enum == ModelProvider.VLLM:
        skip_model_validation = True

    validate_provider_requirements(
        api_key=resolved_api_key,
        model_url=model_url,
        model_id=model_id,
        skip_model_validation=skip_model_validation,
    )

    return ProviderSettings(
        provider=provider_enum,
        model_id=model_id,
        model_url=model_url,
        api_key=resolved_api_key,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        skip_model_validation=skip_model_validation,
        debug=debug,
        vllm_config_path=vllm_config_path,
        lora_path=lora_path,
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
        self._provider = ProviderFactory.create(provider_settings)

    def extract_entities(self, funding_statement: str) -> ExtractionResult:
        return self._provider.extract(funding_statement, self.prompt, self.examples)

    def extract_entities_with_reasoning(
        self, funding_statement: str
    ) -> Tuple[ExtractionResult, List[str]]:
        result = self._provider.extract(funding_statement, self.prompt, self.examples)
        reasoning = self._provider.drain_reasoning()
        return result, reasoning

    def extract_entities_batch(
        self, statements: List[Tuple[str, str]]
    ) -> Tuple[Dict[str, ExtractionResult], List[str]]:
        """Extract entities from multiple statements in a single batched call.

        Uses langextract's multi-document batching to send batch_length prompts
        per engine.generate() call, enabling GPU parallelism without threading.

        Args:
            statements: List of (doc_id, statement_text) tuples.

        Returns:
            Tuple of (dict mapping doc_id to ExtractionResult, reasoning traces).
        """
        if not statements:
            return {}, []

        documents = [
            lx.data.Document(text=text, document_id=doc_id)
            for doc_id, text in statements
        ]
        text_by_id = {doc_id: text for doc_id, text in statements}

        # Build params using first statement as a template, then override
        params = self._provider.build_extract_params(
            statements[0][1], self.prompt, self.examples
        )
        params["text_or_documents"] = documents
        params["show_progress"] = True

        try:
            annotated_docs = lx.extract(**params)
        except ValueError:
            logger.exception("Batch extraction failed")
            return {doc_id: ExtractionResult(statement=text, funders=[])
                    for doc_id, text in statements}, []

        results: Dict[str, ExtractionResult] = {}
        for ann_doc in annotated_docs:
            doc_id = ann_doc.document_id
            text = text_by_id.get(doc_id, "")
            results[doc_id] = BaseProvider._convert_extractions_to_result(
                ann_doc.extractions or [], text
            )

        # Fill in any missing documents with empty results
        for doc_id, text in statements:
            if doc_id not in results:
                results[doc_id] = ExtractionResult(statement=text, funders=[])

        reasoning = self._provider.drain_reasoning()
        return results, reasoning


def extract_structured_entities(
    funding_statement: str,
    provider: str = "openai",
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
    vllm_config_path: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> ExtractionResult:
    settings = build_provider_settings(
        provider=provider,
        model_id=model_id,
        model_url=model_url,
        api_key=api_key,
        timeout=timeout,
        skip_model_validation=skip_model_validation,
        debug=debug,
        reasoning_effort=reasoning_effort,
        vllm_config_path=vllm_config_path,
        lora_path=lora_path,
    )
    service = StructuredExtractionService(
        provider_settings=settings,
        prompt_file=prompt_file,
        examples_file=examples_file,
        custom_config_dir=custom_config_dir,
    )
    return service.extract_entities(funding_statement)
