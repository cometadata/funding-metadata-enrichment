"""Gemini provider implementation."""

import langextract as lx

from funding_extractor.entities.models import ExtractionResult
from funding_extractor.providers.base import BaseProvider, ModelProvider


class GeminiProvider(BaseProvider):
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.GEMINI

    def build_extract_params(self, statement: str, prompt: str, examples) -> dict:
        return {
            "text_or_documents": statement,
            "prompt_description": prompt,
            "examples": examples,
            "temperature": 0.1,
            "extraction_passes": 3,
            "max_workers": 1,
            "fence_output": False,
            "use_schema_constraints": True,
            "debug": self.debug,
            "language_model_type": lx.inference.GeminiLanguageModel,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "language_model_params": {"timeout": self.timeout},
        }

    def extract(self, statement: str, prompt: str, examples) -> ExtractionResult:
        params = self.build_extract_params(statement, prompt, examples)
        return self._execute_extract(params, statement)
