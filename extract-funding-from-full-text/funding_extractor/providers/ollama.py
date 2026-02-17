"""Ollama provider implementation."""

import langextract as lx

from funding_extractor.core.models import ExtractionResult
from funding_extractor.providers.base import BaseProvider, ModelProvider


class OllamaProvider(BaseProvider):
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OLLAMA

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
            "language_model_type": lx.inference.OllamaLanguageModel,
            "language_model_params": {
                "model": self.model_id,
                "model_url": self.model_url or "http://localhost:11434",
                "timeout": self.timeout,
            },
        }

    def extract(self, statement: str, prompt: str, examples) -> ExtractionResult:
        params = self.build_extract_params(statement, prompt, examples)
        return self._execute_extract(params, statement)
