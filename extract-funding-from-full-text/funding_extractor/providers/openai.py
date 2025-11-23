"""OpenAI provider implementations."""

import os

import langextract as lx

from funding_extractor.core.models import ExtractionResult
from funding_extractor.providers.base import BaseProvider, ModelProvider


class OpenAIProvider(BaseProvider):
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    def _use_custom_endpoint(self) -> bool:
        return bool(self.model_url and not self.model_url.startswith("https://api.openai.com"))

    def build_extract_params(self, statement: str, prompt: str, examples) -> dict:
        params = {
            "text_or_documents": statement,
            "prompt_description": prompt,
            "examples": examples,
            "temperature": 0.1,
            "extraction_passes": 3,
            "max_workers": 1,
            "debug": self.debug,
            "fence_output": True,
            "use_schema_constraints": False,
        }

        if self._use_custom_endpoint():
            if self.model_url:
                os.environ["OPENAI_BASE_URL"] = self.model_url
            else:
                os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000")

            openai_model_class = lx.inference.OpenAILanguageModel
            language_model = openai_model_class(
                model_id=self.model_id,
                api_key=self.api_key or "dummy-key",
                timeout=self.timeout,
            )
            params["model"] = language_model
        else:
            params["language_model_type"] = lx.inference.OpenAILanguageModel
            params["model_id"] = self.model_id
            params["api_key"] = self.api_key
            params["language_model_params"] = {
                "model_id": self.model_id,
                "api_key": self.api_key,
                "timeout": self.timeout,
            }

        return params

    def extract(self, statement: str, prompt: str, examples) -> ExtractionResult:
        params = self.build_extract_params(statement, prompt, examples)
        return self._execute_extract(params, statement)


class LocalOpenAIProvider(OpenAIProvider):
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.LOCAL_OPENAI

    def _use_custom_endpoint(self) -> bool:
        # Local endpoints always use the custom URL or fallback localhost.
        return True
