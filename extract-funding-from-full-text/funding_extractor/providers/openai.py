"""OpenAI provider implementations."""

import os
from functools import wraps

import langextract as lx

from funding_extractor.entities.models import ExtractionResult
from funding_extractor.providers.base import BaseProvider, ModelProvider


def _patch_client_for_extra_body(language_model):
    """Wrap the OpenAI client's create method to move unsupported top-level
    kwargs (like ``reasoning``) into ``extra_body`` so that providers such as
    OpenRouter receive them correctly."""
    original_create = language_model._client.chat.completions.create

    @wraps(original_create)
    def patched_create(**kwargs):
        extra_body = kwargs.pop("extra_body", {}) or {}
        for key in ("reasoning",):
            if key in kwargs:
                extra_body[key] = kwargs.pop(key)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return original_create(**kwargs)

    language_model._client.chat.completions.create = patched_create


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
            model_kwargs = dict(
                model_id=self.model_id,
                api_key=self.api_key or "dummy-key",
                timeout=self.timeout,
            )
            if self.reasoning_effort:
                model_kwargs["reasoning"] = {"effort": self.reasoning_effort}
            language_model = openai_model_class(**model_kwargs)
            if self.reasoning_effort:
                _patch_client_for_extra_body(language_model)
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
