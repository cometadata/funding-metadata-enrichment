import os
from functools import wraps

import langextract as lx

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

        if self.model_url:
            os.environ["OPENAI_BASE_URL"] = self.model_url

        model_kwargs = dict(
            model_id=self.model_id,
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "dummy-key"),
            timeout=self.timeout,
        )
        if self.reasoning_effort:
            model_kwargs["reasoning"] = {"effort": self.reasoning_effort}

        language_model = lx.inference.OpenAILanguageModel(**model_kwargs)

        if self.reasoning_effort:
            _patch_client_for_extra_body(language_model)

        params["model"] = language_model
        return params
