"""Factory for provider implementations."""

from typing import Dict, Optional, Type

from funding_extractor.config.settings import ProviderSettings
from funding_extractor.exceptions import ProviderNotFoundError
from funding_extractor.providers.base import BaseProvider, ModelProvider
from funding_extractor.providers.openai import OpenAIProvider


def _import_vllm_provider():
    from funding_extractor.providers.vllm import VLLMProvider
    return VLLMProvider


class ProviderFactory:
    _providers: Dict[ModelProvider, Type[BaseProvider]] = {
        ModelProvider.OPENAI: OpenAIProvider,
    }

    @classmethod
    def create(cls, settings: ProviderSettings) -> BaseProvider:
        if settings.provider == ModelProvider.VLLM:
            provider_cls = _import_vllm_provider()
        else:
            provider_cls = cls._providers.get(settings.provider)

        if provider_cls is None:
            raise ProviderNotFoundError(f"Provider '{settings.provider}' is not supported.")

        kwargs = dict(
            model_id=settings.model_id,
            model_url=settings.model_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            debug=settings.debug,
            reasoning_effort=settings.reasoning_effort,
        )
        if settings.provider == ModelProvider.VLLM:
            kwargs["vllm_config_path"] = settings.vllm_config_path
            kwargs["lora_path"] = settings.lora_path

        return provider_cls(**kwargs)
