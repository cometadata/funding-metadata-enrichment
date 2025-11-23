"""Factory for provider implementations."""

from typing import Dict, Optional, Type

from funding_extractor.config.settings import ProviderSettings
from funding_extractor.exceptions import ProviderNotFoundError
from funding_extractor.providers.base import BaseProvider, ModelProvider
from funding_extractor.providers.gemini import GeminiProvider
from funding_extractor.providers.ollama import OllamaProvider
from funding_extractor.providers.openai import LocalOpenAIProvider, OpenAIProvider


class ProviderFactory:
    _providers: Dict[ModelProvider, Type[BaseProvider]] = {
        ModelProvider.GEMINI: GeminiProvider,
        ModelProvider.OLLAMA: OllamaProvider,
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.LOCAL_OPENAI: LocalOpenAIProvider,
    }

    @classmethod
    def create(cls, settings: ProviderSettings) -> BaseProvider:
        provider_cls: Optional[Type[BaseProvider]] = cls._providers.get(settings.provider)
        if provider_cls is None:
            raise ProviderNotFoundError(f"Provider '{settings.provider}' is not supported.")
        return provider_cls(
            model_id=settings.model_id,
            model_url=settings.model_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            debug=settings.debug,
        )
