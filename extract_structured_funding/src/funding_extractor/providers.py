import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

import langextract as lx


class ModelProvider(str, Enum):
    """Supported model providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL_OPENAI = "local_openai"


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    provider: ModelProvider
    default_model: str
    requires_api_key: bool
    default_url: str | None = None


def get_provider_config(provider: ModelProvider) -> ProviderConfig:
    """Get configuration for a specific provider.

    Args:
        provider: The model provider

    Returns:
        Provider configuration
    """
    configs = {
        ModelProvider.GEMINI: ProviderConfig(
            provider=ModelProvider.GEMINI,
            default_model="gemini-2.5-flash-lite",
            requires_api_key=True,
            default_url=None,
        ),
        ModelProvider.OLLAMA: ProviderConfig(
            provider=ModelProvider.OLLAMA,
            default_model="llama3.2",
            requires_api_key=False,
            default_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ),
        ModelProvider.OPENAI: ProviderConfig(
            provider=ModelProvider.OPENAI,
            default_model="gpt-4o-mini",
            requires_api_key=True,
            default_url=None,
        ),
        ModelProvider.LOCAL_OPENAI: ProviderConfig(
            provider=ModelProvider.LOCAL_OPENAI,
            default_model="gpt-4o-mini",
            requires_api_key=False,
            default_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000"),
        ),
    }
    return configs[provider]


def get_language_model_class(provider: ModelProvider) -> Any:
    """Get the langextract language model class for a provider.

    Args:
        provider: The model provider

    Returns:
        The appropriate language model class
    """
    if provider == ModelProvider.GEMINI:
        return lx.inference.GeminiLanguageModel
    elif provider == ModelProvider.OLLAMA:
        return lx.inference.OllamaLanguageModel
    elif provider in (ModelProvider.OPENAI, ModelProvider.LOCAL_OPENAI):
        return lx.inference.OpenAILanguageModel
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_available_ollama_models(model_url: str | None = None) -> list[str]:
    """Get list of available models from Ollama.

    Args:
        model_url: Ollama API URL (defaults to http://localhost:11434)

    Returns:
        List of available model names

    Raises:
        ConnectionError: If cannot connect to Ollama
    """
    url = model_url or "http://localhost:11434"
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Cannot connect to Ollama at {url}: {e}") from e


def validate_gemini_model(model_id: str) -> bool:
    """Validate that a Gemini model ID is in the expected format.

    Args:
        model_id: The model ID to validate

    Returns:
        True if the model ID appears valid
    """
    valid_patterns = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-pro-vision",
    ]

    for pattern in valid_patterns:
        if model_id.startswith(pattern):
            return True

    import re

    pattern = r"^gemini-\d+\.\d+-[a-z]+(-[a-z]+)*$"
    return bool(re.match(pattern, model_id.lower()))


def validate_openai_model(model_id: str) -> bool:
    """Validate that an OpenAI model ID is in the expected format.

    Args:
        model_id: The model ID to validate

    Returns:
        True if the model ID appears valid
    """
    valid_patterns = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
        "o3-mini",
    ]

    for pattern in valid_patterns:
        if model_id == pattern or model_id.startswith(f"{pattern}-"):
            return True

    return False


def validate_ollama_model(model_id: str, model_url: str | None = None) -> None:
    """Validate that an Ollama model is available.

    Args:
        model_id: The model ID to validate
        model_url: Ollama API URL

    Raises:
        ValueError: If model is not available
        ConnectionError: If cannot connect to Ollama
    """
    try:
        available_models = get_available_ollama_models(model_url)

        model_base = model_id.split(":")[0]

        model_found = any(
            model == model_id or model.startswith(f"{model_base}:")
            for model in available_models
        )

        if not model_found:
            available_str = (
                ", ".join(sorted(available_models)) if available_models else "none"
            )
            raise ValueError(
                f"Model '{model_id}' is not available in Ollama. "
                f"Available models: {available_str}. "
                f"Install it with: ollama pull {model_id}"
            )
    except ConnectionError:
        raise


def validate_provider_requirements(
    provider: ModelProvider,
    api_key: str | None,
    model_url: str | None,
    model_id: str | None = None,
    skip_model_validation: bool = False,
) -> None:
    """Validate that provider requirements are met.

    Args:
        provider: The model provider
        api_key: Optional API key
        model_url: Optional model URL
        model_id: Optional model ID to validate
        skip_model_validation: Skip model validation checks

    Raises:
        ValueError: If requirements are not met
        ConnectionError: If cannot connect to provider (Ollama)
    """
    config = get_provider_config(provider)

    if config.requires_api_key and not api_key:
        env_var_name = (
            "OPENAI_API_KEY" if provider == ModelProvider.OPENAI else "GEMINI_API_KEY"
        )
        raise ValueError(
            f"{provider.value} requires an API key. "
            f"Set {env_var_name} environment variable or use --api-key"
        )

    if model_id and not skip_model_validation:
        if provider == ModelProvider.OLLAMA:
            validate_ollama_model(model_id, model_url)
        elif provider == ModelProvider.GEMINI:
            if not validate_gemini_model(model_id):
                raise ValueError(
                    f"Model '{model_id}' does not appear to be a valid Gemini model. "
                    f"Expected format: gemini-X.X-[flash|pro|lite] "
                    f"(e.g., gemini-2.5-flash, gemini-2.5-pro)"
                )
        elif provider == ModelProvider.OPENAI:
            is_custom_endpoint = model_url and not model_url.startswith(
                "https://api.openai.com"
            )
            if not is_custom_endpoint and not validate_openai_model(model_id):
                raise ValueError(
                    f"Model '{model_id}' does not appear to be a valid OpenAI model. "
                    f"Expected format: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc."
                )
