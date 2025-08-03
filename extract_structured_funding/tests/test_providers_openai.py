"""Tests for OpenAI provider functionality."""
import os
from unittest.mock import patch

import pytest

from funding_extractor.providers import (
    ModelProvider,
    get_language_model_class,
    get_provider_config,
    validate_openai_model,
    validate_provider_requirements,
)


class TestOpenAIProvider:
    """Test OpenAI provider configuration and validation."""

    def test_openai_provider_config(self) -> None:
        """Test OpenAI provider configuration."""
        config = get_provider_config(ModelProvider.OPENAI)
        assert config.provider == ModelProvider.OPENAI
        assert config.default_model == "gpt-4o-mini"
        assert config.requires_api_key is True
        assert config.default_url is None

    def test_local_openai_provider_config(self) -> None:
        """Test local OpenAI provider configuration."""
        original_value = os.environ.pop("OPENAI_BASE_URL", None)
        try:
            config = get_provider_config(ModelProvider.LOCAL_OPENAI)
            assert config.provider == ModelProvider.LOCAL_OPENAI
            assert config.default_model == "gpt-4o-mini"
            assert config.requires_api_key is False
            assert config.default_url == "http://localhost:8000"
        finally:
            if original_value is not None:
                os.environ["OPENAI_BASE_URL"] = original_value

    def test_local_openai_provider_config_with_env(self) -> None:
        """Test local OpenAI provider uses environment variable for URL."""
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://custom:1234"}):
            config = get_provider_config(ModelProvider.LOCAL_OPENAI)
            assert config.default_url == "http://custom:1234"

    def test_get_language_model_class_openai(self) -> None:
        """Test getting OpenAI language model class."""
        import langextract as lx

        model_class = get_language_model_class(ModelProvider.OPENAI)
        assert model_class == lx.inference.OpenAILanguageModel

    def test_get_language_model_class_local_openai(self) -> None:
        """Test getting local OpenAI language model class."""
        import langextract as lx

        model_class = get_language_model_class(ModelProvider.LOCAL_OPENAI)
        assert model_class == lx.inference.OpenAILanguageModel


class TestOpenAIModelValidation:
    """Test OpenAI model validation."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("gpt-4o", True),
            ("gpt-4o-mini", True),
            ("gpt-4o-2024-11-20", True),
            ("gpt-4-turbo", True),
            ("gpt-4-turbo-preview", True),
            ("gpt-4", True),
            ("gpt-3.5-turbo", True),
            ("gpt-3.5-turbo-16k", True),
            ("o1-preview", True),
            ("o1-mini", True),
            ("o3-mini", True),
            ("invalid-model", False),
            ("claude-3", False),
            ("llama-2", False),
        ],
    )
    def test_validate_openai_model(self, model_id: str, expected: bool) -> None:
        """Test OpenAI model ID validation."""
        result = validate_openai_model(model_id)
        assert result == expected


class TestOpenAIProviderRequirements:
    """Test OpenAI provider requirement validation."""

    def test_openai_requires_api_key(self) -> None:
        """Test that OpenAI provider requires an API key."""
        with pytest.raises(ValueError) as exc_info:
            validate_provider_requirements(
                provider=ModelProvider.OPENAI,
                api_key=None,
                model_url=None,
            )
        assert "openai requires an API key" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_openai_with_api_key_passes(self) -> None:
        """Test that OpenAI provider passes with API key."""
        # Should not raise
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url=None,
        )

    def test_openai_model_validation(self) -> None:
        """Test OpenAI model validation in requirements check."""
        with pytest.raises(ValueError) as exc_info:
            validate_provider_requirements(
                provider=ModelProvider.OPENAI,
                api_key="test-key",
                model_url=None,
                model_id="invalid-model",
            )
        assert "does not appear to be a valid OpenAI model" in str(exc_info.value)

    def test_openai_valid_model_passes(self) -> None:
        """Test valid OpenAI model passes validation."""
        # Should not raise
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url=None,
            model_id="gpt-4o-mini",
        )

    def test_local_openai_no_api_key_required(self) -> None:
        """Test that local OpenAI provider doesn't require API key."""
        # Should not raise even without API key
        validate_provider_requirements(
            provider=ModelProvider.LOCAL_OPENAI,
            api_key=None,
            model_url="http://localhost:8000",
        )

    def test_skip_model_validation(self) -> None:
        """Test skipping model validation."""
        # Should not raise even with invalid model
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url=None,
            model_id="invalid-model",
            skip_model_validation=True,
        )

    def test_openai_custom_endpoint_skips_validation(self) -> None:
        """Test that custom OpenAI endpoints skip model validation."""
        # Should not raise even with non-standard model when using custom endpoint
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url="http://192.168.1.206:1234",
            model_id="qwen3:8b",
        )

    def test_openai_custom_https_endpoint_skips_validation(self) -> None:
        """Test that custom HTTPS OpenAI endpoints skip model validation."""
        # Should not raise for custom HTTPS endpoints
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url="https://custom.example.com/v1",
            model_id="custom-model-7b",
        )

    def test_openai_localhost_endpoint_skips_validation(self) -> None:
        """Test that localhost OpenAI endpoints skip model validation."""
        # Should not raise for localhost endpoints
        validate_provider_requirements(
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            model_url="http://localhost:8000",
            model_id="local-llama-3",
        )

    def test_openai_default_api_validates_model(self) -> None:
        """Test that default OpenAI API still validates models."""
        # Should raise for invalid model when using default API
        with pytest.raises(ValueError) as exc_info:
            validate_provider_requirements(
                provider=ModelProvider.OPENAI,
                api_key="test-key",
                model_url="https://api.openai.com/v1",
                model_id="invalid-model",
            )
        assert "does not appear to be a valid OpenAI model" in str(exc_info.value)
