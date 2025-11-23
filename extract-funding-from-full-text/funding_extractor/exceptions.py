"""Project-specific exception hierarchy."""


class FundingExtractorError(Exception):
    """Base exception for the funding extractor package."""


class ConfigurationError(FundingExtractorError):
    """Raised when configuration values are invalid or missing."""


class ProviderConfigurationError(ConfigurationError):
    """Raised when provider configuration cannot be resolved."""


class ValidationError(FundingExtractorError):
    """Raised when inputs fail validation."""


class ProviderError(FundingExtractorError):
    """Base class for provider-related failures."""


class ProviderNotFoundError(ProviderError):
    """Raised when a provider cannot be constructed."""


class ProviderAPIError(ProviderError):
    """Raised when provider API calls fail."""


class ModelLoadError(FundingExtractorError):
    """Raised when models cannot be loaded."""


class DocumentLoadError(FundingExtractorError):
    """Raised when documents cannot be read."""


class ExtractionError(FundingExtractorError):
    """Raised when extraction fails."""


class CheckpointError(FundingExtractorError):
    """Raised when checkpoints cannot be read or written."""


class ProcessingError(FundingExtractorError):
    """Raised when processing logic encounters unrecoverable issues."""
