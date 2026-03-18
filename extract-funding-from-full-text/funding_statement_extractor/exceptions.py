
class FundingExtractorError(Exception):
    """Base exception for the funding statement extractor package."""


class ConfigurationError(FundingExtractorError):
    """Raised when configuration values are invalid or missing."""


class ValidationError(FundingExtractorError):
    """Raised when inputs fail validation."""


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
