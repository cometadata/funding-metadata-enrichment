class FundingStatementExtractorError(Exception):
    pass


class ConfigurationError(FundingStatementExtractorError):
    pass


class ValidationError(FundingStatementExtractorError):
    pass


class ModelLoadError(FundingStatementExtractorError):
    pass


class DocumentLoadError(FundingStatementExtractorError):
    pass


class ExtractionError(FundingStatementExtractorError):
    pass


class CheckpointError(FundingStatementExtractorError):
    pass


class ProcessingError(FundingStatementExtractorError):
    pass
