class FundingStatementExtractorCascadeError(Exception):
    pass


class ConfigurationError(FundingStatementExtractorCascadeError):
    pass


class DocumentLoadError(FundingStatementExtractorCascadeError):
    pass


class ModelLoadError(FundingStatementExtractorCascadeError):
    pass


class ExtractionError(FundingStatementExtractorCascadeError):
    pass


class CheckpointError(FundingStatementExtractorCascadeError):
    pass


class ProcessingError(FundingStatementExtractorCascadeError):
    pass
