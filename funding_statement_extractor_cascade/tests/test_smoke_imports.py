def test_package_imports():
    import funding_statement_extractor_cascade
    from funding_statement_extractor_cascade import exceptions

    assert issubclass(exceptions.ConfigurationError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.DocumentLoadError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.ModelLoadError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.ExtractionError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.CheckpointError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.ProcessingError, exceptions.FundingStatementExtractorCascadeError)


def test_top_level_public_api():
    import funding_statement_extractor_cascade as pkg

    assert hasattr(pkg, "CascadeConfig")
    assert hasattr(pkg, "CascadeResult")
    assert hasattr(pkg, "DocPayload")
    assert hasattr(pkg, "BatchResult")
    assert hasattr(pkg, "__version__")
