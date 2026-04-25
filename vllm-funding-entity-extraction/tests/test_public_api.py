def test_public_api_imports():
    from funding_entity_extractor import (
        Award,
        Funder,
        StatementExtraction,
        extract_statements,
        extract_one,
        SYSTEM_PROMPT,
        USER_TEMPLATE,
    )
    assert callable(extract_statements)
    assert callable(extract_one)
    assert isinstance(SYSTEM_PROMPT, str)
    assert isinstance(USER_TEMPLATE, str)
