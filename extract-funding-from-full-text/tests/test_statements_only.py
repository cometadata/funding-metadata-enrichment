import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.cli.main import process_document_task
from funding_extractor.config.settings import (
    ApplicationConfig,
    ConfigPaths,
    ExtractionSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    ProviderSettings,
    RuntimeSettings,
)
from funding_extractor.io.loaders import DocumentPayload


def _make_config(statements_only=False, normalize=False):
    return ApplicationConfig(
        input=InputSettings(path=Path("/tmp/fake")),
        output=OutputSettings(output_path=Path("/tmp/out.json"), checkpoint_path=Path("/tmp/cp")),
        extraction=ExtractionSettings(),
        processing=ProcessingSettings(
            statements_only=statements_only,
            normalize=normalize,
        ),
        provider=ProviderSettings(),
        runtime=RuntimeSettings(),
        config_paths=ConfigPaths(),
    )


def _make_document(text, doc_id="test-doc"):
    return DocumentPayload(
        document_id=doc_id,
        content=text,
        file_path=None,
        checkpoint_key=f"test:{doc_id}",
    )


@patch("funding_extractor.cli.main.extract_structured_entities")
def test_statements_only_creates_single_statement(mock_extract):
    from funding_extractor.entities.models import ExtractionResult
    mock_extract.return_value = ExtractionResult(statement="mock", funders=[])

    config = _make_config(statements_only=True)
    doc = _make_document("This work was funded by NIH grant R01-GM123456.")
    result = process_document_task(doc, config, queries={})

    assert result is not None
    assert len(result.funding_statements) == 1

    stmt = result.funding_statements[0]
    assert stmt.statement == "This work was funded by NIH grant R01-GM123456."
    assert stmt.score == 1.0
    assert stmt.query == "statements-only"


def test_statements_only_skips_empty_content():
    config = _make_config(statements_only=True)
    doc = _make_document("   ")
    result = process_document_task(doc, config, queries={})

    assert result is not None
    assert result.funding_statements == []
    assert result.extraction_results == []


@patch("funding_extractor.cli.main.extract_structured_entities")
def test_statements_only_produces_statements_no_extractions(mock_extract):
    """statements-only mode produces statements and attempts extraction."""
    from funding_extractor.entities.models import ExtractionResult
    mock_extract.return_value = ExtractionResult(statement="mock", funders=[])

    config = _make_config(statements_only=True)
    doc = _make_document("Funded by the European Research Council.")
    result = process_document_task(doc, config, queries={})

    assert result is not None
    assert len(result.funding_statements) == 1
