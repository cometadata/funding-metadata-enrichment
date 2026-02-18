# tests/test_models.py
import json
from funding_extractor.statements.models import FundingStatement
from funding_extractor.entities.models import Award, FunderEntity, ExtractionResult
from funding_extractor.models import DocumentResult, ProcessingParameters, ProcessingResults


def test_award_defaults():
    award = Award()
    assert award.funding_scheme == []
    assert award.award_ids == []
    assert award.award_title == []


def test_award_with_values():
    award = Award(
        funding_scheme=["Horizon 2020"],
        award_ids=["736937"],
        award_title=["My Project"],
    )
    assert award.funding_scheme == ["Horizon 2020"]
    assert award.award_ids == ["736937"]
    assert award.award_title == ["My Project"]


def test_funder_entity_defaults():
    funder = FunderEntity()
    assert funder.funder_name is None
    assert funder.awards == []


def test_funder_entity_with_awards():
    funder = FunderEntity(
        funder_name="NSF",
        awards=[
            Award(award_ids=["123"]),
            Award(funding_scheme=["CAREER"], award_ids=["456"]),
        ],
    )
    assert funder.funder_name == "NSF"
    assert len(funder.awards) == 2
    assert funder.awards[0].award_ids == ["123"]
    assert funder.awards[1].funding_scheme == ["CAREER"]


def test_funder_entity_none_name():
    funder = FunderEntity(funder_name=None, awards=[Award(award_ids=["789"])])
    assert funder.funder_name is None
    assert funder.awards[0].award_ids == ["789"]


def test_document_result_to_dict_nested_awards():
    doc = DocumentResult(
        filename="test.md",
        funding_statements=[
            FundingStatement(statement="Funded by NSF", score=30.0, query="funding_statement")
        ],
        extraction_results=[
            ExtractionResult(
                statement="Funded by NSF",
                funders=[
                    FunderEntity(
                        funder_name="NSF",
                        awards=[
                            Award(
                                funding_scheme=["CAREER"],
                                award_ids=["123", "456"],
                                award_title=[],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    d = doc.to_dict()
    funder_dict = d["extractions"][0]["funders"][0]
    assert funder_dict["funder_name"] == "NSF"
    assert "awards" in funder_dict
    assert funder_dict["awards"][0]["funding_scheme"] == ["CAREER"]
    assert funder_dict["awards"][0]["award_ids"] == ["123", "456"]
    assert funder_dict["awards"][0]["award_title"] == []
    # Old flat keys should NOT be present
    assert "funding_scheme" not in funder_dict
    assert "award_ids" not in funder_dict
    assert "award_title" not in funder_dict


def test_processing_results_from_dict_new_format():
    data = {
        "timestamp": "2025-01-01T00:00:00",
        "parameters": {"input_path": "/tmp", "input_format": "markdown"},
        "results": {
            "doc.md": {
                "funding_statements": [
                    {"statement": "Funded by NSF", "score": 30.0, "query": "q"}
                ],
                "extractions": [
                    {
                        "statement": "Funded by NSF",
                        "funders": [
                            {
                                "funder_name": "NSF",
                                "awards": [
                                    {
                                        "funding_scheme": ["CAREER"],
                                        "award_ids": ["123"],
                                        "award_title": [],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        },
        "summary": {},
    }
    results = ProcessingResults.from_dict(data)
    funder = results.results["doc.md"].extraction_results[0].funders[0]
    assert funder.funder_name == "NSF"
    assert len(funder.awards) == 1
    assert funder.awards[0].funding_scheme == ["CAREER"]
    assert funder.awards[0].award_ids == ["123"]


def test_processing_results_from_dict_old_format_backward_compat():
    """Old-format JSON (flat funder fields) should be converted to nested awards."""
    data = {
        "timestamp": "2025-01-01T00:00:00",
        "parameters": {"input_path": "/tmp", "input_format": "markdown"},
        "results": {
            "doc.md": {
                "funding_statements": [
                    {"statement": "Funded by NSF", "score": 30.0, "query": "q"}
                ],
                "extractions": [
                    {
                        "statement": "Funded by NSF",
                        "funders": [
                            {
                                "funder_name": "NSF",
                                "funding_scheme": "CAREER",
                                "award_ids": ["123", "456"],
                                "award_title": "My Grant",
                            }
                        ],
                    }
                ],
            }
        },
        "summary": {},
    }
    results = ProcessingResults.from_dict(data)
    funder = results.results["doc.md"].extraction_results[0].funders[0]
    assert funder.funder_name == "NSF"
    assert len(funder.awards) == 1
    assert funder.awards[0].funding_scheme == ["CAREER"]
    assert funder.awards[0].award_ids == ["123", "456"]
    assert funder.awards[0].award_title == ["My Grant"]


def test_processing_results_from_dict_old_format_null_fields():
    """Old format with null scheme/title should produce empty lists."""
    data = {
        "timestamp": "2025-01-01T00:00:00",
        "parameters": {"input_path": "/tmp", "input_format": "markdown"},
        "results": {
            "doc.md": {
                "funding_statements": [],
                "extractions": [
                    {
                        "statement": "Funded by NSF",
                        "funders": [
                            {
                                "funder_name": "NSF",
                                "funding_scheme": None,
                                "award_ids": ["123"],
                                "award_title": None,
                            }
                        ],
                    }
                ],
            }
        },
        "summary": {},
    }
    results = ProcessingResults.from_dict(data)
    funder = results.results["doc.md"].extraction_results[0].funders[0]
    assert funder.awards[0].funding_scheme == []
    assert funder.awards[0].award_title == []
    assert funder.awards[0].award_ids == ["123"]


def test_roundtrip_to_dict_from_dict():
    """Serialize to dict and back, ensure equivalence."""
    original = ProcessingResults(
        timestamp="2025-01-01T00:00:00",
        parameters=ProcessingParameters(input_path="/tmp"),
        results={
            "doc.md": DocumentResult(
                filename="doc.md",
                extraction_results=[
                    ExtractionResult(
                        statement="text",
                        funders=[
                            FunderEntity(
                                funder_name="NSF",
                                awards=[
                                    Award(funding_scheme=["CAREER"], award_ids=["1"], award_title=[]),
                                    Award(funding_scheme=[], award_ids=["2", "3"], award_title=["Title"]),
                                ],
                            ),
                            FunderEntity(
                                funder_name=None,
                                awards=[Award(award_ids=["4"])],
                            ),
                        ],
                    )
                ],
            )
        },
        summary={},
    )
    d = original.to_dict()
    restored = ProcessingResults.from_dict(d)
    f0 = restored.results["doc.md"].extraction_results[0].funders[0]
    f1 = restored.results["doc.md"].extraction_results[0].funders[1]
    assert f0.funder_name == "NSF"
    assert len(f0.awards) == 2
    assert f0.awards[0].funding_scheme == ["CAREER"]
    assert f0.awards[1].award_ids == ["2", "3"]
    assert f0.awards[1].award_title == ["Title"]
    assert f1.funder_name is None
    assert f1.awards[0].award_ids == ["4"]


def test_update_summary_includes_total_awards():
    results = ProcessingResults(
        timestamp="2025-01-01T00:00:00",
        parameters=ProcessingParameters(input_path="/tmp"),
        results={
            "doc.md": DocumentResult(
                filename="doc.md",
                funding_statements=[
                    FundingStatement(statement="text", score=1.0, query="q")
                ],
                extraction_results=[
                    ExtractionResult(
                        statement="text",
                        funders=[
                            FunderEntity(
                                funder_name="NSF",
                                awards=[Award(award_ids=["1"]), Award(award_ids=["2"])],
                            ),
                            FunderEntity(
                                funder_name="NIH",
                                awards=[Award(award_ids=["3"])],
                            ),
                        ],
                    )
                ],
            )
        },
        summary={},
    )
    results.update_summary()
    assert results.summary["total_funders"] == 2
    assert results.summary["total_awards"] == 3
