import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.benchmark.dataset import (
    GoldAward,
    GoldDocument,
    GoldFunder,
    normalize_doi,
    build_gold_lookup,
    _hf_row_to_gold_document,
)


def test_normalize_doi_basic():
    assert normalize_doi("10.1234/abcd") == "10.1234/abcd"


def test_normalize_doi_strips_url_prefix():
    assert normalize_doi("https://doi.org/10.1234/abcd") == "10.1234/abcd"


def test_normalize_doi_strips_doi_prefix():
    assert normalize_doi("doi:10.1234/abcd") == "10.1234/abcd"


def test_normalize_doi_lowercases():
    assert normalize_doi("10.1234/ABCD") == "10.1234/abcd"


def test_normalize_doi_strips_whitespace():
    assert normalize_doi("  10.1234/abcd  ") == "10.1234/abcd"


def test_hf_row_to_gold_document():
    row = {
        "doi": "10.48550/arxiv.2303.07677",
        "funding_statement": "Funded by NSF grant 123.",
        "funders": [
            {
                "funder_name": "NSF",
                "awards": [
                    {"funding_scheme": [], "award_ids": ["123"], "award_title": []}
                ],
            }
        ],
        "markdown": "# Paper\nContent here",
    }
    doc = _hf_row_to_gold_document(row)
    assert doc.doi == "10.48550/arxiv.2303.07677"
    assert doc.funding_statement == "Funded by NSF grant 123."
    assert doc.markdown == "# Paper\nContent here"
    assert len(doc.funders) == 1
    assert doc.funders[0].funder_name == "NSF"
    assert len(doc.funders[0].awards) == 1
    assert doc.funders[0].awards[0].award_ids == ["123"]


def test_hf_row_to_gold_document_null_funder_name():
    row = {
        "doi": "10.1234/test",
        "funding_statement": "Supported by grant 456.",
        "funders": [
            {
                "funder_name": None,
                "awards": [
                    {"funding_scheme": ["Program X"], "award_ids": ["456"], "award_title": []}
                ],
            }
        ],
        "markdown": "text",
    }
    doc = _hf_row_to_gold_document(row)
    assert doc.funders[0].funder_name is None
    assert doc.funders[0].awards[0].funding_schemes == ["Program X"]


def test_build_gold_lookup():
    docs = [
        GoldDocument(doi="10.1234/a", funding_statement="s1", funders=[], markdown=""),
        GoldDocument(doi="10.1234/B", funding_statement="s2", funders=[], markdown=""),
    ]
    lookup = build_gold_lookup(docs)
    assert "10.1234/a" in lookup
    assert "10.1234/b" in lookup  # lowercased
