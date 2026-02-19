import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.benchmark.evaluator import load_precomputed_predictions, match_predictions_to_gold
from funding_extractor.benchmark.dataset import GoldAward, GoldDocument, GoldFunder, normalize_doi


def test_load_precomputed_predictions_new_format(tmp_path):
    data = {
        "timestamp": "2025-01-01T00:00:00",
        "parameters": {"input_path": "/tmp", "input_format": "markdown"},
        "results": {
            "10.1234/test": {
                "funding_statements": [{"statement": "Funded by NSF", "score": 0.0, "query": "q"}],
                "extractions": [
                    {
                        "statement": "Funded by NSF",
                        "funders": [
                            {
                                "funder_name": "NSF",
                                "awards": [
                                    {"funding_scheme": [], "award_ids": ["123"], "award_title": []}
                                ],
                            }
                        ],
                    }
                ],
            }
        },
        "summary": {},
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    preds = load_precomputed_predictions(p)
    assert "10.1234/test" in preds
    assert preds["10.1234/test"]["statements"] == ["Funded by NSF"]
    assert len(preds["10.1234/test"]["funders"]) == 1
    assert preds["10.1234/test"]["funders"][0].funder_name == "NSF"


def test_match_predictions_to_gold_by_doi():
    gold_lookup = {
        "10.1234/a": GoldDocument(doi="10.1234/a", funding_statement="s", funders=[], markdown=""),
    }
    preds = {
        "10.1234/a": {"statements": ["s"], "funders": []},
    }
    matched, unmatched_gold, unmatched_pred = match_predictions_to_gold(preds, gold_lookup)
    assert len(matched) == 1
    assert matched[0][0] == "10.1234/a"
    assert unmatched_gold == 0
    assert unmatched_pred == 0


def test_match_predictions_to_gold_filename_fallback():
    gold_lookup = {
        "10.1234/test": GoldDocument(doi="10.1234/test", funding_statement="s", funders=[], markdown=""),
    }
    preds = {
        "10.1234-test.md": {"statements": ["s"], "funders": []},
    }
    matched, unmatched_gold, unmatched_pred = match_predictions_to_gold(preds, gold_lookup)
    assert len(matched) == 1


def test_match_predictions_unmatched():
    gold_lookup = {
        "10.1234/a": GoldDocument(doi="10.1234/a", funding_statement="s", funders=[], markdown=""),
    }
    preds = {
        "10.9999/z": {"statements": ["s"], "funders": []},
    }
    matched, unmatched_gold, unmatched_pred = match_predictions_to_gold(preds, gold_lookup)
    assert len(matched) == 1  # gold doc with empty prediction
    assert unmatched_pred == 1
