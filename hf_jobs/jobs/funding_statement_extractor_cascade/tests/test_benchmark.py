import pytest

from benchmark_hf_job import (
    best_match_token_f1,
    score_predictions,
    strict_token_set_recall,
)


@pytest.mark.parametrize("pred,gold,want", [
    ("a b c", "a b c", 1.0),
    ("a b", "a b c", pytest.approx(0.8)),     # F1 of {a,b} vs {a,b,c}
    ("", "a b", 0.0),
    ("", "", 1.0),                              # empty/empty = perfect match
])
def test_best_match_token_f1(pred, gold, want):
    assert best_match_token_f1(pred, gold) == want


def test_strict_token_set_recall_full_match():
    assert strict_token_set_recall("a b c", "a b c") == 1.0


def test_strict_token_set_recall_partial():
    # gold tokens {a,b,c}, pred tokens {a,b} → recall=2/3
    assert strict_token_set_recall("a b", "a b c") == pytest.approx(2/3)


def test_score_predictions_aggregates_correctly():
    preds = [{"arxiv_id": "1", "funding_statement": "Funded by NSF."},
              {"arxiv_id": "2", "funding_statement": ""}]
    golds = [{"arxiv_id": "1", "gold_funding": "Funded by NSF grant 123."},
              {"arxiv_id": "2", "gold_funding": ""}]
    summary = score_predictions(preds, golds, id_col="arxiv_id",
                                  pred_col="funding_statement",
                                  gold_col="gold_funding")
    assert summary["n"] == 2
    assert "best_at_0.85_f1" in summary
    assert "strict_tsr_at_0.95_f1" in summary


def test_score_predictions_matches_reported_quality_on_synthetic_set():
    # Construct 20 synthetic gold/pred pairs where 19 match best@0.85 and 1 doesn't.
    preds = [{"arxiv_id": str(i), "funding_statement": f"Funded by grant {i} NSF NIH"}
              for i in range(20)]
    golds = [{"arxiv_id": str(i), "gold_funding": f"Funded by grant {i} NSF NIH"}
              for i in range(19)]
    golds.append({"arxiv_id": "19", "gold_funding": "Completely unrelated text here"})
    summary = score_predictions(preds, golds, id_col="arxiv_id",
                                  pred_col="funding_statement",
                                  gold_col="gold_funding")
    assert summary["best_at_0.85_f1"] == pytest.approx(19/20)
