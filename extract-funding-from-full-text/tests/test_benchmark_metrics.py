# tests/test_benchmark_metrics.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.benchmark.metrics import (
    LevelMetrics,
    compute_level_metrics,
    evaluate_document,
    aggregate_metrics,
)
from funding_extractor.benchmark.dataset import GoldAward, GoldDocument, GoldFunder
from funding_extractor.core.models import Award, FunderEntity


def test_level_metrics_perfect():
    m = compute_level_metrics(
        gold_items=["a", "b"],
        pred_items=["a", "b"],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_level_metrics_no_predictions():
    m = compute_level_metrics(
        gold_items=["a", "b"],
        pred_items=[],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.gold_count == 2
    assert m.pred_count == 0


def test_level_metrics_no_gold():
    m = compute_level_metrics(
        gold_items=[],
        pred_items=["a"],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.recall == 0.0
    assert m.precision == 0.0
    assert m.gold_count == 0
    assert m.pred_count == 1


def test_level_metrics_partial_match():
    m = compute_level_metrics(
        gold_items=["a", "b", "c"],
        pred_items=["a", "b"],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.gold_matched == 2
    assert m.pred_matched == 2
    assert m.recall < 1.0
    assert m.precision == 1.0


def test_evaluate_document_all_levels():
    gold = GoldDocument(
        doi="10.1234/test",
        funding_statement="Funded by NSF grant 123.",
        funders=[
            GoldFunder(
                funder_name="NSF",
                awards=[GoldAward(funding_schemes=[], award_ids=["123"], award_titles=[])],
            )
        ],
        markdown="",
    )
    pred_statements = ["Funded by NSF grant 123."]
    pred_funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=[], award_ids=["123"], award_title=[])],
        )
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=pred_statements,
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.doi == "10.1234/test"
    assert dm.statement.recall == 1.0
    assert dm.funder.recall == 1.0
    assert dm.award_id.recall == 1.0


def test_aggregate_metrics_micro_average():
    m1 = LevelMetrics(gold_count=2, pred_count=2, gold_matched=2, pred_matched=2,
                       precision=1.0, recall=1.0, f1=1.0, f0_5=1.0, f1_5=1.0)
    m2 = LevelMetrics(gold_count=2, pred_count=2, gold_matched=0, pred_matched=0,
                       precision=0.0, recall=0.0, f1=0.0, f0_5=0.0, f1_5=0.0)

    from funding_extractor.benchmark.metrics import DocumentMetrics
    docs = [
        DocumentMetrics(doi="a", statement=m1, funder=m1, award_id=m1, funding_scheme=m1, award_title=m1),
        DocumentMetrics(doi="b", statement=m2, funder=m2, award_id=m2, funding_scheme=m2, award_title=m2),
    ]
    agg = aggregate_metrics(docs)
    assert agg.statement.gold_count == 4
    assert agg.statement.gold_matched == 2
    assert agg.statement.recall == 0.5
    assert agg.statement.precision == 0.5
