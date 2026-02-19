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
from funding_extractor.entities.models import Award, FunderEntity


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


def test_level_metrics_duplicate_predictions_penalized():
    """Duplicate predictions should not inflate precision."""
    m = compute_level_metrics(
        gold_items=["NSF"],
        pred_items=["NSF", "NSF"],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.gold_matched == 1
    assert m.pred_matched == 1
    assert m.recall == 1.0
    assert m.precision == 0.5


def test_level_metrics_duplicate_gold_penalized():
    """Duplicate gold items should not inflate recall."""
    m = compute_level_metrics(
        gold_items=["NSF", "NSF"],
        pred_items=["NSF"],
        threshold=0.8,
        use_fuzzy=True,
    )
    assert m.gold_matched == 1
    assert m.pred_matched == 1
    assert m.precision == 1.0
    assert m.recall == 0.5


def test_evaluate_document_cross_funder_awards_not_matched():
    """Awards should only match within their matched funder, not globally."""
    gold = GoldDocument(
        doi="10.1234/test",
        funding_statement="Funded by NSF and NIH.",
        funders=[
            GoldFunder(
                funder_name="NSF",
                awards=[GoldAward(funding_schemes=[], award_ids=["123"], award_titles=[])],
            ),
            GoldFunder(
                funder_name="NIH",
                awards=[GoldAward(funding_schemes=[], award_ids=["456"], award_titles=[])],
            ),
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=[], award_ids=["456"], award_title=[])],
        ),
        FunderEntity(
            funder_name="NIH",
            awards=[Award(funding_scheme=[], award_ids=["123"], award_title=[])],
        ),
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by NSF and NIH."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.precision == 1.0
    assert dm.funder.recall == 1.0
    assert dm.award_id.gold_matched == 0
    assert dm.award_id.pred_matched == 0
    assert dm.award_id.precision == 0.0
    assert dm.award_id.recall == 0.0


def test_evaluate_document_per_funder_correct_awards():
    """Awards correctly assigned to funders should match."""
    gold = GoldDocument(
        doi="10.1234/test",
        funding_statement="Funded by NSF 123 and NIH 456.",
        funders=[
            GoldFunder(
                funder_name="NSF",
                awards=[GoldAward(funding_schemes=["CAREER"], award_ids=["123"], award_titles=[])],
            ),
            GoldFunder(
                funder_name="NIH",
                awards=[GoldAward(funding_schemes=[], award_ids=["456"], award_titles=[])],
            ),
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=["CAREER"], award_ids=["123"], award_title=[])],
        ),
        FunderEntity(
            funder_name="NIH",
            awards=[Award(funding_scheme=[], award_ids=["456"], award_title=[])],
        ),
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by NSF 123 and NIH 456."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
    )
    assert dm.funder.f1 == 1.0
    assert dm.award_id.f1 == 1.0
    assert dm.funding_scheme.f1 == 1.0


def test_evaluate_document_duplicate_funders_merged():
    """Duplicate pred funders with same name should merge before evaluation.

    If gold has NSF with awards [123, 456] and pred splits into two NSF entries,
    one with 123 and one with 456, all awards should still match after merging.
    """
    gold = GoldDocument(
        doi="10.1234/test",
        funding_statement="Funded by NSF.",
        funders=[
            GoldFunder(
                funder_name="NSF",
                awards=[GoldAward(funding_schemes=[], award_ids=["123", "456"], award_titles=[])],
            ),
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=[], award_ids=["123"], award_title=[])],
        ),
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=[], award_ids=["456"], award_title=[])],
        ),
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by NSF."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 1
    assert dm.funder.pred_count == 1
    assert dm.funder.f1 == 1.0
    assert dm.award_id.gold_count == 2
    assert dm.award_id.pred_count == 2
    assert dm.award_id.f1 == 1.0


def test_unnamed_funders_both_sides_matching_awards():
    """When both gold and pred have unnamed funders with matching award IDs,
    the awards should get credit."""
    gold = GoldDocument(
        doi="10.1234/unnamed",
        funding_statement="Funded by grant 123.",
        funders=[
            GoldFunder(
                funder_name=None,
                awards=[GoldAward(funding_schemes=[], award_ids=["123"], award_titles=[])],
            )
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name=None,
            awards=[Award(funding_scheme=[], award_ids=["123"], award_title=[])],
        )
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by grant 123."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 0
    assert dm.funder.pred_count == 0
    assert dm.award_id.gold_count == 1
    assert dm.award_id.pred_count == 1
    assert dm.award_id.gold_matched == 1
    assert dm.award_id.pred_matched == 1
    assert dm.award_id.f1 == 1.0


def test_unnamed_funders_both_sides_non_matching_awards():
    """When both gold and pred have unnamed funders but different award IDs,
    the awards should not match."""
    gold = GoldDocument(
        doi="10.1234/unnamed-mismatch",
        funding_statement="Funded by grant 123.",
        funders=[
            GoldFunder(
                funder_name=None,
                awards=[GoldAward(funding_schemes=[], award_ids=["123"], award_titles=[])],
            )
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name=None,
            awards=[Award(funding_scheme=[], award_ids=["999"], award_title=[])],
        )
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by grant 123."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 0
    assert dm.funder.pred_count == 0
    assert dm.award_id.gold_count == 1
    assert dm.award_id.pred_count == 1
    assert dm.award_id.gold_matched == 0
    assert dm.award_id.pred_matched == 0
    assert dm.award_id.f1 == 0.0


def test_unnamed_funder_only_in_gold():
    """When only gold has an unnamed funder, its awards should be false negatives."""
    gold = GoldDocument(
        doi="10.1234/gold-only",
        funding_statement="Funded by grant 123.",
        funders=[
            GoldFunder(
                funder_name=None,
                awards=[GoldAward(funding_schemes=[], award_ids=["123"], award_titles=[])],
            )
        ],
        markdown="",
    )
    pred_funders = []
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by grant 123."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 0
    assert dm.funder.pred_count == 0
    assert dm.award_id.gold_count == 1
    assert dm.award_id.pred_count == 0
    assert dm.award_id.gold_matched == 0
    assert dm.award_id.recall == 0.0


def test_unnamed_funder_only_in_pred():
    """When only pred has an unnamed funder, its awards should be false positives."""
    gold = GoldDocument(
        doi="10.1234/pred-only",
        funding_statement="Funded.",
        funders=[],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name=None,
            awards=[Award(funding_scheme=[], award_ids=["999"], award_title=[])],
        )
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 0
    assert dm.funder.pred_count == 0
    assert dm.award_id.gold_count == 0
    assert dm.award_id.pred_count == 1
    assert dm.award_id.pred_matched == 0
    assert dm.award_id.precision == 0.0


def test_named_and_unnamed_funders_mixed():
    """Named funder matching should be unaffected by unnamed funders.
    Unnamed funders should match each other separately for awards."""
    gold = GoldDocument(
        doi="10.1234/mixed",
        funding_statement="Funded by NSF and others.",
        funders=[
            GoldFunder(
                funder_name="NSF",
                awards=[GoldAward(funding_schemes=["CAREER"], award_ids=["NSF-001"], award_titles=[])],
            ),
            GoldFunder(
                funder_name=None,
                awards=[GoldAward(funding_schemes=[], award_ids=["ANON-42"], award_titles=[])],
            ),
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[Award(funding_scheme=["CAREER"], award_ids=["NSF-001"], award_title=[])],
        ),
        FunderEntity(
            funder_name=None,
            awards=[Award(funding_scheme=[], award_ids=["ANON-42"], award_title=[])],
        ),
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by NSF and others."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 1
    assert dm.funder.pred_count == 1
    assert dm.funder.f1 == 1.0
    assert dm.award_id.gold_count == 2
    assert dm.award_id.pred_count == 2
    assert dm.award_id.gold_matched == 2
    assert dm.award_id.f1 == 1.0
    assert dm.funding_scheme.gold_count == 1
    assert dm.funding_scheme.pred_count == 1
    assert dm.funding_scheme.f1 == 1.0


def test_unnamed_funders_matching_scheme_and_title():
    """Unnamed funders with matching funding_scheme and award_title should get credit."""
    gold = GoldDocument(
        doi="10.1234/scheme-title",
        funding_statement="Funded by a fellowship.",
        funders=[
            GoldFunder(
                funder_name=None,
                awards=[GoldAward(
                    funding_schemes=["Presidential Fellowship"],
                    award_ids=[],
                    award_titles=["Study of Climate Change"],
                )],
            )
        ],
        markdown="",
    )
    pred_funders = [
        FunderEntity(
            funder_name=None,
            awards=[Award(
                funding_scheme=["Presidential Fellowship"],
                award_ids=[],
                award_title=["Study of Climate Change"],
            )],
        )
    ]
    dm = evaluate_document(
        gold=gold,
        pred_statements=["Funded by a fellowship."],
        pred_funders=pred_funders,
        threshold=0.8,
        funder_threshold=0.8,
        id_match_mode="normalized",
    )
    assert dm.funder.gold_count == 0
    assert dm.funder.pred_count == 0
    assert dm.award_id.gold_count == 0
    assert dm.award_id.pred_count == 0
    assert dm.funding_scheme.gold_count == 1
    assert dm.funding_scheme.pred_count == 1
    assert dm.funding_scheme.gold_matched == 1
    assert dm.funding_scheme.f1 == 1.0
    assert dm.award_title.gold_count == 1
    assert dm.award_title.pred_count == 1
    assert dm.award_title.gold_matched == 1
    assert dm.award_title.f1 == 1.0


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
