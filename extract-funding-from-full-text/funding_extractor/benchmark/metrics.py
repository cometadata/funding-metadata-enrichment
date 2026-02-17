# funding_extractor/benchmark/metrics.py
"""Multi-level metrics computation for benchmark evaluation."""

from dataclasses import dataclass, field
from typing import List

from funding_extractor.benchmark.dataset import GoldDocument, GoldFunder
from funding_extractor.benchmark.matching import (
    award_ids_match,
    best_match_score,
)
from funding_extractor.core.models import FunderEntity


@dataclass
class LevelMetrics:
    gold_count: int = 0
    pred_count: int = 0
    gold_matched: int = 0
    pred_matched: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f0_5: float = 0.0
    f1_5: float = 0.0


@dataclass
class DocumentMetrics:
    doi: str = ""
    statement: LevelMetrics = field(default_factory=LevelMetrics)
    funder: LevelMetrics = field(default_factory=LevelMetrics)
    award_id: LevelMetrics = field(default_factory=LevelMetrics)
    funding_scheme: LevelMetrics = field(default_factory=LevelMetrics)
    award_title: LevelMetrics = field(default_factory=LevelMetrics)


@dataclass
class AggregateMetrics:
    statement: LevelMetrics = field(default_factory=LevelMetrics)
    funder: LevelMetrics = field(default_factory=LevelMetrics)
    award_id: LevelMetrics = field(default_factory=LevelMetrics)
    funding_scheme: LevelMetrics = field(default_factory=LevelMetrics)
    award_title: LevelMetrics = field(default_factory=LevelMetrics)
    num_documents: int = 0
    num_documents_with_gold_funding: int = 0
    num_documents_with_pred_funding: int = 0


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def _build_level_metrics(gold_count: int, pred_count: int, gold_matched: int, pred_matched: int) -> LevelMetrics:
    precision = pred_matched / pred_count if pred_count else 0.0
    recall = gold_matched / gold_count if gold_count else 0.0
    f1 = _f_beta(precision, recall, 1.0)
    return LevelMetrics(
        gold_count=gold_count,
        pred_count=pred_count,
        gold_matched=gold_matched,
        pred_matched=pred_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        f0_5=_f_beta(precision, recall, 0.5),
        f1_5=_f_beta(precision, recall, 1.5),
    )


def compute_level_metrics(
    gold_items: List[str],
    pred_items: List[str],
    threshold: float,
    use_fuzzy: bool = True,
    id_match_mode: str = "normalized",
) -> LevelMetrics:
    gold_count = len(gold_items)
    pred_count = len(pred_items)
    gold_matched = 0
    pred_matched = 0

    if use_fuzzy:
        for g in gold_items:
            if best_match_score(g, pred_items) >= threshold:
                gold_matched += 1
        for p in pred_items:
            if best_match_score(p, gold_items) >= threshold:
                pred_matched += 1
    else:
        for g in gold_items:
            for p in pred_items:
                if award_ids_match(p, g, mode=id_match_mode):
                    gold_matched += 1
                    break
        for p in pred_items:
            for g in gold_items:
                if award_ids_match(p, g, mode=id_match_mode):
                    pred_matched += 1
                    break

    return _build_level_metrics(gold_count, pred_count, gold_matched, pred_matched)


def _flatten_gold_award_ids(funders: List[GoldFunder]) -> List[str]:
    ids = []
    for f in funders:
        for a in f.awards:
            ids.extend(a.award_ids)
    return ids


def _flatten_pred_award_ids(funders: List[FunderEntity]) -> List[str]:
    ids = []
    for f in funders:
        for a in f.awards:
            ids.extend(a.award_ids)
    return ids


def _flatten_gold_schemes(funders: List[GoldFunder]) -> List[str]:
    schemes = []
    for f in funders:
        for a in f.awards:
            schemes.extend(a.funding_schemes)
    return schemes


def _flatten_pred_schemes(funders: List[FunderEntity]) -> List[str]:
    schemes = []
    for f in funders:
        for a in f.awards:
            schemes.extend(a.funding_scheme)
    return schemes


def _flatten_gold_titles(funders: List[GoldFunder]) -> List[str]:
    titles = []
    for f in funders:
        for a in f.awards:
            titles.extend(a.award_titles)
    return titles


def _flatten_pred_titles(funders: List[FunderEntity]) -> List[str]:
    titles = []
    for f in funders:
        for a in f.awards:
            titles.extend(a.award_title)
    return titles


def evaluate_document(
    gold: GoldDocument,
    pred_statements: List[str],
    pred_funders: List[FunderEntity],
    threshold: float = 0.8,
    funder_threshold: float = 0.8,
    id_match_mode: str = "normalized",
) -> DocumentMetrics:
    # Statement level: gold is one string, treat as list of 1
    gold_stmts = [gold.funding_statement] if gold.funding_statement else []
    stmt_metrics = compute_level_metrics(gold_stmts, pred_statements, threshold, use_fuzzy=True)

    # Funder level: compare names (skip None funder names in gold)
    gold_funder_names = [f.funder_name for f in gold.funders if f.funder_name]
    pred_funder_names = [f.funder_name for f in pred_funders if f.funder_name]
    funder_metrics = compute_level_metrics(gold_funder_names, pred_funder_names, funder_threshold, use_fuzzy=True)

    # Award ID level: flatten and compare globally
    gold_ids = _flatten_gold_award_ids(gold.funders)
    pred_ids = _flatten_pred_award_ids(pred_funders)
    id_metrics = compute_level_metrics(gold_ids, pred_ids, threshold=0.0, use_fuzzy=False, id_match_mode=id_match_mode)

    # Funding scheme level
    gold_schemes = _flatten_gold_schemes(gold.funders)
    pred_schemes = _flatten_pred_schemes(pred_funders)
    scheme_metrics = compute_level_metrics(gold_schemes, pred_schemes, threshold, use_fuzzy=True)

    # Award title level
    gold_titles = _flatten_gold_titles(gold.funders)
    pred_titles = _flatten_pred_titles(pred_funders)
    title_metrics = compute_level_metrics(gold_titles, pred_titles, threshold, use_fuzzy=True)

    return DocumentMetrics(
        doi=gold.doi,
        statement=stmt_metrics,
        funder=funder_metrics,
        award_id=id_metrics,
        funding_scheme=scheme_metrics,
        award_title=title_metrics,
    )


def aggregate_metrics(doc_metrics: List[DocumentMetrics]) -> AggregateMetrics:
    levels = ["statement", "funder", "award_id", "funding_scheme", "award_title"]
    agg = AggregateMetrics(num_documents=len(doc_metrics))

    for level in levels:
        total_gold = sum(getattr(dm, level).gold_count for dm in doc_metrics)
        total_pred = sum(getattr(dm, level).pred_count for dm in doc_metrics)
        total_gold_matched = sum(getattr(dm, level).gold_matched for dm in doc_metrics)
        total_pred_matched = sum(getattr(dm, level).pred_matched for dm in doc_metrics)
        setattr(agg, level, _build_level_metrics(total_gold, total_pred, total_gold_matched, total_pred_matched))

    agg.num_documents_with_gold_funding = sum(1 for dm in doc_metrics if dm.statement.gold_count > 0)
    agg.num_documents_with_pred_funding = sum(1 for dm in doc_metrics if dm.statement.pred_count > 0)

    return agg
