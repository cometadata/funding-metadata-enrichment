# funding_extractor/benchmark/metrics.py
"""Multi-level metrics computation for benchmark evaluation."""

from dataclasses import dataclass, field
from typing import List, Tuple

from funding_extractor.benchmark.dataset import GoldDocument, GoldFunder
from funding_extractor.benchmark.matching import (
    greedy_match,
    greedy_match_ids,
    similarity,
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

    if use_fuzzy:
        matched_pairs = greedy_match(gold_items, pred_items, similarity, threshold)
    else:
        matched_pairs = greedy_match_ids(gold_items, pred_items, id_match_mode)

    matched_count = len(matched_pairs)
    return _build_level_metrics(gold_count, pred_count, matched_count, matched_count)


def _collect_awards_from_gold(funder: GoldFunder) -> Tuple[List[str], List[str], List[str]]:
    """Extract flat lists of (award_ids, funding_schemes, award_titles) from a gold funder."""
    ids, schemes, titles = [], [], []
    for a in funder.awards:
        ids.extend(a.award_ids)
        schemes.extend(a.funding_schemes)
        titles.extend(a.award_titles)
    return ids, schemes, titles


def _collect_awards_from_pred(funder: FunderEntity) -> Tuple[List[str], List[str], List[str]]:
    """Extract flat lists of (award_ids, funding_scheme, award_title) from a pred funder."""
    ids, schemes, titles = [], [], []
    for a in funder.awards:
        ids.extend(a.award_ids)
        schemes.extend(a.funding_scheme)
        titles.extend(a.award_title)
    return ids, schemes, titles


def _merge_gold_funders(funders: List[GoldFunder], threshold: float) -> List[GoldFunder]:
    """Merge gold funders with similar names, combining their awards."""
    from funding_extractor.benchmark.dataset import GoldAward
    groups: List[Tuple[str, List[GoldAward]]] = []
    unnamed_awards: List[GoldAward] = []

    for f in funders:
        if not f.funder_name:
            unnamed_awards.extend(f.awards)
            continue
        merged = False
        for i, (name, awards) in enumerate(groups):
            if similarity(f.funder_name, name) >= threshold:
                awards.extend(f.awards)
                merged = True
                break
        if not merged:
            groups.append((f.funder_name, list(f.awards)))

    result = [GoldFunder(funder_name=name, awards=awards) for name, awards in groups]
    if unnamed_awards:
        result.append(GoldFunder(funder_name=None, awards=unnamed_awards))
    return result


def _merge_pred_funders(funders: List[FunderEntity], threshold: float) -> List[FunderEntity]:
    """Merge predicted funders with similar names, combining their awards."""
    from funding_extractor.core.models import Award
    groups: List[Tuple[str, List[Award]]] = []
    unnamed_awards: List[Award] = []

    for f in funders:
        if not f.funder_name:
            unnamed_awards.extend(f.awards)
            continue
        merged = False
        for i, (name, awards) in enumerate(groups):
            if similarity(f.funder_name, name) >= threshold:
                awards.extend(f.awards)
                merged = True
                break
        if not merged:
            groups.append((f.funder_name, list(f.awards)))

    result = [FunderEntity(funder_name=name, awards=awards) for name, awards in groups]
    if unnamed_awards:
        result.append(FunderEntity(funder_name=None, awards=unnamed_awards))
    return result


def _evaluate_per_funder(
    gold_funders: List[GoldFunder],
    pred_funders: List[FunderEntity],
    funder_threshold: float,
    threshold: float,
    id_match_mode: str,
) -> Tuple[LevelMetrics, LevelMetrics, LevelMetrics, LevelMetrics]:
    """Merge funders by name, match 1-to-1, then evaluate awards within each pair.

    Returns (funder_metrics, id_metrics, scheme_metrics, title_metrics).
    """
    # Merge funders with similar names before evaluation
    gold_merged = _merge_gold_funders(gold_funders, funder_threshold)
    pred_merged = _merge_pred_funders(pred_funders, funder_threshold)

    # Filter to funders with names for name-based matching
    gold_named_indices = [i for i, f in enumerate(gold_merged) if f.funder_name]
    pred_named_indices = [i for i, f in enumerate(pred_merged) if f.funder_name]

    gold_named = [gold_merged[i].funder_name for i in gold_named_indices]
    pred_named = [pred_merged[i].funder_name for i in pred_named_indices]

    funder_matches = greedy_match(gold_named, pred_named, similarity, funder_threshold)

    funder_matched_count = len(funder_matches)
    funder_metrics = _build_level_metrics(
        len(gold_named), len(pred_named), funder_matched_count, funder_matched_count
    )

    # Map matched indices back to merged funder lists
    matched_gold_set: set = set()
    matched_pred_set: set = set()
    paired = []
    for gm_idx, pm_idx, _score in funder_matches:
        gi = gold_named_indices[gm_idx]
        pi = pred_named_indices[pm_idx]
        paired.append((gi, pi))
        matched_gold_set.add(gi)
        matched_pred_set.add(pi)

    # Pair unnamed funders for award-level evaluation (no funder-level credit)
    unnamed_gold_idx = next(
        (i for i, f in enumerate(gold_merged) if not f.funder_name), None
    )
    unnamed_pred_idx = next(
        (i for i, f in enumerate(pred_merged) if not f.funder_name), None
    )
    if unnamed_gold_idx is not None and unnamed_pred_idx is not None:
        paired.append((unnamed_gold_idx, unnamed_pred_idx))
        matched_gold_set.add(unnamed_gold_idx)
        matched_pred_set.add(unnamed_pred_idx)

    # Evaluate awards within each matched funder pair
    total_id_gold = total_id_pred = total_id_matched = 0
    total_scheme_gold = total_scheme_pred = total_scheme_matched = 0
    total_title_gold = total_title_pred = total_title_matched = 0

    for gi, pi in paired:
        g_ids, g_schemes, g_titles = _collect_awards_from_gold(gold_merged[gi])
        p_ids, p_schemes, p_titles = _collect_awards_from_pred(pred_merged[pi])

        id_matches = greedy_match_ids(g_ids, p_ids, id_match_mode)
        total_id_gold += len(g_ids)
        total_id_pred += len(p_ids)
        total_id_matched += len(id_matches)

        scheme_matches = greedy_match(g_schemes, p_schemes, similarity, threshold)
        total_scheme_gold += len(g_schemes)
        total_scheme_pred += len(p_schemes)
        total_scheme_matched += len(scheme_matches)

        title_matches = greedy_match(g_titles, p_titles, similarity, threshold)
        total_title_gold += len(g_titles)
        total_title_pred += len(p_titles)
        total_title_matched += len(title_matches)

    # Unmatched gold funders: their awards are false negatives
    for gi in range(len(gold_merged)):
        if gi not in matched_gold_set:
            g_ids, g_schemes, g_titles = _collect_awards_from_gold(gold_merged[gi])
            total_id_gold += len(g_ids)
            total_scheme_gold += len(g_schemes)
            total_title_gold += len(g_titles)

    # Unmatched pred funders: their awards are false positives
    for pi in range(len(pred_merged)):
        if pi not in matched_pred_set:
            p_ids, p_schemes, p_titles = _collect_awards_from_pred(pred_merged[pi])
            total_id_pred += len(p_ids)
            total_scheme_pred += len(p_schemes)
            total_title_pred += len(p_titles)

    id_metrics = _build_level_metrics(total_id_gold, total_id_pred, total_id_matched, total_id_matched)
    scheme_metrics = _build_level_metrics(total_scheme_gold, total_scheme_pred, total_scheme_matched, total_scheme_matched)
    title_metrics = _build_level_metrics(total_title_gold, total_title_pred, total_title_matched, total_title_matched)

    return funder_metrics, id_metrics, scheme_metrics, title_metrics


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

    # Funder + award levels: per-funder evaluation
    funder_metrics, id_metrics, scheme_metrics, title_metrics = _evaluate_per_funder(
        gold_funders=gold.funders,
        pred_funders=pred_funders,
        funder_threshold=funder_threshold,
        threshold=threshold,
        id_match_mode=id_match_mode,
    )

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
