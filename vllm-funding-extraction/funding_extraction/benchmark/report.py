import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from funding_extraction.benchmark.metrics import AggregateMetrics, DocumentMetrics, LevelMetrics


def _level_to_dict(m: LevelMetrics) -> Dict[str, Any]:
    return {
        "gold_count": m.gold_count,
        "pred_count": m.pred_count,
        "gold_matched": m.gold_matched,
        "pred_matched": m.pred_matched,
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "f1": round(m.f1, 4),
        "f0_5": round(m.f0_5, 4),
        "f1_5": round(m.f1_5, 4),
    }


def build_report(
    dataset_id: str,
    split: str,
    threshold: float,
    funder_threshold: float,
    id_match_mode: str,
    mode: str,
    total_gold: int,
    total_predicted: int,
    matched_count: int,
    aggregate: AggregateMetrics,
    doc_metrics: Optional[List[DocumentMetrics]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "benchmark_config": {
            "dataset": dataset_id,
            "split": split,
            "threshold": threshold,
            "funder_threshold": funder_threshold,
            "id_match_mode": id_match_mode,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "total_gold_documents": total_gold,
            "total_predicted_documents": total_predicted,
            "matched_documents": matched_count,
            "unmatched_gold": total_gold - matched_count,
            "unmatched_predictions": total_predicted - matched_count,
        },
        "aggregate_metrics": {
            "statement": _level_to_dict(aggregate.statement),
            "funder": _level_to_dict(aggregate.funder),
            "award_id": _level_to_dict(aggregate.award_id),
            "funding_scheme": _level_to_dict(aggregate.funding_scheme),
            "award_title": _level_to_dict(aggregate.award_title),
        },
    }
    if verbose and doc_metrics:
        report["per_document"] = [
            {
                "doi": dm.doi,
                "statement": _level_to_dict(dm.statement),
                "funder": _level_to_dict(dm.funder),
                "award_id": _level_to_dict(dm.award_id),
                "funding_scheme": _level_to_dict(dm.funding_scheme),
                "award_title": _level_to_dict(dm.award_title),
            }
            for dm in doc_metrics
        ]
    return report


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def push_metrics_to_hub(
    reports: Dict[str, Dict[str, Any]],
    dataset_id: str,
    config_name: str,
) -> None:
    """Push aggregate metrics to HuggingFace as a dataset config.

    Args:
        reports: Mapping of split name to report dict (e.g. {"train": ..., "test": ...}).
        dataset_id: HuggingFace dataset ID.
        config_name: Config name for the pushed metrics.
    """
    from datasets import Dataset, DatasetDict

    split_datasets = {}
    for split_name, report in reports.items():
        rows = []
        for level, metrics in report["aggregate_metrics"].items():
            rows.append({"level": level, **metrics})
        split_datasets[split_name] = Dataset.from_list(rows)

    dd = DatasetDict(split_datasets)
    dd.push_to_hub(dataset_id, config_name=config_name)


def _fmt(val: float) -> str:
    return f"{val:.4f}"


def print_console_summary(
    aggregate: AggregateMetrics,
    dataset_id: str,
    split: str,
    threshold: float,
    funder_threshold: float,
    mode: str,
    total_gold: int,
    total_predicted: int,
    matched_count: int,
) -> None:
    print(f"\nBenchmark: {dataset_id} ({split} split)")
    print(f"Mode: {mode} | Threshold: {threshold:.2f} | Funder Threshold: {funder_threshold:.2f}")
    print(f"\nDocuments:  {total_gold} gold | {total_predicted} predicted | {matched_count} matched")

    header = f"{'Level':<20} {'Gold':>6} {'Pred':>6} {'P':>8} {'R':>8} {'F1':>8} {'F0.5':>8} {'F1.5':>8}"
    print(f"\n{header}")
    print("\u2500" * len(header))

    levels = [
        ("Statement", aggregate.statement),
        ("Funder", aggregate.funder),
        ("Award ID", aggregate.award_id),
        ("Funding Scheme", aggregate.funding_scheme),
        ("Award Title", aggregate.award_title),
    ]
    for label, m in levels:
        print(
            f"{label:<20} {m.gold_count:>6} {m.pred_count:>6} "
            f"{_fmt(m.precision):>8} {_fmt(m.recall):>8} {_fmt(m.f1):>8} "
            f"{_fmt(m.f0_5):>8} {_fmt(m.f1_5):>8}"
        )
    print()
