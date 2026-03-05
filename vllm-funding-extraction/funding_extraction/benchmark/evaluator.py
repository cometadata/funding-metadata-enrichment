from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from funding_extraction.benchmark.dataset import (
    GoldDocument,
    build_gold_lookup,
    load_hf_dataset,
    normalize_doi,
)
from funding_extraction.benchmark.matching import doi_from_filename
from funding_extraction.benchmark.metrics import (
    AggregateMetrics,
    DocumentMetrics,
    aggregate_metrics,
    evaluate_document,
)
from funding_extraction.benchmark.report import (
    build_report,
    print_console_summary,
    save_report,
)
from funding_extraction.models import Award, FunderEntity


def load_hf_predictions(
    dataset_id: str,
    config_name: str,
    split: str = "test",
) -> Dict[str, Dict[str, Any]]:
    """Load predictions from a HuggingFace dataset config pushed by the benchmark job."""
    from datasets import load_dataset

    if split == "both":
        splits_to_load = ["train", "test"]
    else:
        splits_to_load = [split]

    preds: Dict[str, Dict[str, Any]] = {}
    for s in splits_to_load:
        ds = load_dataset(dataset_id, config_name, split=s)
        for row in ds:
            doi = row["doi"]
            funders: List[FunderEntity] = []
            for f in row.get("funders", []):
                awards = [
                    Award(
                        funding_scheme=a.get("funding_scheme", []),
                        award_ids=a.get("award_ids", []),
                        award_title=a.get("award_title", []),
                    )
                    for a in f.get("awards", [])
                ]
                funders.append(FunderEntity(funder_name=f.get("funder_name"), awards=awards))
            preds[doi] = {"statements": [], "funders": funders}

    return preds


def match_predictions_to_gold(
    predictions: Dict[str, Dict[str, Any]],
    gold_lookup: Dict[str, GoldDocument],
) -> Tuple[List[Tuple[str, GoldDocument, Dict[str, Any]]], int, int]:
    matched: List[Tuple[str, GoldDocument, Dict[str, Any]]] = []
    matched_gold_dois = set()
    matched_pred_keys = set()

    for pred_key, pred_data in predictions.items():
        norm_key = normalize_doi(pred_key)
        if norm_key in gold_lookup:
            matched.append((norm_key, gold_lookup[norm_key], pred_data))
            matched_gold_dois.add(norm_key)
            matched_pred_keys.add(pred_key)
            continue

        doi = doi_from_filename(pred_key)
        if doi:
            norm_doi = normalize_doi(doi)
            if norm_doi in gold_lookup:
                matched.append((norm_doi, gold_lookup[norm_doi], pred_data))
                matched_gold_dois.add(norm_doi)
                matched_pred_keys.add(pred_key)
                continue

    for gold_doi, gold_doc in gold_lookup.items():
        if gold_doi not in matched_gold_dois:
            matched.append((gold_doi, gold_doc, {"statements": [], "funders": []}))

    unmatched_gold = len(gold_lookup) - len(matched_gold_dois)
    unmatched_pred = len(predictions) - len(matched_pred_keys)

    return matched, unmatched_gold, unmatched_pred


def run_benchmark(
    dataset_id: str = "cometadata/preprint-funding-pdfs-md-conversion",
    split: str = "test",
    max_samples: Optional[int] = None,
    predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.8,
    funder_threshold: float = 0.8,
    id_match_mode: str = "normalized",
    output_json: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    gold_documents = load_hf_dataset(dataset_id, split, max_samples)
    gold_lookup = build_gold_lookup(gold_documents)

    if predictions is None:
        raise ValueError("predictions must be provided (use --hf-predictions)")

    mode = "hf-predictions"

    matched_pairs, unmatched_gold, unmatched_pred = match_predictions_to_gold(predictions, gold_lookup)

    doc_metrics: List[DocumentMetrics] = []
    for doi, gold_doc, pred_data in matched_pairs:
        dm = evaluate_document(
            gold=gold_doc,
            pred_statements=pred_data["statements"],
            pred_funders=pred_data["funders"],
            threshold=threshold,
            funder_threshold=funder_threshold,
            id_match_mode=id_match_mode,
        )
        doc_metrics.append(dm)

    agg = aggregate_metrics(doc_metrics)

    matched_count = len(matched_pairs) - unmatched_gold
    total_predicted = len(predictions)
    total_gold = len(gold_documents)

    if not quiet:
        print_console_summary(
            aggregate=agg,
            dataset_id=dataset_id,
            split=split,
            threshold=threshold,
            funder_threshold=funder_threshold,
            mode=mode,
            total_gold=total_gold,
            total_predicted=total_predicted,
            matched_count=matched_count,
        )

    report = build_report(
        dataset_id=dataset_id,
        split=split,
        threshold=threshold,
        funder_threshold=funder_threshold,
        id_match_mode=id_match_mode,
        mode=mode,
        total_gold=total_gold,
        total_predicted=total_predicted,
        matched_count=matched_count,
        aggregate=agg,
        doc_metrics=doc_metrics if verbose else None,
        verbose=verbose,
    )

    if output_json:
        save_report(report, output_json)
        if not quiet:
            print(f"Report saved to {output_json}")

    return report
