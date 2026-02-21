import argparse
import sys
from pathlib import Path

from funding_extractor.benchmark.evaluator import load_hf_predictions, run_benchmark
from funding_extractor.benchmark.report import push_metrics_to_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark funding extraction against a HuggingFace gold dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate pre-computed results against test split
  %(prog)s --predictions results.json --threshold 0.8 --output-json report.json

  # Evaluate against train split
  %(prog)s --predictions results.json --split train --threshold 0.8 --output-json report.json

  # Quick test with limited samples
  %(prog)s --predictions results.json --max-samples 10 --threshold 0.8 --output-json report.json

  # Verbose per-document breakdown
  %(prog)s --predictions results.json --threshold 0.8 --output-json report.json --verbose

  # Evaluate predictions from a HuggingFace dataset config
  %(prog)s --hf-predictions cometadata/funding-extraction-harness-benchmark \\
    --hf-config qwen3-8b-entities-non-thinking --split both --output-json report.json

  # Evaluate and push metrics to HuggingFace
  %(prog)s --hf-predictions cometadata/funding-extraction-harness-benchmark \\
    --hf-config qwen3-8b-entities-non-thinking --split both \\
    --push-to-hub cometadata/funding-extraction-harness-benchmark \\
    --push-config qwen3-8b-entities-non-thinking-metrics
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--predictions",
        type=Path,
        help="Path to pre-computed results JSON from the extractor",
    )
    mode_group.add_argument(
        "--statements-predictions",
        type=Path,
        help="Path to stage 1 JSONL output (evaluate statement extraction only)",
    )
    mode_group.add_argument(
        "--hf-predictions",
        help="HuggingFace dataset ID containing benchmark predictions (use with --hf-config)",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run full extraction pipeline against HF dataset (not yet implemented)",
    )
    parser.add_argument(
        "--hf-config",
        help="HuggingFace dataset config name for --hf-predictions",
    )

    parser.add_argument(
        "--dataset",
        default="cometadata/preprint-funding-pdfs-md-conversion",
        help="HuggingFace dataset ID (default: cometadata/preprint-funding-pdfs-md-conversion)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="test",
        help="Which split(s) to evaluate (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples (for quick testing)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy text matching (default: 0.8)",
    )
    parser.add_argument(
        "--funder-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for funder name matching (default: 0.8)",
    )
    parser.add_argument(
        "--id-match-mode",
        choices=["exact", "normalized"],
        default="normalized",
        help="How to match award IDs (default: normalized)",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to write JSON report",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-document breakdown in report")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output except errors")

    parser.add_argument(
        "--push-to-hub",
        help="Push metrics to this HuggingFace dataset ID",
    )
    parser.add_argument(
        "--push-config",
        help="Config name for pushed metrics (required with --push-to-hub)",
    )

    return parser.parse_args()


def _benchmark_kwargs(args: argparse.Namespace, split: str) -> dict:
    return dict(
        dataset_id=args.dataset,
        split=split,
        max_samples=args.max_samples,
        threshold=args.threshold,
        funder_threshold=args.funder_threshold,
        id_match_mode=args.id_match_mode,
        verbose=args.verbose,
        quiet=args.quiet,
    )


def _resolve_splits(split: str) -> list[str]:
    return ["train", "test"] if split == "both" else [split]


def _load_predictions(args: argparse.Namespace, split: str) -> dict:
    """Load predictions for a single split based on the chosen mode."""
    if args.hf_predictions:
        return load_hf_predictions(
            dataset_id=args.hf_predictions,
            config_name=args.hf_config,
            split=split,
        )

    if args.statements_predictions:
        from collections import defaultdict

        from funding_extractor.statements.io import read_statements_jsonl

        grouped = defaultdict(list)
        for record in read_statements_jsonl(args.statements_predictions):
            grouped[record["document_id"]].append(record["statement"])
        return {
            doc_id: {"statements": stmts, "funders": []}
            for doc_id, stmts in grouped.items()
        }

    # --predictions mode returns None; run_benchmark loads from file
    return None


def main() -> None:
    args = parse_args()

    if args.push_to_hub and not args.push_config:
        print("Error: --push-config is required when using --push-to-hub.")
        sys.exit(1)

    if args.hf_predictions and not args.hf_config:
        print("Error: --hf-config is required when using --hf-predictions.")
        sys.exit(1)

    if args.live:
        print("Error: --live mode is not yet implemented. Use --predictions with a pre-computed results file.")
        sys.exit(1)

    splits = _resolve_splits(args.split)
    reports: dict[str, dict] = {}

    for split in splits:
        predictions = _load_predictions(args, split)
        kwargs = _benchmark_kwargs(args, split)

        # Suffix output file with split name when evaluating multiple splits
        output_json = None
        if args.output_json:
            if len(splits) > 1:
                output_json = args.output_json.with_stem(
                    f"{args.output_json.stem}_{split}"
                )
            else:
                output_json = args.output_json

        if predictions is not None:
            report = run_benchmark(
                predictions=predictions, output_json=output_json, **kwargs
            )
        else:
            report = run_benchmark(
                predictions_path=args.predictions, output_json=output_json, **kwargs
            )

        reports[split] = report

    if reports and args.push_to_hub:
        push_metrics_to_hub(reports, args.push_to_hub, args.push_config)
        if not args.quiet:
            print(f"Metrics pushed to {args.push_to_hub} (config: {args.push_config})")


if __name__ == "__main__":
    main()
