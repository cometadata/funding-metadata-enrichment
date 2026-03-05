import argparse
import sys
from pathlib import Path

from funding_extraction.benchmark.evaluator import load_hf_predictions, run_benchmark
from funding_extraction.benchmark.report import push_metrics_to_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark funding extraction against a HuggingFace gold dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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

    parser.add_argument(
        "--hf-predictions",
        required=True,
        help="HuggingFace dataset ID containing benchmark predictions (use with --hf-config)",
    )
    parser.add_argument(
        "--hf-config",
        required=True,
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


def main() -> None:
    args = parse_args()

    if args.push_to_hub and not args.push_config:
        print("Error: --push-config is required when using --push-to-hub.")
        sys.exit(1)

    splits = _resolve_splits(args.split)
    reports: dict[str, dict] = {}

    for split in splits:
        predictions = load_hf_predictions(
            dataset_id=args.hf_predictions,
            config_name=args.hf_config,
            split=split,
        )
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

        report = run_benchmark(
            predictions=predictions, output_json=output_json, **kwargs
        )

        reports[split] = report

    if reports and args.push_to_hub:
        push_metrics_to_hub(reports, args.push_to_hub, args.push_config)
        if not args.quiet:
            print(f"Metrics pushed to {args.push_to_hub} (config: {args.push_config})")


if __name__ == "__main__":
    main()
