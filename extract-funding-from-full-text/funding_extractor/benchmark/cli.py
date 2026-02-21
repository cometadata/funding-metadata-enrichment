import argparse
import sys
from pathlib import Path

from funding_extractor.benchmark.evaluator import load_hf_predictions, run_benchmark


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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_predictions:
        if not args.hf_config:
            print("Error: --hf-config is required when using --hf-predictions.")
            sys.exit(1)

        predictions = load_hf_predictions(
            dataset_id=args.hf_predictions,
            config_name=args.hf_config,
            split=args.split,
        )
        run_benchmark(
            dataset_id=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            predictions=predictions,
            threshold=args.threshold,
            funder_threshold=args.funder_threshold,
            id_match_mode=args.id_match_mode,
            output_json=args.output_json,
            verbose=args.verbose,
            quiet=args.quiet,
        )
        return

    if args.statements_predictions:
        from funding_extractor.statements.io import read_statements_jsonl
        from collections import defaultdict

        grouped = defaultdict(list)
        for record in read_statements_jsonl(args.statements_predictions):
            grouped[record["document_id"]].append(record["statement"])

        predictions = {
            doc_id: {"statements": stmts, "funders": []}
            for doc_id, stmts in grouped.items()
        }

        run_benchmark(
            dataset_id=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            predictions=predictions,
            threshold=args.threshold,
            funder_threshold=args.funder_threshold,
            id_match_mode=args.id_match_mode,
            output_json=args.output_json,
            verbose=args.verbose,
            quiet=args.quiet,
        )
        return

    if args.live:
        print("Error: --live mode is not yet implemented. Use --predictions with a pre-computed results file.")
        sys.exit(1)

    run_benchmark(
        dataset_id=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        predictions_path=args.predictions,
        threshold=args.threshold,
        funder_threshold=args.funder_threshold,
        id_match_mode=args.id_match_mode,
        output_json=args.output_json,
        verbose=args.verbose,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
