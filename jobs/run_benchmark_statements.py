# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "extract-funding-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=extract-funding-from-full-text",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
# ]
# ///
"""Run Stage 1: ColBERT semantic search for funding statements.

Extracts funding statements from the benchmark dataset's markdown documents
using ColBERT semantic search, then pushes results to HuggingFace as the
"statements" config of the output dataset.

Usage (HF job):
  hf jobs uv run \\
      --flavor a100-large \\
      --secrets HF_TOKEN \\
      --timeout 2h \\
      run_benchmark_statements.py

Usage (local):
  python run_benchmark_statements.py --max-samples 5
"""

import argparse
import gc
import logging
import os
import sys
import time
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

from funding_extractor.config.loader import load_queries
from funding_extractor.statements.extraction import SemanticExtractionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ColBERT statement extraction on benchmark dataset"
    )
    parser.add_argument(
        "--source-dataset",
        default="cometadata/preprint-funding-pdfs-md-conversion",
        help="Source benchmark dataset on HuggingFace",
    )
    parser.add_argument(
        "--output-dataset",
        default="cometadata/funding-extraction-harness-benchmark",
        help="Output dataset on HuggingFace",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)",
    )
    parser.add_argument(
        "--enable-pattern-rescue",
        action="store_true",
        help="Enable pattern-based rescue",
    )
    parser.add_argument(
        "--enable-post-filter",
        action="store_true",
        help="Enable post-filtering",
    )
    return parser.parse_args()


def run_stage1(
    source_dataset: str,
    max_samples: Optional[int],
    enable_pattern_rescue: bool,
    enable_post_filter: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Run ColBERT statement extraction on all documents across both splits."""
    queries = load_queries()
    service = SemanticExtractionService()
    results_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "test"):
        logger.info("Stage 1 [%s]: loading dataset split", split)
        ds = load_dataset(source_dataset, split=split)
        split_results: list[dict[str, Any]] = []

        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break

            doi = row["doi"]
            markdown = row["markdown"]

            try:
                statements = service.extract_funding_statements(
                    queries=queries,
                    content=markdown,
                    model_name="lightonai/GTE-ModernColBERT-v1",
                    top_k=5,
                    threshold=10.0,
                    batch_size=32,
                )

                if enable_pattern_rescue:
                    rescued = service.rescue_by_patterns(
                        content=markdown,
                        existing_statements=statements,
                    )
                    statements.extend(rescued)

                if enable_post_filter and statements:
                    from funding_extractor.statements.post_filter import (
                        apply_post_filter,
                    )
                    statements = apply_post_filter(statements)

                split_results.append({
                    "doi": doi,
                    "statements": [
                        {
                            "statement": s.statement,
                            "score": s.score,
                            "query": s.query,
                            "paragraph_idx": s.paragraph_idx
                            if s.paragraph_idx is not None
                            else -1,
                        }
                        for s in statements
                    ],
                })
            except Exception:
                logger.exception("Stage 1 failed for doi=%s", doi)
                split_results.append({"doi": doi, "statements": []})

            if (i + 1) % 50 == 0:
                logger.info(
                    "Stage 1 [%s]: processed %d documents", split, i + 1
                )

        logger.info(
            "Stage 1 [%s]: finished %d documents", split, len(split_results)
        )
        results_by_split[split] = split_results

    del service
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_by_split


def push_statements(
    results_by_split: dict[str, list[dict[str, Any]]],
    output_dataset: str,
) -> None:
    statements_dd = DatasetDict({
        split: Dataset.from_list(rows)
        for split, rows in results_by_split.items()
    })
    logger.info("Pushing statements config to %s", output_dataset)
    statements_dd.push_to_hub(output_dataset, config_name="statements")


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.warning("HF_TOKEN not set; push_to_hub may fail")

    if not torch.cuda.is_available():
        logger.error("CUDA not available - GPU required for ColBERT")
        sys.exit(1)
    logger.info("CUDA device: %s", torch.cuda.get_device_name())

    t0 = time.time()

    logger.info("=== Stage 1: ColBERT Statement Extraction ===")
    results = run_stage1(
        source_dataset=args.source_dataset,
        max_samples=args.max_samples,
        enable_pattern_rescue=args.enable_pattern_rescue,
        enable_post_filter=args.enable_post_filter,
    )
    for split, split_results in results.items():
        total_stmts = sum(len(r["statements"]) for r in split_results)
        logger.info(
            "Stage 1 [%s]: %d docs, %d statements",
            split,
            len(split_results),
            total_stmts,
        )

    logger.info("=== Pushing to HuggingFace Hub ===")
    push_statements(results, args.output_dataset)

    elapsed = time.time() - t0
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
