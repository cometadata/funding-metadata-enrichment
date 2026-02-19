# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "extract-funding-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=extract-funding-from-full-text",
#     "vllm>=0.15.0",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
# ]
# ///
"""Run funding extraction on benchmark dataset and push results to HuggingFace.

Two-stage pipeline:
  Stage 1: ColBERT semantic search for funding statements
  Stage 2: vLLM entity extraction for structured funder/award info

Stages can be run independently or together:
  --stages statements   Run only Stage 1 (ColBERT statement extraction)
  --stages entities     Run only Stage 2 (vLLM entity extraction on gold statements)
  --stages both         Run both stages end-to-end

Output: HuggingFace dataset with "statements" and/or "entities" configs,
each containing train and test splits.

Usage (local):
  python run_benchmark_extraction.py --stages entities --max-samples 5
  python run_benchmark_extraction.py --stages both --max-samples 5

Usage (HF job):
  hf jobs uv run run_benchmark_extraction.py \
      --flavor l4x1 \
      --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
      --secret HF_TOKEN \
      --timeout 2h \
      -- --stages entities --model-id Qwen/Qwen3-8B
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
import time
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

from funding_extractor.config.loader import load_queries
from funding_extractor.entities.extraction import (
    StructuredExtractionService,
    build_provider_settings,
)
from funding_extractor.statements.extraction import SemanticExtractionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run funding extraction benchmark on HF dataset"
    )
    parser.add_argument(
        "--stages",
        choices=["statements", "entities", "both"],
        default="both",
        help="Which stages to run: 'statements' (Stage 1 only), "
        "'entities' (Stage 2 on gold statements), or 'both'",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-8B",
        help="vLLM model HuggingFace ID",
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
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max model context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization fraction",
    )
    parser.add_argument(
        "--enable-pattern-rescue",
        action="store_true",
        help="Enable pattern-based rescue in Stage 1",
    )
    parser.add_argument(
        "--enable-post-filter",
        action="store_true",
        help="Enable post-filtering in Stage 1",
    )
    return parser.parse_args()


def load_gold_statements(
    source_dataset: str,
    max_samples: Optional[int],
) -> dict[str, list[dict[str, Any]]]:
    """Load gold funding statements from the benchmark dataset.

    Wraps each document's funding_statement into the same dict format
    as run_stage1 output so it can be fed directly to run_stage2.
    """
    results_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "test"):
        logger.info("Loading gold statements [%s]", split)
        ds = load_dataset(source_dataset, split=split)
        split_results: list[dict[str, Any]] = []

        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break

            funding_statement = row.get("funding_statement", "")
            statements = []
            if funding_statement and funding_statement.strip():
                statements.append({
                    "statement": funding_statement.strip(),
                    "score": 0.0,
                    "query": "gold",
                    "paragraph_idx": -1,
                })

            split_results.append({
                "doi": row["doi"],
                "statements": statements,
            })

        logger.info(
            "Gold statements [%s]: %d documents, %d with statements",
            split,
            len(split_results),
            sum(1 for r in split_results if r["statements"]),
        )
        results_by_split[split] = split_results

    return results_by_split


def run_stage1(
    source_dataset: str,
    max_samples: Optional[int],
    enable_pattern_rescue: bool,
    enable_post_filter: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Run ColBERT statement extraction on all documents across both splits.

    Returns dict mapping split name to list of per-document result dicts.
    """
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

    # Free ColBERT model from GPU
    del service
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            "GPU memory after Stage 1 cleanup: %.1f MB allocated",
            torch.cuda.memory_allocated() / 1e6,
        )

    return results_by_split


def write_vllm_config(model_id: str, args: argparse.Namespace) -> str:
    """Write a temporary vLLM config YAML and return the path."""
    config_content = (
        f'model: "{model_id}"\n'
        f"\n"
        f"engine:\n"
        f"  tensor_parallel_size: 1\n"
        f"  max_model_len: {args.max_model_len}\n"
        f"  gpu_memory_utilization: {args.gpu_memory_utilization}\n"
        f'  dtype: "auto"\n'
        f"  enable_prefix_caching: true\n"
        f"\n"
        f"sampling:\n"
        f"  temperature: 0.1\n"
        f"  max_tokens: 2048\n"
    )
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    )
    tmp.write(config_content)
    tmp.flush()
    tmp.close()
    return tmp.name


def run_stage2(
    stage1_results_by_split: dict[str, list[dict[str, Any]]],
    model_id: str,
    vllm_config_path: str,
) -> dict[str, list[dict[str, Any]]]:
    """Run vLLM entity extraction on all statements from Stage 1.

    Returns dict mapping split name to list of per-document result dicts.
    """
    provider_settings = build_provider_settings(
        provider="vllm",
        model_id=model_id,
        vllm_config_path=vllm_config_path,
        skip_model_validation=True,
    )
    service = StructuredExtractionService(provider_settings=provider_settings)

    total_statements = sum(
        len(doc["statements"])
        for split_results in stage1_results_by_split.values()
        for doc in split_results
    )
    logger.info("Stage 2: %d total statements to process", total_statements)

    results_by_split: dict[str, list[dict[str, Any]]] = {}
    processed = 0

    for split, split_results in stage1_results_by_split.items():
        entity_results: list[dict[str, Any]] = []

        for doc in split_results:
            doi = doc["doi"]
            all_funders: list[dict[str, Any]] = []

            for stmt_data in doc["statements"]:
                statement_text = stmt_data["statement"]
                try:
                    extraction = service.extract_entities(statement_text)
                    for funder in extraction.funders:
                        all_funders.append({
                            "funder_name": funder.funder_name or "",
                            "awards": [
                                {
                                    "funding_scheme": award.funding_scheme,
                                    "award_ids": award.award_ids,
                                    "award_title": award.award_title,
                                }
                                for award in funder.awards
                            ],
                        })
                except Exception:
                    logger.exception(
                        "Stage 2 failed for doi=%s statement=%.80s",
                        doi,
                        statement_text,
                    )

                processed += 1
                if processed % 50 == 0:
                    logger.info(
                        "Stage 2: processed %d/%d statements",
                        processed,
                        total_statements,
                    )

            entity_results.append({"doi": doi, "funders": all_funders})

        logger.info(
            "Stage 2 [%s]: finished %d documents", split, len(entity_results)
        )
        results_by_split[split] = entity_results

    return results_by_split


def push_datasets(
    output_dataset: str,
    stage1_by_split: Optional[dict[str, list[dict[str, Any]]]] = None,
    stage2_by_split: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> None:
    """Build and push HF datasets. Pushes only the configs with data."""
    if stage1_by_split:
        statements_dd = DatasetDict({
            split: Dataset.from_list(rows)
            for split, rows in stage1_by_split.items()
        })
        logger.info("Pushing statements config to %s", output_dataset)
        statements_dd.push_to_hub(output_dataset, config_name="statements")

    if stage2_by_split:
        entities_dd = DatasetDict({
            split: Dataset.from_list(rows)
            for split, rows in stage2_by_split.items()
        })
        logger.info("Pushing entities config to %s", output_dataset)
        entities_dd.push_to_hub(output_dataset, config_name="entities")


def main() -> None:
    args = parse_args()
    run_statements = args.stages in ("statements", "both")
    run_entities = args.stages in ("entities", "both")

    # Authenticate
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.warning("HF_TOKEN not set; push_to_hub may fail")

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available - GPU required for this pipeline")
        sys.exit(1)
    logger.info("CUDA device: %s", torch.cuda.get_device_name())

    t0 = time.time()
    stage1_by_split: Optional[dict[str, list[dict[str, Any]]]] = None
    stage2_by_split: Optional[dict[str, list[dict[str, Any]]]] = None

    # --- STAGE 1: ColBERT Statement Extraction ---
    if run_statements:
        logger.info("=== STAGE 1: Statement Extraction ===")
        stage1_by_split = run_stage1(
            source_dataset=args.source_dataset,
            max_samples=args.max_samples,
            enable_pattern_rescue=args.enable_pattern_rescue,
            enable_post_filter=args.enable_post_filter,
        )
        for split, results in stage1_by_split.items():
            total_stmts = sum(len(r["statements"]) for r in results)
            logger.info(
                "Stage 1 [%s]: %d docs, %d statements",
                split,
                len(results),
                total_stmts,
            )

    # --- STAGE 2: vLLM Entity Extraction ---
    if run_entities:
        logger.info("=== STAGE 2: Entity Extraction ===")

        # Use Stage 1 output if available, otherwise load gold statements
        if stage1_by_split is not None:
            statements_input = stage1_by_split
        else:
            logger.info("Loading gold statements from benchmark dataset")
            statements_input = load_gold_statements(
                source_dataset=args.source_dataset,
                max_samples=args.max_samples,
            )

        vllm_config_path = write_vllm_config(args.model_id, args)
        try:
            stage2_by_split = run_stage2(
                stage1_results_by_split=statements_input,
                model_id=args.model_id,
                vllm_config_path=vllm_config_path,
            )
        finally:
            os.unlink(vllm_config_path)

        for split, results in stage2_by_split.items():
            total_funders = sum(len(r["funders"]) for r in results)
            logger.info(
                "Stage 2 [%s]: %d docs, %d funders",
                split,
                len(results),
                total_funders,
            )

    # --- PUSH TO HUB ---
    logger.info("=== Pushing datasets to HuggingFace Hub ===")
    push_datasets(
        output_dataset=args.output_dataset,
        stage1_by_split=stage1_by_split,
        stage2_by_split=stage2_by_split,
    )

    elapsed = time.time() - t0
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
