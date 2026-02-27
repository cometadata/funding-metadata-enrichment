# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "extract-funding-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=extract-funding-from-full-text",
#     "vllm>=0.15.0",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
# ]
# ///
"""Run Stage 2: vLLM entity extraction on benchmark funding statements.

Loads gold funding statements from the benchmark dataset and extracts
structured funder/award entities using vLLM, then pushes results to
HuggingFace as the "entities" config of the output dataset.

Usage (HF job):
  hf jobs uv run \\
      --flavor a100-large \\
      --secrets HF_TOKEN \\
      --timeout 2h \\
      run_benchmark_entities.py --model-id Qwen/Qwen3-8B

Usage (local):
  python run_benchmark_entities.py --max-samples 5
"""

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import login

from funding_extractor.entities.extraction import (
    StructuredExtractionService,
    build_provider_settings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM entity extraction on benchmark funding statements"
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
        default=32768,
        help="vLLM max model context length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max generation tokens (thinking + response combined)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization fraction",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online"],
        default="offline",
        help="vLLM inference mode: offline (in-process) or online (HTTP server)",
    )
    parser.add_argument(
        "--lora-path",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--lora-name",
        help="LoRA adapter name (for online mode server)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port for vLLM server in online mode (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel extraction workers (online mode only; forced to 1 for offline)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode (adds reasoning parser for online, strips tags for offline)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Cap thinking tokens (online mode only); must be less than --max-tokens",
    )
    parser.add_argument(
        "--config-name",
        default="entities",
        help="HuggingFace dataset config name for push_to_hub (default: entities)",
    )
    return parser.parse_args()


def load_gold_statements(
    source_dataset: str,
    max_samples: Optional[int],
) -> dict[str, list[dict[str, Any]]]:
    """Load gold funding statements from the benchmark dataset."""
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


def write_vllm_config(model_id: str, args: argparse.Namespace) -> str:
    """Write a temporary vLLM config YAML and return the path."""
    mode = getattr(args, "mode", "offline")
    port = getattr(args, "server_port", 8000)
    lora_path = getattr(args, "lora_path", None)
    lora_name = getattr(args, "lora_name", None)
    enable_thinking = getattr(args, "enable_thinking", False)
    thinking_budget = getattr(args, "thinking_budget", None)

    lora_path_val = f'"{lora_path}"' if lora_path else "null"
    lora_name_val = f'"{lora_name}"' if lora_name else "null"
    enable_thinking_val = "true" if enable_thinking else "false"
    if enable_thinking and thinking_budget is None:
        thinking_budget = args.max_tokens // 2
    thinking_budget_val = str(thinking_budget) if thinking_budget is not None else "null"
    extraction_timeout = 600 if enable_thinking else 120

    if enable_thinking:
        temperature = 0.6
        top_p = 0.95
        presence_penalty = 1.5
    else:
        temperature = 0.7
        top_p = 0.8
        presence_penalty = 0.0

    config_content = (
        f'model: "{model_id}"\n'
        f'mode: "{mode}"\n'
        f"\n"
        f"lora:\n"
        f"  path: {lora_path_val}\n"
        f"  name: {lora_name_val}\n"
        f"\n"
        f"engine:\n"
        f"  tensor_parallel_size: 1\n"
        f"  max_model_len: {args.max_model_len}\n"
        f"  gpu_memory_utilization: {args.gpu_memory_utilization}\n"
        f'  dtype: "auto"\n'
        f"  enable_prefix_caching: true\n"
        f"\n"
        f"server:\n"
        f'  url: "http://localhost:{port}/v1"\n'
        f"  timeout: {extraction_timeout}\n"
        f"\n"
        f"sampling:\n"
        f"  temperature: {temperature}\n"
        f"  top_p: {top_p}\n"
        f"  top_k: 20\n"
        f"  max_tokens: {args.max_tokens}\n"
        f"  enable_thinking: {enable_thinking_val}\n"
        f"  thinking_budget: {thinking_budget_val}\n"
        f"  presence_penalty: {presence_penalty}\n"
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(config_content)
    tmp.flush()
    tmp.close()
    return tmp.name


def start_vllm_server(model_id: str, args: argparse.Namespace) -> subprocess.Popen:
    """Start vLLM server as a background process and wait for readiness."""
    port = getattr(args, "server_port", 8000)
    lora_path = getattr(args, "lora_path", None)
    lora_name = getattr(args, "lora_name", None)
    enable_thinking = getattr(args, "enable_thinking", False)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", str(port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", "auto",
        "--disable-log-requests",
    ]
    if enable_thinking:
        cmd.extend(["--reasoning-parser", "deepseek_r1"])
    if lora_path:
        adapter_name = lora_name or "default"
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"{adapter_name}={lora_path}",
        ])

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    health_url = f"http://localhost:{port}/health"
    max_wait = 300
    poll_interval = 5
    waited = 0
    while waited < max_wait:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            logger.info("vLLM server ready on port %d", port)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {proc.returncode}")
            time.sleep(poll_interval)
            waited += poll_interval

    proc.terminate()
    raise RuntimeError(f"vLLM server not ready after {max_wait}s")


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the vLLM server process."""
    if proc.poll() is None:
        logger.info("Shutting down vLLM server (PID %d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Force-killing vLLM server")
            proc.kill()
            proc.wait()


def _funders_from_extraction(extraction) -> list[dict[str, Any]]:
    """Convert an ExtractionResult's funders to serializable dicts."""
    return [
        {
            "funder_name": funder.funder_name or "",
            "awards": [
                {
                    "funding_scheme": award.funding_scheme,
                    "award_ids": award.award_ids,
                    "award_title": award.award_title,
                }
                for award in funder.awards
            ],
        }
        for funder in extraction.funders
    ]


def run_stage2(
    statements_by_split: dict[str, list[dict[str, Any]]],
    model_id: str,
    vllm_config_path: str,
    workers: int = 1,
    mode: str = "offline",
) -> dict[str, list[dict[str, Any]]]:
    """Run vLLM entity extraction on all statements."""
    if mode == "offline" and workers > 1:
        logger.warning(
            "Forcing workers=1 for offline mode "
            "(in-process vLLM handles its own batching)"
        )
        workers = 1
    workers = max(1, workers)

    provider_settings = build_provider_settings(
        provider="vllm",
        model_id=model_id,
        vllm_config_path=vllm_config_path,
        skip_model_validation=True,
    )
    service = StructuredExtractionService(provider_settings=provider_settings)

    total_statements = sum(
        len(doc["statements"])
        for split_results in statements_by_split.values()
        for doc in split_results
    )
    logger.info(
        "Stage 2: %d total statements to process (workers=%d)",
        total_statements,
        workers,
    )

    results_by_split: dict[str, list[dict[str, Any]]] = {}
    processed = 0

    for split, split_results in statements_by_split.items():
        if workers > 1:
            # Parallel path: flatten all statements, process concurrently
            work_items: list[tuple[int, str]] = []
            for doc_idx, doc in enumerate(split_results):
                for stmt_data in doc["statements"]:
                    work_items.append((doc_idx, stmt_data["statement"]))

            doc_funders: dict[int, list[dict[str, Any]]] = defaultdict(list)
            doc_reasoning: dict[int, list[str]] = defaultdict(list)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                future_to_item = {
                    executor.submit(
                        service.extract_entities_with_reasoning, stmt_text
                    ): (doc_idx, stmt_text)
                    for doc_idx, stmt_text in work_items
                }
                for future in concurrent.futures.as_completed(future_to_item):
                    doc_idx, stmt_text = future_to_item[future]
                    try:
                        extraction, reasoning = future.result()
                        doc_funders[doc_idx].extend(
                            _funders_from_extraction(extraction)
                        )
                        doc_reasoning[doc_idx].extend(reasoning)
                    except Exception:
                        logger.exception(
                            "Stage 2 failed for doi=%s statement=%.80s",
                            split_results[doc_idx]["doi"],
                            stmt_text,
                        )
                    processed += 1
                    if processed % 50 == 0:
                        logger.info(
                            "Stage 2: processed %d/%d statements",
                            processed,
                            total_statements,
                        )

            entity_results = [
                {
                    "doi": doc["doi"],
                    "funders": doc_funders.get(doc_idx, []),
                    "reasoning": doc_reasoning.get(doc_idx, []),
                }
                for doc_idx, doc in enumerate(split_results)
            ]
        else:
            # Sequential path: existing behaviour, zero overhead
            entity_results: list[dict[str, Any]] = []

            for doc in split_results:
                doi = doc["doi"]
                all_funders: list[dict[str, Any]] = []
                all_reasoning: list[str] = []

                for stmt_data in doc["statements"]:
                    statement_text = stmt_data["statement"]
                    try:
                        extraction, reasoning = (
                            service.extract_entities_with_reasoning(
                                statement_text
                            )
                        )
                        all_funders.extend(
                            _funders_from_extraction(extraction)
                        )
                        all_reasoning.extend(reasoning)
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

                entity_results.append({
                    "doi": doi,
                    "funders": all_funders,
                    "reasoning": all_reasoning,
                })

        logger.info(
            "Stage 2 [%s]: finished %d documents", split, len(entity_results)
        )
        results_by_split[split] = entity_results

    return results_by_split


def push_entities(
    results_by_split: dict[str, list[dict[str, Any]]],
    output_dataset: str,
    config_name: str = "entities",
) -> None:
    features = Features({
        "doi": Value("string"),
        "funders": [{
            "funder_name": Value("string"),
            "awards": [{
                "award_ids": Sequence(Value("string")),
                "award_title": Sequence(Value("string")),
                "funding_scheme": Sequence(Value("string")),
            }],
        }],
        "reasoning": Sequence(Value("string")),
    })
    entities_dd = DatasetDict({
        split: Dataset.from_list(rows, features=features)
        for split, rows in results_by_split.items()
    })
    logger.info("Pushing config '%s' to %s", config_name, output_dataset)
    entities_dd.push_to_hub(output_dataset, config_name=config_name)


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.warning("HF_TOKEN not set; push_to_hub may fail")

    if not torch.cuda.is_available():
        logger.error("CUDA not available - GPU required for vLLM")
        sys.exit(1)
    logger.info("CUDA device: %s", torch.cuda.get_device_name())

    t0 = time.time()

    logger.info("=== Loading gold statements ===")
    statements = load_gold_statements(
        source_dataset=args.source_dataset,
        max_samples=args.max_samples,
    )

    mode = args.mode
    logger.info("=== Stage 2: vLLM Entity Extraction (mode=%s) ===", mode)
    vllm_config_path = write_vllm_config(args.model_id, args)
    server_proc = None
    try:
        if mode == "online":
            server_proc = start_vllm_server(args.model_id, args)
        results = run_stage2(
            statements_by_split=statements,
            model_id=args.model_id,
            vllm_config_path=vllm_config_path,
            workers=args.workers,
            mode=mode,
        )
    finally:
        if server_proc is not None:
            stop_vllm_server(server_proc)
        os.unlink(vllm_config_path)

    for split, split_results in results.items():
        total_funders = sum(len(r["funders"]) for r in split_results)
        logger.info(
            "Stage 2 [%s]: %d docs, %d funders",
            split,
            len(split_results),
            total_funders,
        )

    logger.info("=== Pushing to HuggingFace Hub ===")
    push_entities(results, args.output_dataset, config_name=args.config_name)

    elapsed = time.time() - t0
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
