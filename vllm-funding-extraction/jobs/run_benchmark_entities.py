# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm-funding-extraction @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=vllm-funding-extraction",
#     "vllm>=0.16.1.dev0",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
#     "pyarrow>=23.0.0",
# ]
#
# [tool.uv]
# index-strategy = "unsafe-best-match"
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/9433acb8dfdafa560dbee4d67bc286ab3543db39"
# ///
"""Run vLLM entity extraction on benchmark funding statements.

Loads gold funding statements from the benchmark dataset and extracts
structured funder/award entities using vLLM, then pushes results to
HuggingFace as a named config of the output dataset.

Usage (HF job):
  hf jobs uv run \\
      --flavor a100-large \\
      --secrets HF_TOKEN \\
      --timeout 2h \\
      run_benchmark_entities.py --vllm-config qwen3-8b.yaml

Usage (local):
  python run_benchmark_entities.py --vllm-config llama-3.1-8b.yaml --max-samples 5
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import yaml

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import login

from funding_extraction import ExtractionService, load_vllm_config

logger = logging.getLogger(__name__)


def resolve_lora_path(lora_config: dict) -> Optional[str]:
    """Resolve a LoRA path, downloading from HF Hub if subfolder is specified."""
    path = lora_config.get("path")
    subfolder = lora_config.get("subfolder")
    if not path or not subfolder:
        return path
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=path,
        allow_patterns=f"{subfolder}/**",
    )
    resolved = os.path.join(local_dir, subfolder)
    logger.info("Resolved LoRA subfolder: %s -> %s", path, resolved)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM entity extraction on benchmark funding statements"
    )
    parser.add_argument(
        "--vllm-config",
        default="llama-3.1-8b.yaml",
        help="vLLM config file name or path (default: llama-3.1-8b.yaml)",
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
        "--split",
        choices=["train", "test", "both"],
        default="test",
        help="Which split(s) to run: train, test, or both (default: test)",
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
        default=None,
        help="Override config max model context length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override config max generation tokens",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override config GPU memory utilization fraction",
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
        default=None,
        help="Override config parallel extraction workers",
    )
    parser.add_argument(
        "--extraction-passes",
        type=int,
        default=None,
        help="Override config extraction passes per statement",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Override config thinking token budget",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Override config HuggingFace dataset config name for push_to_hub",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        default=None,
        help="Enable guided JSON decoding",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def load_gold_statements(
    source_dataset: str,
    max_samples: Optional[int],
    splits: tuple[str, ...] = ("test",),
) -> dict[str, list[dict[str, Any]]]:
    """Load gold funding statements from the benchmark dataset."""
    results_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in splits:
        logger.info("Loading gold statements [%s]", split)
        ds = load_dataset(source_dataset, split=split)
        split_results: list[dict[str, Any]] = []

        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break

            funding_statement = row.get("funding_statement", "")
            statements = []
            if funding_statement and funding_statement.strip():
                statements.append(funding_statement.strip())

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


def _resolve_config_path(config_arg: str) -> str:
    """Resolve a vLLM config path."""
    if os.path.isfile(config_arg):
        return config_arg

    # Look in bundled configs relative to repo root
    bundled = (
        Path(__file__).resolve().parent.parent
        / "funding_extraction"
        / "configs"
        / "vllm"
        / config_arg
    )
    if bundled.is_file():
        return str(bundled)

    # Try relative to the installed package
    try:
        import funding_extraction
        pkg_dir = Path(funding_extraction.__file__).resolve().parent
        pkg_config = pkg_dir / "configs" / "vllm" / config_arg
        if pkg_config.is_file():
            return str(pkg_config)
    except (ImportError, AttributeError):
        pass

    raise FileNotFoundError(
        f"vLLM config not found: {config_arg}. "
        f"Pass an absolute path or a filename from configs/vllm/."
    )


def load_and_patch_config(args: argparse.Namespace) -> tuple[str, str, dict]:
    """Load a named vLLM config, apply CLI overrides, write to temp file.

    Returns (temp_config_path, model_id, original_config_data).
    """
    config_path = _resolve_config_path(args.vllm_config)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    original_data = yaml.safe_load(yaml.dump(data))

    model_id = data.get("model")
    if not model_id:
        raise ValueError(
            f"Config {config_path} has no 'model' set. "
            f"Use a model-specific config or add 'model:' to the YAML."
        )

    if args.max_model_len is not None:
        data.setdefault("engine", {})["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        data.setdefault("engine", {})["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_tokens is not None:
        data.setdefault("sampling", {})["max_tokens"] = args.max_tokens
    if args.thinking_budget is not None:
        data.setdefault("sampling", {})["thinking_budget"] = args.thinking_budget
    if args.extraction_passes is not None:
        data.setdefault("sampling", {})["extraction_passes"] = args.extraction_passes
    if args.guided_decoding is not None:
        data.setdefault("sampling", {})["guided_decoding"] = args.guided_decoding

    mode = data.get("mode", "offline")
    if mode == "online":
        port = getattr(args, "server_port", 8000)
        data.setdefault("server", {})["url"] = f"http://localhost:{port}/v1"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp, default_flow_style=False)
    tmp.flush()
    tmp.close()

    return tmp.name, model_id, original_data


def start_vllm_server(model_id: str, config_data: dict, args: argparse.Namespace) -> subprocess.Popen:
    """Start vLLM server as a background process and wait for readiness."""
    port = getattr(args, "server_port", 8000)
    lora_config = config_data.get("lora", {})
    lora_path = resolve_lora_path(lora_config)
    lora_name = lora_config.get("name")
    sampling = config_data.get("sampling", {})
    enable_thinking = sampling.get("enable_thinking", False)
    engine = config_data.get("engine", {})

    max_model_len = engine.get("max_model_len", 16384)
    if args.max_model_len is not None:
        max_model_len = args.max_model_len
    gpu_mem = engine.get("gpu_memory_utilization", 0.9)
    if args.gpu_memory_utilization is not None:
        gpu_mem = args.gpu_memory_utilization

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--dtype", engine.get("dtype", "auto"),
        "--no-enable-log-requests",
    ]
    server_config = config_data.get("server", {})
    reasoning_parser = server_config.get("reasoning_parser")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if not enable_thinking:
        default_kwargs = json.dumps({"enable_thinking": False})
        cmd.extend(["--default-chat-template-kwargs", default_kwargs])
    if engine.get("enable_prefix_caching", False):
        cmd.append("--enable-prefix-caching")
    if engine.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    if lora_path:
        adapter_name = lora_name or "default"
        max_lora_rank = lora_config.get("max_rank", 64)
        max_loras = lora_config.get("max_loras", 1)
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"{adapter_name}={lora_path}",
            "--max-lora-rank", str(max_lora_rank),
            "--max-loras", str(max_loras),
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


def _funders_to_dicts(extraction) -> list[dict[str, Any]]:
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


def run_extraction(
    statements_by_split: dict[str, list[dict[str, Any]]],
    vllm_config_path: str,
    workers: int = 64,
) -> dict[str, list[dict[str, Any]]]:
    """Run vLLM entity extraction on all statements."""
    config = load_vllm_config(vllm_config_path)
    service = ExtractionService(config)

    total_statements = sum(
        len(doc["statements"])
        for split_results in statements_by_split.values()
        for doc in split_results
    )
    logger.info(
        "Extraction: %d total statements to process (workers=%d)",
        total_statements,
        workers,
    )

    results_by_split: dict[str, list[dict[str, Any]]] = {}

    for split, split_results in statements_by_split.items():
        # Flatten to (doc_id, statement) pairs
        work_items: list[tuple[str, str]] = []
        doc_id_to_idx: dict[str, int] = {}
        for doc_idx, doc in enumerate(split_results):
            for stmt_text in doc["statements"]:
                uid = f"{doc_idx}:{len(work_items)}"
                work_items.append((uid, stmt_text))
                doc_id_to_idx[uid] = doc_idx

        logger.info("Extraction [%s]: %d statements", split, len(work_items))

        extraction_results = service.extract_concurrent(
            work_items, workers=workers, warmup_count=min(4, len(work_items))
        )

        # Group results by document
        from collections import defaultdict
        doc_funders: dict[int, list[dict[str, Any]]] = defaultdict(list)
        doc_reasoning: dict[int, list[str]] = defaultdict(list)

        for uid, (result, reasoning) in extraction_results.items():
            doc_idx = doc_id_to_idx[uid]
            doc_funders[doc_idx].extend(_funders_to_dicts(result))
            doc_reasoning[doc_idx].extend(reasoning)

        entity_results = [
            {
                "doi": doc["doi"],
                "funders": doc_funders.get(doc_idx, []),
                "reasoning": doc_reasoning.get(doc_idx, []),
            }
            for doc_idx, doc in enumerate(split_results)
        ]

        logger.info("Extraction [%s]: finished %d documents", split, len(entity_results))
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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

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
    splits = ("train", "test") if args.split == "both" else (args.split,)
    statements = load_gold_statements(
        source_dataset=args.source_dataset,
        max_samples=args.max_samples,
        splits=splits,
    )

    vllm_config_path, model_id, config_data = load_and_patch_config(args)

    mode = config_data.get("mode", "offline")
    benchmark = config_data.get("benchmark", {})
    workers = args.workers if args.workers is not None else benchmark.get("workers", 64)
    config_name = args.config_name if args.config_name is not None else benchmark.get("config_name")

    if not config_name:
        logger.error("No config_name: set benchmark.config_name in YAML or pass --config-name")
        sys.exit(1)

    logger.info("=== Entity Extraction (mode=%s, model=%s) ===", mode, model_id)
    server_proc = None
    try:
        if mode == "online":
            server_proc = start_vllm_server(model_id, config_data, args)
        results = run_extraction(
            statements_by_split=statements,
            vllm_config_path=vllm_config_path,
            workers=workers,
        )
    finally:
        if server_proc is not None:
            stop_vllm_server(server_proc)
        os.unlink(vllm_config_path)

    for split, split_results in results.items():
        total_funders = sum(len(r["funders"]) for r in split_results)
        logger.info(
            "Extraction [%s]: %d docs, %d funders",
            split,
            len(split_results),
            total_funders,
        )

    logger.info("=== Pushing to HuggingFace Hub ===")
    push_entities(results, args.output_dataset, config_name=config_name)

    elapsed = time.time() - t0
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
