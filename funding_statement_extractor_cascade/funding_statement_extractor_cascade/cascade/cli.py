import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
from funding_statement_extractor_cascade.cascade.loader import load_models
from funding_statement_extractor_cascade.cascade.types import CascadeConfig, DocPayload
from funding_statement_extractor_cascade.config.settings import (
    ApplicationConfig,
    CascadeRuntimeSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    RuntimeSettings,
)
from funding_statement_extractor_cascade.exceptions import ConfigurationError
from funding_statement_extractor_cascade.io.checkpointing import (
    CheckpointRepository,
    get_file_hash,
)
from funding_statement_extractor_cascade.io.loaders import (
    DocumentPayload,
    build_markdown_documents,
    determine_input_format,
    stream_parquet_documents,
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True,
                        help="Input markdown file, directory, or parquet dataset")
    parser.add_argument("-o", "--output", default="funding_results.json",
                        help="Output JSON file (default: funding_results.json)")
    parser.add_argument("--input-format", choices=["markdown", "parquet"])
    parser.add_argument("--parquet-text-column",
                        help='Column with markdown text (default: auto-detect, preferring "vlm_markdown")')
    parser.add_argument("--parquet-id-column",
                        help='Column with a unique id (default: auto-detect, preferring "arxiv_id")')
    parser.add_argument("--parquet-batch-size", type=int, default=64)

    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip Stage 3 (Qwen cleaner); return raw stage-2 span verbatim.")
    parser.add_argument("--chunk-keep-threshold", type=float, default=0.4)
    parser.add_argument("--chunk-top-k", type=int, default=2)
    parser.add_argument("--span-no-answer-threshold", type=float, default=0.5)
    parser.add_argument("--span-max-length", type=int, default=300)
    parser.add_argument("--cleaner-context-chars", type=int, default=400)
    parser.add_argument("--cleaner-max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--batch-size", type=int, default=50,
                        help="Documents between checkpoint flushes (default: 50)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--checkpoint-file")


def build_config(args: argparse.Namespace) -> ApplicationConfig:
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = (Path(args.checkpoint_file) if args.checkpoint_file
                       else Path(f"{args.output}.checkpoint"))
    return ApplicationConfig(
        input=InputSettings(
            path=input_path,
            input_format=args.input_format,
            parquet_text_column=args.parquet_text_column,
            parquet_id_column=args.parquet_id_column,
            parquet_batch_size=args.parquet_batch_size,
        ),
        output=OutputSettings(output_path=output_path, checkpoint_path=checkpoint_path),
        cascade=CascadeRuntimeSettings(
            chunk_keep_threshold=args.chunk_keep_threshold,
            chunk_top_k=args.chunk_top_k,
            span_no_answer_threshold=args.span_no_answer_threshold,
            span_max_length=args.span_max_length,
            cleaner_context_chars=args.cleaner_context_chars,
            cleaner_max_new_tokens=args.cleaner_max_new_tokens,
            skip_cleanup=args.skip_cleanup,
            device=args.device,
            dtype=args.dtype,
        ),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(batch_size=args.batch_size,
                                resume=args.resume,
                                force=args.force,
                                retry_failed=args.retry_failed,
                                verbose=args.verbose),
    )


def _docs_iter(cfg: ApplicationConfig, checkpoint_repo: CheckpointRepository
                ) -> Iterator[Tuple[DocumentPayload, str]]:
    fmt = determine_input_format(cfg.input.path, cfg.input.input_format)
    cfg.input.input_format = fmt

    processed = checkpoint_repo.data.get("processed_files", {})
    if cfg.runtime.force:
        processed = {}

    if fmt == "parquet":
        text_fallbacks = ([] if cfg.input.parquet_text_column
                          else ["vlm_markdown", "markdown", "content", "text", "body"])
        id_fallbacks = ([] if cfg.input.parquet_id_column
                        else ["arxiv_id", "source_id", "doc_id", "document_id", "id", "file_name"])
        stream, _ = stream_parquet_documents(
            input_path=cfg.input.path,
            text_column=cfg.input.parquet_text_column,
            id_column=cfg.input.parquet_id_column,
            batch_size=cfg.input.parquet_batch_size,
            fallback_text_columns=text_fallbacks,
            fallback_id_columns=id_fallbacks,
        )
        for doc in stream:
            h = get_file_hash(doc.checkpoint_key)
            if cfg.runtime.force or h not in processed:
                yield doc, h
        return

    for doc in build_markdown_documents(cfg.input.path):
        h = get_file_hash(doc.checkpoint_key)
        if cfg.runtime.force or h not in processed:
            yield doc, h


def _build_payloads(docs_iter, checkpoint_repo, verbose):
    for document, doc_hash in docs_iter:
        try:
            text = document.load_text()
        except Exception as exc:
            if verbose:
                print(f"  warn: failed to read {document.document_id}: {exc}")
            checkpoint_repo.record(doc_hash, {
                "path": document.file_path or document.document_id,
                "document_id": document.document_id,
                "processed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": f"load_text: {type(exc).__name__}: {exc}",
            })
            continue
        yield DocPayload(
            doc_id=document.document_id,
            text=text,
            metadata={"doc_hash": doc_hash,
                      "path": document.file_path or document.document_id},
        )


def run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    try:
        cfg.validate()
    except ConfigurationError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    cascade_cfg = CascadeConfig(
        chunk_keep_threshold=cfg.cascade.chunk_keep_threshold,
        chunk_top_k=cfg.cascade.chunk_top_k,
        span_no_answer_threshold=cfg.cascade.span_no_answer_threshold,
        span_max_length=cfg.cascade.span_max_length,
        cleaner_context_chars=cfg.cascade.cleaner_context_chars,
        cleaner_max_new_tokens=cfg.cascade.cleaner_max_new_tokens,
        skip_cleanup=cfg.cascade.skip_cleanup,
        device=cfg.cascade.device,
        dtype=cfg.cascade.dtype,
    )

    models = load_models(cascade_cfg)

    checkpoint_repo = CheckpointRepository(cfg.output.checkpoint_path)
    if cfg.runtime.resume and not cfg.runtime.force:
        checkpoint_repo.load(resume=True)

    results: dict = {}
    if cfg.runtime.resume and not cfg.runtime.force and cfg.output.output_path.exists():
        try:
            results = json.loads(cfg.output.output_path.read_text()).get("results", {})
        except (json.JSONDecodeError, OSError):
            results = {}
    payloads = _build_payloads(_docs_iter(cfg, checkpoint_repo), checkpoint_repo,
                                cfg.runtime.verbose)
    processed = 0
    for batch in extract_funding_batch(documents=payloads, models=models, cfg=cascade_cfg):
        results[batch.doc_id] = {
            "funding_statement": batch.result.funding_statement,
            "rough_span": batch.result.rough_span,
            "chunk_prob": batch.result.chunk_prob,
            "no_answer_prob": batch.result.no_answer_prob,
            "latency_ms": batch.result.latency_ms,
            "error": batch.result.error,
        }
        h = (batch.metadata or {}).get("doc_hash") or get_file_hash(batch.doc_id)
        checkpoint_repo.record(h, {
            "path": (batch.metadata or {}).get("path", batch.doc_id),
            "document_id": batch.doc_id,
            "processed_at": datetime.now().isoformat(),
            "found_funding": bool(batch.result.funding_statement),
            "status": "failed" if batch.result.error else "ok",
            "error": batch.result.error,
        })
        processed += 1
        if processed % max(1, cfg.runtime.batch_size) == 0:
            _flush(cfg, results, checkpoint_repo)

    _flush(cfg, results, checkpoint_repo)


def _flush(cfg: ApplicationConfig, results: dict, repo: CheckpointRepository) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "input_path": str(cfg.input.path),
            "input_format": cfg.input.input_format,
            "skip_cleanup": cfg.cascade.skip_cleanup,
            "dtype": cfg.cascade.dtype,
        },
        "results": results,
        "summary": {
            "total_files": len(results),
            "files_with_funding": sum(1 for r in results.values() if r["funding_statement"]),
        },
    }
    cfg.output.output_path.write_text(json.dumps(payload, indent=2))
    repo.save()
