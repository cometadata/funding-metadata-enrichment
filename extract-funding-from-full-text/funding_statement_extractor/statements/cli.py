import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from funding_statement_extractor.config.loader import load_queries
from funding_statement_extractor.config.settings import (
    ApplicationConfig,
    ConfigPaths,
    ExtractionSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    RuntimeSettings,
)
from funding_statement_extractor.exceptions import ConfigurationError
from funding_statement_extractor.io.checkpointing import CheckpointRepository, get_file_hash
from funding_statement_extractor.io.loaders import (
    DocumentPayload,
    build_markdown_documents,
    determine_input_format,
    stream_parquet_documents,
)
from funding_statement_extractor.processing.normalization import (
    is_improperly_formatted,
    normalize_funding_statement,
)
from funding_statement_extractor.statements.extraction import extract_funding_statements
from funding_statement_extractor.statements.io import load_existing_results, save_results
from funding_statement_extractor.statements.models import (
    DocumentResult,
    ProcessingParameters,
    ProcessingResults,
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="Input markdown file, directory, or parquet dataset")
    parser.add_argument("-o", "--output", default="funding_results.json", help="Output JSON file (default: funding_results.json)")
    parser.add_argument("--input-format", choices=["markdown", "parquet"], help="Force the input format instead of auto-detecting")
    parser.add_argument(
        "--parquet-text-column",
        help='Column containing markdown text when reading parquet datasets (default: auto-detect, preferring "markdown")',
    )
    parser.add_argument("--parquet-id-column", help="Column containing a unique identifier for each parquet row")
    parser.add_argument("--parquet-batch-size", type=int, default=64, help="Number of parquet rows to load per batch (default: 64)")
    parser.add_argument("-q", "--queries", help="YAML file containing search queries (uses defaults if not provided)")
    parser.add_argument("--config-dir", help="Custom directory containing configuration files")
    parser.add_argument("--patterns-file", help="YAML file containing funding detection patterns")

    parser.add_argument("--normalize", action="store_true", help="Normalize funding statements (fix whitespace, accents, etc.)")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip semantic extraction (use existing results file)")

    parser.add_argument(
        "--colbert-model",
        default="lightonai/GTE-ModernColBERT-v1",
        help="ColBERT model for semantic extraction (default: lightonai/GTE-ModernColBERT-v1)",
    )
    parser.add_argument("--threshold", type=float, default=10.0, help="Minimum score threshold for relevance (default: 10.0)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top paragraphs to analyze per query (default: 5)")
    parser.add_argument("--enable-pattern-rescue", action="store_true", help="Enable pattern-based rescue for statements not in top-k (improves recall)")
    parser.add_argument("--enable-post-filter", action="store_true", help="Enable post-filtering for precision recovery")
    parser.add_argument("--enable-paragraph-prefilter", action="store_true", help="Pre-filter paragraphs by funding keywords before encoding (~6-9x speedup, may affect recall)")

    parser.add_argument("--batch-size", type=int, default=50, help="Number of documents between fsync/checkpoint flushes (default: 50)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (auto-detected if not specified)")

    parser.add_argument("--legacy-engine", action="store_true",
                        help="Use the per-doc ProcessPoolExecutor loop instead of the batch engine.")
    parser.add_argument("--paragraphs-per-batch", type=int, default=4096,
                        help="Pipeline batch size (paragraphs) for the GPU consumer.")
    parser.add_argument("--encode-batch-size", type=int, default=512,
                        help="Sub-batch size passed to model.encode.")
    parser.add_argument("--queue-depth", type=int, default=128,
                        help="Inter-stage queue depth in the batch engine.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-process docs marked failed in the checkpoint.")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                        help="Model dtype for the batch engine (default: auto).")

    parser.add_argument("--checkpoint-file", help="Checkpoint file for saving progress")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files (ignore checkpoint)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")


def build_config(args: argparse.Namespace) -> ApplicationConfig:
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else Path(f"{args.output}.checkpoint")

    config = ApplicationConfig(
        input=InputSettings(
            path=input_path,
            input_format=args.input_format,
            parquet_text_column=args.parquet_text_column,
            parquet_id_column=args.parquet_id_column,
            parquet_batch_size=args.parquet_batch_size,
        ),
        output=OutputSettings(output_path=output_path, checkpoint_path=checkpoint_path),
        extraction=ExtractionSettings(
            colbert_model=args.colbert_model,
            threshold=args.threshold,
            top_k=args.top_k,
            semantic_batch_size=32,
        ),
        processing=ProcessingSettings(
            normalize=args.normalize,
            skip_extraction=args.skip_extraction,
            enable_pattern_rescue=args.enable_pattern_rescue,
            enable_post_filter=args.enable_post_filter,
            enable_paragraph_prefilter=args.enable_paragraph_prefilter,
        ),
        runtime=RuntimeSettings(
            batch_size=args.batch_size,
            workers=args.workers,
            resume=args.resume,
            force=args.force,
            verbose=args.verbose,
            legacy_engine=args.legacy_engine,
            paragraphs_per_batch=args.paragraphs_per_batch,
            encode_batch_size=args.encode_batch_size,
            queue_depth=args.queue_depth,
            retry_failed=args.retry_failed,
            dtype=args.dtype,
        ),
        config_paths=ConfigPaths(
            queries_file=Path(args.queries) if args.queries else None,
            config_dir=Path(args.config_dir) if args.config_dir else None,
            patterns_file=Path(args.patterns_file) if args.patterns_file else None,
        ),
    )
    return config


def process_document_task(document: DocumentPayload, config: ApplicationConfig, queries: Dict[str, str]) -> Optional[DocumentResult]:
    verbose = config.runtime.verbose
    if verbose:
        print(f"Processing: {document.document_id}")

    result = DocumentResult(filename=document.document_id)

    if config.processing.skip_extraction:
        return result

    try:
        content = document.load_text()
    except Exception as exc:
        print(f"  Warning: Failed to read content for {document.document_id}: {exc}")
        return None

    statements = extract_funding_statements(
        content=content,
        queries=queries,
        model_name=config.extraction.colbert_model,
        top_k=config.extraction.top_k,
        threshold=config.extraction.threshold,
        batch_size=config.extraction.semantic_batch_size,
        patterns_file=str(config.config_paths.patterns_file) if config.config_paths.patterns_file else None,
        custom_config_dir=str(config.config_paths.config_dir) if config.config_paths.config_dir else None,
        enable_pattern_rescue=config.processing.enable_pattern_rescue,
        enable_paragraph_prefilter=config.processing.enable_paragraph_prefilter,
    )

    if config.processing.enable_post_filter and statements:
        from funding_statement_extractor.statements.post_filter import apply_post_filter
        statements = apply_post_filter(
            statements,
            high_confidence_threshold=30.0,
            low_confidence_threshold=10.0,
        )

    if not statements:
        if verbose:
            print(f"  No funding statements found in {document.document_id}")
        result.funding_statements = []
        return result

    for stmt in statements:
        if config.processing.normalize:
            normalized = normalize_funding_statement(stmt.statement)
            stmt.original = stmt.statement
            stmt.statement = normalized

        if is_improperly_formatted(stmt.statement):
            stmt.is_problematic = True

    result.funding_statements = statements

    if verbose:
        print(f"  Found {len(statements)} funding statements")
        for stmt in statements:
            preview = stmt.statement.replace('\n', ' ')[:240]
            print(f"    [{stmt.score:.1f}] {preview}")

    return result


class FundingExtractorApp:
    def __init__(self, config: ApplicationConfig, queries: Dict[str, str]) -> None:
        self.config = config
        self.queries = queries

    def _prepare_run(self) -> Optional[Dict[str, object]]:
        """Shared setup: input detection, checkpoint load, document iterator, results init.

        Returns a dict of run state, or None if there is nothing to do.
        """
        cfg = self.config
        input_format = determine_input_format(cfg.input.path, cfg.input.input_format)
        cfg.input.input_format = input_format

        checkpoint_repo = CheckpointRepository(cfg.output.checkpoint_path)
        resume_checkpoint = cfg.runtime.resume and not cfg.runtime.force
        checkpoint_data = checkpoint_repo.load(resume=resume_checkpoint)
        checkpoint_data.setdefault("processed_files", {})
        processed_lookup = checkpoint_data.get("processed_files", {})

        # --retry-failed: remove failed entries from the lookup so they will be
        # re-processed by the iterator below.
        if cfg.runtime.retry_failed and processed_lookup:
            retry_hashes = [
                h for h, meta in processed_lookup.items()
                if isinstance(meta, dict) and meta.get("status") == "failed"
            ]
            for h in retry_hashes:
                processed_lookup.pop(h, None)
            if retry_hashes:
                print(f"Retrying {len(retry_hashes)} previously failed documents")

        if resume_checkpoint:
            processed_total = len(processed_lookup)
            print(f"Resuming from checkpoint: {processed_total} documents already processed")

        documents_iter: Iterator[Tuple[DocumentPayload, str]]
        discovered_documents: Optional[int] = None
        total_documents: Optional[int] = None

        if input_format == "parquet":
            text_fallbacks = [] if cfg.input.parquet_text_column else ["markdown", "content", "text", "body"]
            id_fallbacks = [] if cfg.input.parquet_id_column else ["source_id", "doc_id", "document_id", "id", "file_name", "filename"]
            parquet_iterable, discovered_documents = stream_parquet_documents(
                input_path=cfg.input.path,
                text_column=cfg.input.parquet_text_column,
                id_column=cfg.input.parquet_id_column,
                batch_size=cfg.input.parquet_batch_size,
                fallback_text_columns=text_fallbacks,
                fallback_id_columns=id_fallbacks,
            )
            if discovered_documents == 0:
                print("No parquet rows found to process")
                return None

            if discovered_documents is not None:
                print(f"Discovered {discovered_documents} parquet rows to process")
            else:
                print("Scanning parquet dataset for funding statements...")

            def parquet_iterator() -> Iterator[Tuple[DocumentPayload, str]]:
                for doc in parquet_iterable:
                    doc_hash = get_file_hash(doc.checkpoint_key)
                    if cfg.runtime.force or doc_hash not in processed_lookup:
                        yield doc, doc_hash

            documents_iter = parquet_iterator()
            if discovered_documents is not None and not cfg.runtime.force:
                total_documents = max(discovered_documents - len(processed_lookup), 0)
            else:
                total_documents = discovered_documents
        else:
            markdown_docs = build_markdown_documents(cfg.input.path)
            if not markdown_docs:
                print("No markdown files found")
                sys.exit(0)

            print(f"Found {len(markdown_docs)} markdown files to process")

            docs_with_hashes: List[Tuple[DocumentPayload, str]] = []
            for doc in markdown_docs:
                doc_hash = get_file_hash(doc.checkpoint_key)
                if cfg.runtime.force or doc_hash not in processed_lookup:
                    docs_with_hashes.append((doc, doc_hash))

            if not docs_with_hashes:
                print("All documents already processed")
                return None

            total_documents = len(docs_with_hashes)

            def markdown_iterator() -> Iterator[Tuple[DocumentPayload, str]]:
                for item in docs_with_hashes:
                    yield item

            documents_iter = markdown_iterator()
            print(f"Processing {total_documents} files...")

        results = load_existing_results(cfg.output.output_path) or ProcessingResults(
            timestamp=datetime.now().isoformat(),
            parameters=ProcessingParameters(
                input_path=str(cfg.input.path),
                input_format=input_format,
                normalize=cfg.processing.normalize,
                threshold=cfg.extraction.threshold,
                top_k=cfg.extraction.top_k,
            ),
            results={},
            summary={},
        )
        results.update_summary()

        return {
            "checkpoint_repo": checkpoint_repo,
            "documents_iter": documents_iter,
            "total_documents": total_documents,
            "results": results,
            "input_format": input_format,
        }

    def _apply_document_result(self, results: ProcessingResults, doc: DocumentResult) -> None:
        summary = results.summary or {}
        for key in ("total_files", "files_with_funding", "total_statements"):
            summary.setdefault(key, 0)

        existing = results.results.get(doc.filename)
        if existing:
            summary["total_statements"] -= len(existing.funding_statements)
            if existing.has_funding():
                summary["files_with_funding"] -= 1
        else:
            summary["total_files"] += 1

        summary["total_statements"] += len(doc.funding_statements)
        if doc.has_funding():
            summary["files_with_funding"] += 1

        results.summary = summary
        results.results[doc.filename] = doc

    def _print_summary(self, results: ProcessingResults, processed_count: int) -> None:
        cfg = self.config
        if processed_count == 0:
            print("No new documents processed (all matched checkpoint or dataset was empty).")
            return

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total documents processed: {results.summary.get('total_files', 0)}")
        print(f"Documents with funding: {results.summary.get('files_with_funding', 0)}")
        print(f"Total statements: {results.summary.get('total_statements', 0)}")
        print(f"\nResults saved to: {cfg.output.output_path}")

    def run(self) -> None:
        if self.config.runtime.legacy_engine:
            self._run_legacy()
        else:
            self._run_batch()

    def _run_legacy(self) -> None:
        cfg = self.config
        state = self._prepare_run()
        if state is None:
            return

        checkpoint_repo: CheckpointRepository = state["checkpoint_repo"]
        documents_iter: Iterator[Tuple[DocumentPayload, str]] = state["documents_iter"]
        total_documents: Optional[int] = state["total_documents"]
        results: ProcessingResults = state["results"]

        batch_results: List[DocumentResult] = []
        processed_count = 0

        if cfg.runtime.workers is not None:
            worker_count = max(1, cfg.runtime.workers)
        else:
            cpu_count = os.cpu_count() or 1
            worker_count = max(1, cpu_count - 1)

        use_parallel = worker_count > 1
        if use_parallel:
            print(f"Using {worker_count} worker processes")

        max_pending_futures = worker_count * 2 if use_parallel else 0
        pending_futures = set()
        future_to_metadata: Dict[concurrent.futures.Future, Dict[str, str]] = {}
        executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

        def build_document_metadata(document: DocumentPayload, doc_hash: str) -> Dict[str, str]:
            return {
                "doc_hash": doc_hash,
                "path": document.file_path or document.document_id,
                "document_id": document.document_id,
            }

        def apply_document_result(doc: DocumentResult) -> None:
            self._apply_document_result(results, doc)

        def flush_batch() -> None:
            nonlocal batch_results
            if not batch_results:
                return

            for doc in batch_results:
                apply_document_result(doc)

            save_results(results, cfg.output.output_path)
            checkpoint_repo.save()

            if total_documents is not None and total_documents > 0:
                print(f"Processed {processed_count}/{total_documents} documents, saved checkpoint")
            else:
                print(f"Processed {processed_count} documents, saved checkpoint")

            batch_results = []

        def handle_completed(metadata: Dict[str, str], doc_result: Optional[DocumentResult]) -> None:
            nonlocal batch_results, processed_count
            if doc_result:
                batch_results.append(doc_result)

            checkpoint_repo.record(
                metadata["doc_hash"],
                {
                    "path": metadata["path"],
                    "document_id": metadata["document_id"],
                    "processed_at": datetime.now().isoformat(),
                    "found_funding": bool(doc_result and doc_result.funding_statements),
                },
            )
            processed_count += 1

            if len(batch_results) >= cfg.runtime.batch_size:
                flush_batch()

        def drain_futures(wait_all: bool = False) -> None:
            if not pending_futures:
                return

            return_when = concurrent.futures.ALL_COMPLETED if wait_all else concurrent.futures.FIRST_COMPLETED
            done, _ = concurrent.futures.wait(pending_futures, return_when=return_when)

            for future in done:
                pending_futures.remove(future)
                metadata = future_to_metadata.pop(future, None)
                if metadata is None:
                    continue
                try:
                    doc_result = future.result()
                except Exception as exc:
                    print(f"  Error processing {metadata['document_id']}: {exc}")
                    doc_result = None
                handle_completed(metadata, doc_result)

        try:
            if use_parallel:
                ctx = multiprocessing.get_context("spawn")
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx)

            for document, doc_hash in documents_iter:
                metadata = build_document_metadata(document, doc_hash)

                if use_parallel and executor:
                    while len(pending_futures) >= max_pending_futures:
                        drain_futures()

                    future = executor.submit(process_document_task, document, cfg, self.queries)
                    pending_futures.add(future)
                    future_to_metadata[future] = metadata
                else:
                    doc_result = process_document_task(document, cfg, self.queries)
                    handle_completed(metadata, doc_result)

            if use_parallel:
                drain_futures(wait_all=True)

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
            if use_parallel:
                for future in list(pending_futures):
                    if future.done():
                        metadata = future_to_metadata.pop(future, None)
                        if metadata is None:
                            pending_futures.remove(future)
                            continue
                        try:
                            doc_result = future.result()
                        except Exception as exc:
                            print(f"  Error processing {metadata['document_id']}: {exc}")
                            doc_result = None
                        handle_completed(metadata, doc_result)
                        pending_futures.remove(future)

                if executor:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None

                pending_futures.clear()
                future_to_metadata.clear()

            flush_batch()
            print("Progress saved. Use --resume to continue.")
            sys.exit(1)

        finally:
            if executor:
                executor.shutdown(wait=True)

        flush_batch()
        self._print_summary(results, processed_count)

    def _run_batch(self) -> None:
        cfg = self.config
        state = self._prepare_run()
        if state is None:
            return

        checkpoint_repo: CheckpointRepository = state["checkpoint_repo"]
        documents_iter: Iterator[Tuple[DocumentPayload, str]] = state["documents_iter"]
        total_documents: Optional[int] = state["total_documents"]
        results: ProcessingResults = state["results"]

        from funding_statement_extractor.statements.batch_extraction import (
            DocPayload,
            extract_funding_statements_batch,
        )

        def docs_iter() -> Iterator["DocPayload"]:
            for document, doc_hash in documents_iter:
                try:
                    text = document.load_text()
                except Exception as exc:
                    print(f"  Warning: Failed to read content for {document.document_id}: {exc}")
                    checkpoint_repo.record(
                        doc_hash,
                        {
                            "path": document.file_path or document.document_id,
                            "document_id": document.document_id,
                            "processed_at": datetime.now().isoformat(),
                            "found_funding": False,
                            "status": "failed",
                            "error": f"load_text: {type(exc).__name__}: {exc}",
                        },
                    )
                    continue

                yield DocPayload(
                    doc_id=document.document_id,
                    text=text,
                    metadata={
                        "doc_hash": doc_hash,
                        "path": document.file_path or document.document_id,
                    },
                )

        fsync_interval = max(1, cfg.runtime.batch_size)
        processed_count = 0
        completions = 0

        try:
            for result in extract_funding_statements_batch(
                documents=docs_iter(),
                queries=self.queries,
                model_name=cfg.extraction.colbert_model,
                top_k=cfg.extraction.top_k,
                threshold=cfg.extraction.threshold,
                enable_paragraph_prefilter=cfg.processing.enable_paragraph_prefilter,
                patterns_file=str(cfg.config_paths.patterns_file) if cfg.config_paths.patterns_file else None,
                custom_config_dir=str(cfg.config_paths.config_dir) if cfg.config_paths.config_dir else None,
                paragraphs_per_batch=cfg.runtime.paragraphs_per_batch,
                encode_batch_size=cfg.runtime.encode_batch_size,
                workers=cfg.runtime.workers,
                queue_depth=cfg.runtime.queue_depth,
                dtype=cfg.runtime.dtype,
            ):
                meta = result.metadata or {}
                doc_hash = meta.get("doc_hash") or get_file_hash(result.doc_id)
                doc_path = meta.get("path", result.doc_id)

                if result.error:
                    logger.warning("doc=%s error=%s", result.doc_id, result.error)
                    checkpoint_repo.record(
                        doc_hash,
                        {
                            "path": doc_path,
                            "document_id": result.doc_id,
                            "processed_at": datetime.now().isoformat(),
                            "found_funding": False,
                            "status": "failed",
                            "error": result.error,
                        },
                    )
                else:
                    doc_result = DocumentResult(
                        filename=result.doc_id,
                        funding_statements=result.statements,
                    )
                    if cfg.processing.normalize:
                        for stmt in doc_result.funding_statements:
                            stmt.original = stmt.statement
                            stmt.statement = normalize_funding_statement(stmt.statement)
                            if is_improperly_formatted(stmt.statement):
                                stmt.is_problematic = True
                    else:
                        for stmt in doc_result.funding_statements:
                            if is_improperly_formatted(stmt.statement):
                                stmt.is_problematic = True

                    self._apply_document_result(results, doc_result)
                    checkpoint_repo.record(
                        doc_hash,
                        {
                            "path": doc_path,
                            "document_id": result.doc_id,
                            "processed_at": datetime.now().isoformat(),
                            "found_funding": bool(doc_result.funding_statements),
                            "status": "ok",
                        },
                    )
                    processed_count += 1

                completions += 1
                if completions % fsync_interval == 0:
                    save_results(results, cfg.output.output_path)
                    checkpoint_repo.save()
                    if total_documents is not None and total_documents > 0:
                        print(f"Processed {completions}/{total_documents} documents, saved checkpoint")
                    else:
                        print(f"Processed {completions} documents, saved checkpoint")

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
            save_results(results, cfg.output.output_path)
            checkpoint_repo.save()
            print("Progress saved. Use --resume to continue.")
            sys.exit(1)

        save_results(results, cfg.output.output_path)
        checkpoint_repo.save()
        self._print_summary(results, completions)


def run(args: argparse.Namespace) -> None:
    config = build_config(args)
    try:
        config.validate()
    except ConfigurationError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    try:
        queries = load_queries(
            queries_file=str(config.config_paths.queries_file) if config.config_paths.queries_file else None,
            custom_config_dir=str(config.config_paths.config_dir) if config.config_paths.config_dir else None,
        )
    except Exception as exc:
        print(f"Error loading queries: {exc}")
        sys.exit(1)

    app = FundingExtractorApp(config=config, queries=queries)
    app.run()
