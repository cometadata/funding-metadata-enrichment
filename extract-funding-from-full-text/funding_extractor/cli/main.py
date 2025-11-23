"""CLI entry point for the funding extractor."""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from funding_extractor.config.loader import load_queries
from funding_extractor.config.settings import (
    ApplicationConfig,
    ConfigPaths,
    ExtractionSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    ProviderSettings,
    RuntimeSettings,
)
from funding_extractor.core.extraction import extract_funding_statements
from funding_extractor.core.models import ExtractionResult, DocumentResult, ProcessingParameters, ProcessingResults
from funding_extractor.core.structured_extraction import extract_structured_entities
from funding_extractor.exceptions import ConfigurationError
from funding_extractor.io.checkpointing import CheckpointRepository, get_file_hash
from funding_extractor.io.loaders import (
    DocumentPayload,
    build_markdown_documents,
    determine_input_format,
    stream_parquet_documents,
)
from funding_extractor.processing.markdown_healer import heal_markdown
from funding_extractor.processing.normalization import (
    is_improperly_formatted,
    normalize_funding_statement,
)
from funding_extractor.providers.base import ModelProvider

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract funding information from markdown documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single markdown file
  %(prog)s -i document.md -o results.json

  # Process a directory with Gemini
  %(prog)s -i /path/to/md/files -o results.json --provider gemini --api-key YOUR_KEY

  # Use an Ollama model with normalization
  %(prog)s -i docs/ -o results.json --provider ollama --model llama3.2 --normalize

  # Skip structured extraction (only extract funding statements)
  %(prog)s -i docs/ -o results.json --skip-structured
        """,
    )

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
    parser.add_argument("--prompt-file", help="Text file containing extraction prompt")
    parser.add_argument("--examples-file", help="JSON file containing extraction examples")

    parser.add_argument("--normalize", action="store_true", help="Normalize funding statements (fix whitespace, accents, etc.)")
    parser.add_argument("--heal-markdown", action="store_true", help="Reflow markdown converted from PDFs before parsing")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip semantic extraction (use existing results file)")
    parser.add_argument("--skip-structured", action="store_true", help="Skip structured entity extraction (only extract statements)")

    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama", "openai", "local_openai"],
        default="gemini",
        help="LLM provider for structured extraction (default: gemini)",
    )
    parser.add_argument("--model", help="Model ID to use (defaults to provider default)")
    parser.add_argument("--model-url", help="API endpoint URL (for ollama or local_openai)")
    parser.add_argument("--api-key", help="API key for the model provider")

    parser.add_argument(
        "--colbert-model",
        default="lightonai/GTE-ModernColBERT-v1",
        help="ColBERT model for semantic extraction (default: lightonai/GTE-ModernColBERT-v1)",
    )
    parser.add_argument("--threshold", type=float, default=28.0, help="Minimum score threshold for relevance (default: 28.0)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top paragraphs to analyze per query (default: 5)")

    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process per batch (default: 10)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (auto-detected if not specified)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for LLM requests (default: 60)")

    parser.add_argument("--checkpoint-file", help="Checkpoint file for saving progress")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files (ignore checkpoint)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip model validation checks")
    parser.add_argument("--debug", action="store_true", help="Enable debug output from langextract")
    return parser.parse_args()


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
            heal_markdown=args.heal_markdown,
            skip_extraction=args.skip_extraction,
            skip_structured=args.skip_structured,
        ),
        provider=ProviderSettings(
            provider=ModelProvider(args.provider),
            model_id=args.model,
            model_url=args.model_url,
            api_key=args.api_key,
            timeout=args.timeout,
            skip_model_validation=args.skip_model_validation,
            debug=getattr(args, "debug", False),
        ),
        runtime=RuntimeSettings(
            batch_size=args.batch_size,
            workers=args.workers,
            resume=args.resume,
            force=args.force,
            verbose=args.verbose,
        ),
        config_paths=ConfigPaths(
            queries_file=Path(args.queries) if args.queries else None,
            config_dir=Path(args.config_dir) if args.config_dir else None,
            patterns_file=Path(args.patterns_file) if args.patterns_file else None,
            prompt_file=Path(args.prompt_file) if args.prompt_file else None,
            examples_file=Path(args.examples_file) if args.examples_file else None,
        ),
    )
    return config


def load_existing_results(output_file: Path) -> Optional[ProcessingResults]:
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return ProcessingResults.from_dict(data)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Warning: Could not load existing results: %s", exc)
    return None


def save_results(results: ProcessingResults, output_file: Path) -> None:
    temp_file = str(output_file) + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as fh:
        json.dump(results.to_dict(), fh, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_file)


def process_document_task(document: DocumentPayload, config: ApplicationConfig, queries: Dict[str, str]) -> Optional[DocumentResult]:
    verbose = config.runtime.verbose
    if verbose:
        print(f"Processing: {document.document_id}")

    result = DocumentResult(filename=document.document_id)

    if not config.processing.skip_extraction:
        try:
            content = document.load_text()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Warning: Failed to read content for {document.document_id}: {exc}")
            return None

        if config.processing.heal_markdown:
            content = heal_markdown(content)

        statements = extract_funding_statements(
            content=content,
            queries=queries,
            model_name=config.extraction.colbert_model,
            top_k=config.extraction.top_k,
            threshold=config.extraction.threshold,
            batch_size=config.extraction.semantic_batch_size,
            patterns_file=str(config.config_paths.patterns_file) if config.config_paths.patterns_file else None,
            custom_config_dir=str(config.config_paths.config_dir) if config.config_paths.config_dir else None,
        )

        if not statements:
            if verbose:
                print(f"  No funding statements found in {document.document_id}")
            result.funding_statements = []
            result.extraction_results = []
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

    if not config.processing.skip_structured and result.funding_statements:
        unique_statements = list(set(stmt.statement for stmt in result.funding_statements))

        extraction_results: List[ExtractionResult] = []
        for statement in unique_statements:
            if verbose:
                print("  Extracting entities from statement...")

            try:
                extraction_result = extract_structured_entities(
                    funding_statement=statement,
                    provider=config.provider.provider,
                    model_id=config.provider.model_id,
                    model_url=config.provider.model_url,
                    api_key=config.provider.api_key,
                    skip_model_validation=config.provider.skip_model_validation,
                    timeout=config.provider.timeout,
                    debug=config.provider.debug,
                    prompt_file=str(config.config_paths.prompt_file) if config.config_paths.prompt_file else None,
                    examples_file=str(config.config_paths.examples_file) if config.config_paths.examples_file else None,
                    custom_config_dir=str(config.config_paths.config_dir) if config.config_paths.config_dir else None,
                )
                extraction_results.append(extraction_result)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  Warning: Entity extraction failed: {exc}")

        result.extraction_results = extraction_results

        if verbose:
            total_funders = sum(len(r.funders) for r in extraction_results)
            print(f"  Extracted {total_funders} funders from {len(extraction_results)} statements")
    else:
        result.extraction_results = result.extraction_results or []

    return result


class FundingExtractorApp:
    def __init__(self, config: ApplicationConfig, queries: Dict[str, str]) -> None:
        self.config = config
        self.queries = queries

    def run(self) -> None:
        cfg = self.config
        input_format = determine_input_format(cfg.input.path, cfg.input.input_format)
        cfg.input.input_format = input_format

        checkpoint_repo = CheckpointRepository(cfg.output.checkpoint_path)
        resume_checkpoint = cfg.runtime.resume and not cfg.runtime.force
        checkpoint_data = checkpoint_repo.load(resume=resume_checkpoint)
        checkpoint_data.setdefault("processed_files", {})
        processed_lookup = checkpoint_data.get("processed_files", {})
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
                return

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
                return

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
                heal_markdown=cfg.processing.heal_markdown,
                provider=cfg.provider.provider.value if not cfg.processing.skip_structured else None,
                model=cfg.provider.model_id if not cfg.processing.skip_structured else None,
                threshold=cfg.extraction.threshold,
                top_k=cfg.extraction.top_k,
            ),
            results={},
            summary={},
        )
        results.update_summary()

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
            summary = results.summary or {}
            for key in ("total_files", "files_with_funding", "total_statements", "total_funders"):
                summary.setdefault(key, 0)

            existing = results.results.get(doc.filename)
            if existing:
                summary["total_statements"] -= len(existing.funding_statements)
                existing_funders = sum(len(res.funders) for res in existing.extraction_results)
                summary["total_funders"] -= existing_funders
                if existing.has_funding():
                    summary["files_with_funding"] -= 1
            else:
                summary["total_files"] += 1

            summary["total_statements"] += len(doc.funding_statements)
            new_funders = sum(len(res.funders) for res in doc.extraction_results)
            summary["total_funders"] += new_funders
            if doc.has_funding():
                summary["files_with_funding"] += 1

            results.summary = summary
            results.results[doc.filename] = doc

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
                except Exception as exc:  # pylint: disable=broad-except
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
                        except Exception as exc:  # pylint: disable=broad-except
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

        if processed_count == 0:
            print("No new documents processed (all matched checkpoint or dataset was empty).")
            return

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total documents processed: {results.summary.get('total_files', 0)}")
        print(f"Documents with funding: {results.summary.get('files_with_funding', 0)}")
        print(f"Total statements: {results.summary.get('total_statements', 0)}")
        if not cfg.processing.skip_structured:
            print(f"Total funders: {results.summary.get('total_funders', 0)}")
        print(f"\nResults saved to: {cfg.output.output_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING)

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
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error loading queries: {exc}")
        sys.exit(1)

    app = FundingExtractorApp(config=config, queries=queries)
    app.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
