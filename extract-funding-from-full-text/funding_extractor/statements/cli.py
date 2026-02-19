import argparse

import os
import sys
import concurrent.futures
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from funding_extractor.config.loader import load_queries
from funding_extractor.io.checkpointing import CheckpointRepository, get_file_hash
from funding_extractor.io.loaders import (
    DocumentPayload,
    build_markdown_documents,
    determine_input_format,
    stream_jsonl_documents,
    stream_parquet_documents,
)
from funding_extractor.processing.markdown_healer import heal_markdown
from funding_extractor.processing.normalization import (
    is_improperly_formatted,
    normalize_funding_statement,
)
from funding_extractor.statements.extraction import extract_funding_statements
from funding_extractor.statements.io import write_statements_jsonl
from funding_extractor.statements.models import FundingStatement
from funding_extractor.statements.post_filter import apply_post_filter




def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="Input markdown file, directory, parquet, or JSONL")
    parser.add_argument("-o", "--output", default="funding_statements.jsonl", help="Output JSONL file (default: funding_statements.jsonl)")
    parser.add_argument("--input-format", choices=["markdown", "parquet", "jsonl"], help="Force input format")
    parser.add_argument("--text-column", help="Column/field containing text")
    parser.add_argument("--id-column", help="Column/field containing document ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Rows per batch for parquet/JSONL (default: 64)")

    parser.add_argument("-q", "--queries", help="YAML file containing search queries")
    parser.add_argument("--config-dir", help="Custom configuration directory")
    parser.add_argument("--patterns-file", help="YAML file containing funding detection patterns")

    parser.add_argument("--retrieval-model", default="lightonai/GTE-ModernColBERT-v1", help="Retrieval model (default: lightonai/GTE-ModernColBERT-v1)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Minimum relevance score (default: 10.0)")
    parser.add_argument("--top-k", type=int, default=5, help="Top paragraphs per query (default: 5)")
    parser.add_argument("--semantic-batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")

    parser.add_argument("--normalize", action="store_true", help="Normalize funding statements")
    parser.add_argument("--heal-markdown", action="store_true", help="Reflow markdown before parsing")
    parser.add_argument("--statements-only", action="store_true", help="Treat each input as a pre-identified statement")
    parser.add_argument("--enable-pattern-rescue", action="store_true", help="Enable pattern-based rescue")
    parser.add_argument("--enable-post-filter", action="store_true", help="Enable post-filtering")

    parser.add_argument("--processing-batch-size", type=int, default=10, help="Documents per checkpoint batch (default: 10)")
    parser.add_argument("--workers", type=int, help="Parallel workers (auto-detected if not set)")
    parser.add_argument("--checkpoint-file", help="Checkpoint file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, reprocess all")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def _extract_from_document(
    document: DocumentPayload,
    queries: Dict[str, str],
    args: argparse.Namespace,
) -> List[Tuple[str, FundingStatement]]:
    try:
        content = document.load_text()
    except Exception as exc:
        print(f"  Warning: Failed to read {document.document_id}: {exc}")
        return []

    if args.heal_markdown:
        content = heal_markdown(content)

    if args.statements_only:
        content = content.strip()
        if not content:
            return []

        stmt = FundingStatement(statement=content, score=1.0, query="statements-only")

        if args.normalize:
            normalized = normalize_funding_statement(content)
            stmt.original = content
            stmt.statement = normalized

        if is_improperly_formatted(stmt.statement):
            stmt.is_problematic = True

        return [(document.document_id, stmt)]

    statements = extract_funding_statements(
        content=content,
        queries=queries,
        model_name=args.retrieval_model,
        top_k=args.top_k,
        threshold=args.threshold,
        batch_size=args.semantic_batch_size,
        patterns_file=args.patterns_file,
        custom_config_dir=args.config_dir,
        enable_pattern_rescue=args.enable_pattern_rescue,
    )

    if args.enable_post_filter and statements:
        statements = apply_post_filter(statements)

    for stmt in statements:
        if args.normalize:
            normalized = normalize_funding_statement(stmt.statement)
            stmt.original = stmt.statement
            stmt.statement = normalized
        if is_improperly_formatted(stmt.statement):
            stmt.is_problematic = True

    return [(document.document_id, stmt) for stmt in statements]


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint_file) if getattr(args, "checkpoint_file", None) else Path(f"{args.output}.checkpoint")

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        sys.exit(1)

    input_format = determine_input_format(input_path, getattr(args, "input_format", None))

    queries: Dict[str, str] = {}
    if not args.statements_only:
        queries = load_queries(
            queries_file=getattr(args, "queries", None),
            custom_config_dir=getattr(args, "config_dir", None),
        )

    checkpoint_repo = CheckpointRepository(checkpoint_path)
    resume = getattr(args, "resume", False)
    force = getattr(args, "force", False)
    checkpoint_data = checkpoint_repo.load(resume=resume and not force)
    checkpoint_data.setdefault("processed_files", {})
    processed_lookup = checkpoint_data.get("processed_files", {})
    verbose = getattr(args, "verbose", False)

    if resume and not force:
        print(f"Resuming from checkpoint: {len(processed_lookup)} documents already processed")

    documents_iter: Iterator[Tuple[DocumentPayload, str]]
    total_documents: Optional[int] = None
    text_fallbacks = [] if getattr(args, "text_column", None) else ["markdown", "content", "text", "body"]
    id_fallbacks = [] if getattr(args, "id_column", None) else ["source_id", "doc_id", "document_id", "id", "file_name", "filename"]

    if input_format == "parquet":
        iterable, discovered = stream_parquet_documents(
            input_path=input_path,
            text_column=getattr(args, "text_column", None),
            id_column=getattr(args, "id_column", None),
            batch_size=getattr(args, "batch_size", 64),
            fallback_text_columns=text_fallbacks,
            fallback_id_columns=id_fallbacks,
        )
        if discovered == 0:
            print("No rows found")
            return
        print(f"Discovered {discovered} rows")

        def parquet_iter() -> Iterator[Tuple[DocumentPayload, str]]:
            for doc in iterable:
                doc_hash = get_file_hash(doc.checkpoint_key)
                if force or doc_hash not in processed_lookup:
                    yield doc, doc_hash

        documents_iter = parquet_iter()
        total_documents = max((discovered or 0) - len(processed_lookup), 0) if not force else discovered

    elif input_format == "jsonl":
        iterable, discovered = stream_jsonl_documents(
            input_path=input_path,
            text_column=getattr(args, "text_column", None),
            id_column=getattr(args, "id_column", None),
            fallback_text_columns=text_fallbacks,
            fallback_id_columns=id_fallbacks,
        )
        if discovered == 0:
            print("No rows found")
            return
        print(f"Discovered {discovered} rows")

        def jsonl_iter() -> Iterator[Tuple[DocumentPayload, str]]:
            for doc in iterable:
                doc_hash = get_file_hash(doc.checkpoint_key)
                if force or doc_hash not in processed_lookup:
                    yield doc, doc_hash

        documents_iter = jsonl_iter()
        total_documents = max((discovered or 0) - len(processed_lookup), 0) if not force else discovered

    else:
        markdown_docs = build_markdown_documents(input_path)
        if not markdown_docs:
            print("No markdown files found")
            return

        docs_with_hashes = [
            (doc, get_file_hash(doc.checkpoint_key))
            for doc in markdown_docs
            if force or get_file_hash(doc.checkpoint_key) not in processed_lookup
        ]
        if not docs_with_hashes:
            print("All documents already processed")
            return

        total_documents = len(docs_with_hashes)
        print(f"Processing {total_documents} files...")
        documents_iter = iter(docs_with_hashes)

    all_statements: List[Tuple[str, FundingStatement]] = []
    processed_count = 0
    batch_buffer: List[Tuple[str, FundingStatement]] = []
    batch_size = getattr(args, "processing_batch_size", 10)

    worker_count = getattr(args, "workers", None)
    if worker_count is None:
        worker_count = max(1, (os.cpu_count() or 1) - 1)
    else:
        worker_count = max(1, worker_count)

    use_parallel = worker_count > 1 and not args.statements_only

    def flush_batch() -> None:
        nonlocal batch_buffer
        if not batch_buffer:
            return
        all_statements.extend(batch_buffer)
        checkpoint_repo.save()
        if total_documents is not None and total_documents > 0:
            print(f"Processed {processed_count}/{total_documents} documents, saved checkpoint")
        else:
            print(f"Processed {processed_count} documents, saved checkpoint")
        batch_buffer = []

    def handle_document(document: DocumentPayload, doc_hash: str, stmts: List[Tuple[str, FundingStatement]]) -> None:
        nonlocal processed_count, batch_buffer
        batch_buffer.extend(stmts)
        checkpoint_repo.record(doc_hash, {
            "document_id": document.document_id,
            "processed_at": datetime.now().isoformat(),
            "statements_found": len(stmts),
        })
        processed_count += 1
        if len(batch_buffer) >= batch_size:
            flush_batch()

    try:
        if use_parallel:
            ctx = multiprocessing.get_context("spawn")
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx)
            print(f"Using {worker_count} worker processes")

            pending = {}
            max_pending = worker_count * 2
            for document, doc_hash in documents_iter:
                while len(pending) >= max_pending:
                    done, _ = concurrent.futures.wait(pending.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        doc_meta = pending.pop(future)
                        try:
                            stmts = future.result()
                        except Exception as exc:
                            print(f"  Error processing {doc_meta[0].document_id}: {exc}")
                            stmts = []
                        handle_document(doc_meta[0], doc_meta[1], stmts)

                future = executor.submit(_extract_from_document, document, queries, args)
                pending[future] = (document, doc_hash)

            for future in concurrent.futures.as_completed(pending.keys()):
                doc_meta = pending[future]
                try:
                    stmts = future.result()
                except Exception as exc:
                    print(f"  Error processing {doc_meta[0].document_id}: {exc}")
                    stmts = []
                handle_document(doc_meta[0], doc_meta[1], stmts)
            executor.shutdown(wait=True)
        else:
            for document, doc_hash in documents_iter:
                if verbose:
                    print(f"Processing: {document.document_id}")
                stmts = _extract_from_document(document, queries, args)
                if verbose and stmts:
                    print(f"  Found {len(stmts)} statements")
                handle_document(document, doc_hash, stmts)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")

    flush_batch()

    count = write_statements_jsonl(output_path, all_statements)
    print(f"\nExtracted {count} funding statements from {processed_count} documents")
    print(f"Output: {output_path}")
