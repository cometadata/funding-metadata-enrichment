import argparse

import sys
import concurrent.futures
import os

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from funding_extractor.entities.extraction import extract_structured_entities
from funding_extractor.entities.io import read_statements_by_document, write_results_json
from funding_extractor.entities.models import ExtractionResult
from funding_extractor.io.checkpointing import CheckpointRepository, get_file_hash
from funding_extractor.models import DocumentResult, ProcessingParameters, ProcessingResults
from funding_extractor.providers.base import ModelProvider
from funding_extractor.statements.models import FundingStatement




def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file of funding statements")
    parser.add_argument("-o", "--output", default="funding_results.json", help="Output JSON file (default: funding_results.json)")

    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama", "openai", "local_openai"],
        default="gemini",
        help="LLM provider (default: gemini)",
    )
    parser.add_argument("--model", help="Model ID")
    parser.add_argument("--model-url", help="API endpoint URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], help="Reasoning effort level")
    parser.add_argument("--timeout", type=int, default=60, help="LLM request timeout (default: 60)")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip model validation")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    parser.add_argument("--prompt-file", help="Extraction prompt file")
    parser.add_argument("--examples-file", help="Extraction examples file")
    parser.add_argument("--config-dir", help="Custom configuration directory")

    parser.add_argument("--workers", type=int, help="Parallel workers (auto-detected if not set)")
    parser.add_argument("--processing-batch-size", type=int, default=10, help="Statements per checkpoint batch (default: 10)")
    parser.add_argument("--checkpoint-file", help="Checkpoint file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, reprocess all")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def _extract_single(
    statement_text: str,
    args: argparse.Namespace,
) -> ExtractionResult:
    return extract_structured_entities(
        funding_statement=statement_text,
        provider=ModelProvider(args.provider),
        model_id=getattr(args, "model", None),
        model_url=getattr(args, "model_url", None),
        api_key=getattr(args, "api_key", None),
        skip_model_validation=getattr(args, "skip_model_validation", False),
        timeout=getattr(args, "timeout", 60),
        debug=getattr(args, "debug", False),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        prompt_file=getattr(args, "prompt_file", None),
        examples_file=getattr(args, "examples_file", None),
        custom_config_dir=getattr(args, "config_dir", None),
    )


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint_file) if getattr(args, "checkpoint_file", None) else Path(f"{args.output}.checkpoint")
    verbose = getattr(args, "verbose", False)
    force = getattr(args, "force", False)
    resume = getattr(args, "resume", False)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    grouped = read_statements_by_document(input_path)
    if not grouped:
        print("No statements found in input file.")
        return

    total_statements = sum(len(stmts) for stmts in grouped.values())
    print(f"Loaded {total_statements} statements from {len(grouped)} documents")

    checkpoint_repo = CheckpointRepository(checkpoint_path)
    checkpoint_data = checkpoint_repo.load(resume=resume and not force)
    checkpoint_data.setdefault("processed_statements", {})
    processed_lookup = checkpoint_data.get("processed_statements", {})

    if resume and not force:
        print(f"Resuming: {len(processed_lookup)} statements already processed")

    # Build results structure
    doc_results: Dict[str, DocumentResult] = {}
    extraction_map: Dict[str, ExtractionResult] = {}

    # Populate statement metadata for all documents
    for doc_id, records in grouped.items():
        doc = DocumentResult(filename=doc_id)
        for record in records:
            stmt = FundingStatement(
                statement=record["statement"],
                original=record.get("original"),
                score=record.get("score", 0.0),
                query=record.get("query", ""),
                paragraph_idx=record.get("paragraph_idx"),
                is_problematic=record.get("is_problematic", False),
            )
            doc.funding_statements.append(stmt)
        doc_results[doc_id] = doc

    # Collect unique statements to extract
    unique_statements = set()
    for records in grouped.values():
        for record in records:
            unique_statements.add(record["statement"])

    statements_to_process = [
        s for s in unique_statements
        if force or get_file_hash(s) not in processed_lookup
    ]

    if not statements_to_process:
        print("All statements already processed")
    else:
        print(f"Extracting entities from {len(statements_to_process)} unique statements...")

        worker_count = getattr(args, "workers", None)
        if worker_count is None:
            worker_count = max(1, (os.cpu_count() or 1) - 1)
        else:
            worker_count = max(1, worker_count)

        use_parallel = worker_count > 1
        processed_count = 0
        batch_size = getattr(args, "processing_batch_size", 10)

        def handle_result(statement_text: str, result: ExtractionResult) -> None:
            nonlocal processed_count
            extraction_map[statement_text] = result
            stmt_hash = get_file_hash(statement_text)
            checkpoint_repo.record(stmt_hash, {
                "statement_preview": statement_text[:100],
                "processed_at": datetime.now().isoformat(),
                "funders_found": len(result.funders),
            })
            processed_count += 1
            if processed_count % batch_size == 0:
                checkpoint_repo.save()
                print(f"  Processed {processed_count}/{len(statements_to_process)} statements")

        if use_parallel:
            print(f"Using {worker_count} worker threads")
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_stmt = {
                    executor.submit(_extract_single, stmt, args): stmt
                    for stmt in statements_to_process
                }
                for future in concurrent.futures.as_completed(future_to_stmt):
                    stmt_text = future_to_stmt[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"  Warning: Extraction failed for statement: {exc}")
                        result = ExtractionResult(statement=stmt_text, funders=[])
                    handle_result(stmt_text, result)
        else:
            for stmt_text in statements_to_process:
                if verbose:
                    print(f"  Extracting: {stmt_text[:80]}...")
                try:
                    result = _extract_single(stmt_text, args)
                except Exception as exc:
                    print(f"  Warning: Extraction failed: {exc}")
                    result = ExtractionResult(statement=stmt_text, funders=[])
                handle_result(stmt_text, result)

        checkpoint_repo.save()

    # Assign extraction results to documents
    for doc_id, doc in doc_results.items():
        seen = set()
        for stmt in doc.funding_statements:
            if stmt.statement in seen:
                continue
            seen.add(stmt.statement)
            if stmt.statement in extraction_map:
                doc.extraction_results.append(extraction_map[stmt.statement])

    results = ProcessingResults(
        timestamp=datetime.now().isoformat(),
        parameters=ProcessingParameters(
            input_path=str(input_path),
            input_format="jsonl",
            provider=args.provider,
            model=getattr(args, "model", None),
        ),
        results=doc_results,
        summary={},
    )
    results.update_summary()

    write_results_json(results, output_path)

    print(f"\nProcessed {len(doc_results)} documents")
    print(f"Total funders: {results.summary.get('total_funders', 0)}")
    print(f"Total awards: {results.summary.get('total_awards', 0)}")
    print(f"Output: {output_path}")
