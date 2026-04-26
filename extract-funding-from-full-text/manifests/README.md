# arxiv-funding-statement-extractions

Funding-statement extractions over `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/`.

- Extractor: `funding_statement_extractor` @ `statement-only-extraction`
- Model: `lightonai/GTE-ModernColBERT-v1`
- Config: Tier 2 (paragraph prefilter, regex floor 11.0, top_k=5, threshold=10.0)
- Hardware: A100-large bf16, batch_size 512
- Files processed: 17589
- Status: {'done': 17589}
- Total rows: 2,687,527
- Total worker seconds: 336167
- Est cost @ $5/hr: $466.90
- Completed: 2026-04-26T04:28:58.460145+00:00

## Schema

Per row in `predictions/*.parquet`:
- `arxiv_id`, `doc_id`, `input_file`, `row_idx`
- `predicted_statements`: list[str]
- `predicted_details`: list[struct{statement, score, query, paragraph_idx}]
- `text_length`, `latency_ms`, `error`
