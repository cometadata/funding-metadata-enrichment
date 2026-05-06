# Compare Rollouts to Gold

Compare rollout predictions against gold annotations. Matches records by DOI, diffs each field (funder names, award IDs, funding schemes, award titles), and reports errors as counts, percentages, and per-DOI breakdowns.

Runs two modes — strict (exact match after normalization) and fuzzy (rapidfuzz, threshold-based) — and writes results to separate subdirectories.

## Installation

```
pip install ftfy rapidfuzz
```

## Usage

```
python compare_rollouts_to_gold.py \
  -p predictions.jsonl \
  -g gold.jsonl \
  [-o output_dir] \
  [-t threshold]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-p` | Predictions JSONL (SFT chat format: system/user/assistant messages + `doi`) | required |
| `-g` | Gold annotations JSONL (`doi`, `funding_statement`, `funders`) | required |
| `-o` | Output directory | `error_analysis` |
| `-t` | Fuzzy similarity threshold (0-100) | `90` |

## Input formats

Predictions — JSONL, one record per line:
```json
{"doi": "10.xxxx/yyyy", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "[{\"funder_name\": ..., \"awards\": [...]}]"}]}
```

Gold — JSONL, one record per line:
```json
{"doi": "10.xxxx/yyyy", "funding_statement": "...", "funders": [{"funder_name": "...", "awards": [{"funding_scheme": [], "award_ids": [], "award_title": []}]}]}
```

## Normalization

Text normalization (applied before all comparisons) uses `ftfy` for encoding repair, replaces smart quotes, collapses whitespace, strips bullets/stars, and lowercases. Fuzzy mode additionally uses `rapidfuzz` (`ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`) with greedy bipartite matching.

## Output

```
<output_dir>/
├── strict/
│   ├── all_errors.csv / .jsonl
│   ├── per_doi_summary.csv / .jsonl
│   ├── error_type_summary.csv
│   ├── side_by_side_errors.csv / .jsonl
│   ├── parse_errors.jsonl
│   └── summary_report.txt
└── fuzzy/
    └── (same files)
```

| File | Contents |
|------|----------|
| `all_errors` | Every individual error with DOI, type, field, gold/pred values, detail |
| `per_doi_summary` | Error count and types per DOI, sorted worst-first |
| `error_type_summary` | Aggregate counts and percentages by error type |
| `side_by_side_errors` | Gold vs pred with explicit missing/extra values per field |
| `parse_errors` | Predictions with malformed JSON output |
| `summary_report.txt` | Full text report |

## Error types

| Type | Meaning |
|------|---------|
| `missing_funder` / `extra_funder` | Funder in gold but not pred, or vice versa |
| `missing_award_id` / `extra_award_id` | Award ID missing or hallucinated |
| `missing_scheme` / `extra_scheme` | Funding scheme missing or hallucinated |
| `missing_award_title` / `extra_award_title` | Award title missing or hallucinated |
| `funder_count_mismatch` | Different total funder-award entry counts |
| `award_count_mismatch` | Different award counts within a matched funder |
| `null_funder_count_mismatch` | Different counts of null-funder entries |
