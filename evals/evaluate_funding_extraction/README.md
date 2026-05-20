# Funding Extraction Evaluation

Evaluate cascade funding-statement predictions against a gold file, scored
across five match-strictness tiers under two match definitions.

## Tiers

| tier | match |
|---|---|
| L0 | exact |
| L1 | equal after whitespace/encoding normalization |
| L2 | equal after aggressive normalization (case / punctuation / markup) |
| L3 | fuzzy similarity ≥ `--fuzzy-threshold` |
| L4 | ≥ 80% of the prediction's chunks individually fuzzy-hit gold |

Each tier is scored under two match definitions:

- **equality** — prediction vs gold as whole statements; penalizes under- and
  over-extraction.
- **containment** — the prediction is found *within* the gold statement;
  credits a correct fragment of a longer gold statement.

Precision/recall are cumulative at each tier — a document is correct at tier X
if it matched at X or stricter:

```
precision@X = correct@X / non-empty predictions
recall@X    = correct@X / documents with a gold statement
```

## Inputs

All inputs are local files — `.parquet`, `.json`, or `.jsonl`.

- Predictions (`-p/--predictions`): a **file** with per-document records
  carrying `doc_id` and `funding_statement` (optionally `text_source`), or a
  **directory** — every parquet/JSON under it is read, and a containing
  subdirectory name labels each file's `text_source` when the column is absent.
  So a downloaded `predictions/<text_source>/<split>.parquet` tree is scored
  per source; non-prediction files (e.g. `worker_status/*.json`) are skipped.
  Results are grouped by `text_source`.
- Gold (`-g/--gold`): a file with per-document records carrying `doc_id` and
  the gold `funding_statements` list.
- Fuzzy threshold (`-t/--fuzzy-threshold`): L3 similarity cutoff 0–100
  (default 85).
- Compare (`-c/--compare`): optional second predictions file/directory; the run
  reports per-tier F1 deltas against it (e.g. cascade with vs. without the
  stage-3 cleaner).
- Output path (`-o/--output-json`): where the metrics JSON is written.

JSON/JSONL records are a list of objects (or, for JSON, a `{id: record}`
mapping). Parquet is read column-wise.

## Usage

```bash
uv run evaluate_funding_extraction.py \
  -p predictions/ \
  -g test.parquet \
  -o evaluation_results.json
```

Compare two prediction runs (e.g. cleaner on vs. off):

```bash
uv run evaluate_funding_extraction.py \
  -p cascade_out/predictions/ \
  -c cascade_out_nocleaner/predictions/ \
  -g test.parquet \
  -o evaluation_cleaner_vs_nocleaner.json
```

Prints per-tier precision/recall/F1 tables for every `text_source` under both
match definitions (plus F1 deltas when `--compare` is given), and writes the
full metrics — including false-positive/negative and unmatched `doc_id`
lists — to the output JSON.
