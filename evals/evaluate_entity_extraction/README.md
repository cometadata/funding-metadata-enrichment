# Entity Extraction Evaluation

Evaluate extracted funder/award entities against ground truth, with funder-level aggregation and per-attribute (award ID, scheme, title) scoring.

## Inputs
- Predictions (`--predictions`): JSONL with `funding_statement` + `predicted_funders` (or `funders`) per line. Also accepts an HF dataset repo id.
- Gold file (`--gold-file`): local JSONL joined to predictions by `funding_statement`. Defaults to `arxiv_test.jsonl` next to the script. Pass empty string to fall back to `--gold-dataset` (HF, joined by DOI).
- Funder threshold (`--funder-threshold`): similarity cutoff for funder name matching (default 0.8).
- Threshold (`--threshold`): similarity cutoff for scheme/title matching (default 0.8).
- Award ID mode (`--id-match-mode`): `normalized` (default) or `exact`.
- Output path (`--output`): where metrics JSON is written (default: `<predictions_basename>_results.json`).

## Usage
```bash
python evaluate_entity_extraction.py \
  --predictions predictions.jsonl \
  --gold-file arxiv_test.jsonl \
  --funder-threshold 0.8 \
  --threshold 0.8 \
  --output evaluation_results.json
```

Reports P/R/F1/F0.5/F1.5 at funder, award_id, scheme, and title levels, split by complete vs. truncated thinking, plus a permissive/balanced/strict similarity diagnostic.
