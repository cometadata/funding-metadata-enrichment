# Funding Extraction Evaluation

Evaluate extraxted funding statements against a gold JSON file.

## Inputs
- Predictions JSON (`-p/--predictions`) derived from funding extraction.
- Gold JSON (`-g/--gold`): list of objects with `filename` and `statements` fields.
- Threshold (`-t/--threshold`): similarity cutoff (0.0-1.0).
- Output path (`-o/--output-json`): where metrics JSON is written.

## Usage
```bash
python evaluate_funding_extraction.py \
  -p new_funding_statements.json \
  -g train.json \
  -t 0.8 \
  -o evaluation_train_results.json
```
