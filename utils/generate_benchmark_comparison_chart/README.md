# Generate Benchmark Comparison Chart

Generates a side-by-side comparison chart (PNG) of precision, recall, and F1 scores for two models across extraction levels (funder, award ID, funding scheme, award title).

## Usage

```bash
uv run generate_benchmark_comparison_chart.py \
  path/to/model_a_benchmark.json \
  path/to/model_b_benchmark.json \
  --label-a "Model A" \
  --label-b "Model B" \
  -o comparison.png
```

## Input Format

Expects benchmark JSON files containing an `aggregate_metrics` key with nested metrics per level:

```json
{
  "aggregate_metrics": {
    "funder": {"precision": 0.95, "recall": 0.90, "f1": 0.92},
    "award_id": {"precision": 0.85, "recall": 0.80, "f1": 0.82},
    "funding_scheme": {"precision": 0.70, "recall": 0.65, "f1": 0.67},
    "award_title": {"precision": 0.60, "recall": 0.55, "f1": 0.57}
  }
}
```
