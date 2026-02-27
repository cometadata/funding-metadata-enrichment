# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "extract-funding-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=extract-funding-from-full-text",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
# ]
# ///
"""Run benchmark evaluation on predictions pushed by run_benchmark_entities.py.

Thin wrapper around ``python -m funding_extractor.benchmark`` so that
``hf jobs uv run`` can resolve dependencies via PEP 723 metadata.
All CLI arguments are forwarded directly.

Usage (HF job):
  hf jobs uv run \\
      --flavor cpu-basic \\
      --secrets HF_TOKEN \\
      --timeout 30m \\
      run_benchmark_metrics.py \\
      --hf-predictions cometadata/funding-extraction-harness-benchmark \\
      --hf-config qwen3-8b-entities-thinking --split both \\
      --push-to-hub cometadata/funding-extraction-harness-benchmark \\
      --push-config qwen3-8b-entities-thinking-metrics

Usage (local):
  python run_benchmark_metrics.py \\
      --hf-predictions cometadata/funding-extraction-harness-benchmark \\
      --hf-config qwen3-8b-entities-thinking --split both
"""

from funding_extractor.benchmark.cli import main

if __name__ == "__main__":
    main()
