# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm-funding-extraction[benchmark] @ git+https://github.com/cometadata/funding-metadata-enrichment.git@20260217-adapt-to-benchmark#subdirectory=vllm-funding-extraction",
#     "datasets>=4.5.0",
#     "huggingface-hub>=0.34.0,<1.0",
# ]
# ///
"""Run benchmark metrics evaluation on HuggingFace predictions.

CPU-only job — no GPU or vLLM dependency needed.

Usage (HF job):
  hf jobs uv run \\
      --flavor cpu-basic \\
      --secrets HF_TOKEN \\
      --timeout 30m \\
      run_benchmark_metrics.py \\
      --hf-predictions cometadata/funding-extraction-harness-benchmark \\
      --hf-config qwen3-8b-entities-non-thinking \\
      --split both \\
      --push-to-hub cometadata/funding-extraction-harness-benchmark \\
      --push-config qwen3-8b-entities-non-thinking-metrics

Usage (local):
  python run_benchmark_metrics.py \\
      --hf-predictions cometadata/funding-extraction-harness-benchmark \\
      --hf-config qwen3-8b-entities-non-thinking \\
      --split both --output-json report.json
"""

import os
import sys

from huggingface_hub import login

from funding_extraction.benchmark.cli import main


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    main()
