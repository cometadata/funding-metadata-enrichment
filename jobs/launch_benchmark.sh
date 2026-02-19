#!/usr/bin/env bash
set -euo pipefail

# Launch funding extraction benchmark as a HuggingFace Job.
#
# Usage:
#   ./launch_benchmark.sh                                          # defaults
#   ./launch_benchmark.sh --stages entities                        # entities only
#   ./launch_benchmark.sh --stages both --flavor a10g-small        # both stages, custom GPU
#   ./launch_benchmark.sh --model Qwen/Qwen3-8B --flavor l4x1     # custom model + GPU

STAGES="both"
FLAVOR="l4x1"
MODEL="Qwen/Qwen3-8B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)   STAGES="$2"; shift 2 ;;
        --flavor)   FLAVOR="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Launching benchmark extraction job:"
echo "  Stages: ${STAGES}"
echo "  Flavor: ${FLAVOR}"
echo "  Model:  ${MODEL}"

hf jobs uv run \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    --timeout 2h \
    "${SCRIPT_DIR}/run_benchmark_extraction.py" \
    --stages "$STAGES" --model-id "$MODEL"
