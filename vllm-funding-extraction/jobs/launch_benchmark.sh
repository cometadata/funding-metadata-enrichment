#!/usr/bin/env bash
set -euo pipefail

# Launch funding extraction benchmark as a HuggingFace Job.
#
# Usage:
#   ./launch_benchmark.sh --vllm-config qwen3-8b.yaml --flavor a100-large
#   ./launch_benchmark.sh --vllm-config qwen3.5-9b-thinking.yaml --flavor a100-large
#   ./launch_benchmark.sh --vllm-config qwen3-8b.yaml --guided-decoding --flavor a100-large

FLAVOR="a100-large"
VLLM_CONFIG=""
WORKERS=""
CONFIG_NAME=""
MAX_TOKENS=""
THINKING_BUDGET=""
EXTRACTION_PASSES=""
GPU_MEM_UTIL=""
MAX_MODEL_LEN=""
GUIDED_DECODING=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flavor)             FLAVOR="$2"; shift 2 ;;
        --vllm-config)        VLLM_CONFIG="$2"; shift 2 ;;
        --workers)            WORKERS="$2"; shift 2 ;;
        --config-name)        CONFIG_NAME="$2"; shift 2 ;;
        --max-tokens)         MAX_TOKENS="$2"; shift 2 ;;
        --thinking-budget)    THINKING_BUDGET="$2"; shift 2 ;;
        --extraction-passes)  EXTRACTION_PASSES="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2"; shift 2 ;;
        --guided-decoding)    GUIDED_DECODING="true"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$VLLM_CONFIG" ]]; then
    echo "Error: --vllm-config is required" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/run_benchmark_entities.py"
SCRIPT_ARGS="--vllm-config ${VLLM_CONFIG}"

[[ -n "$WORKERS" ]] && SCRIPT_ARGS+=" --workers ${WORKERS}"
[[ -n "$CONFIG_NAME" ]] && SCRIPT_ARGS+=" --config-name ${CONFIG_NAME}"
[[ -n "$MAX_TOKENS" ]] && SCRIPT_ARGS+=" --max-tokens ${MAX_TOKENS}"
[[ -n "$THINKING_BUDGET" ]] && SCRIPT_ARGS+=" --thinking-budget ${THINKING_BUDGET}"
[[ -n "$EXTRACTION_PASSES" ]] && SCRIPT_ARGS+=" --extraction-passes ${EXTRACTION_PASSES}"
[[ -n "$GPU_MEM_UTIL" ]] && SCRIPT_ARGS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
[[ -n "$MAX_MODEL_LEN" ]] && SCRIPT_ARGS+=" --max-model-len ${MAX_MODEL_LEN}"
[[ -n "$GUIDED_DECODING" ]] && SCRIPT_ARGS+=" --guided-decoding"

echo "Launching benchmark job:"
echo "  Config: ${VLLM_CONFIG}"
echo "  Flavor: ${FLAVOR}"
[[ -n "$WORKERS" ]] && echo "  Workers: ${WORKERS}"
[[ -n "$MAX_TOKENS" ]] && echo "  Max tokens: ${MAX_TOKENS}"
[[ -n "$THINKING_BUDGET" ]] && echo "  Thinking budget: ${THINKING_BUDGET}"
[[ -n "$EXTRACTION_PASSES" ]] && echo "  Extraction passes: ${EXTRACTION_PASSES}"
[[ -n "$GPU_MEM_UTIL" ]] && echo "  GPU memory: ${GPU_MEM_UTIL}"
[[ -n "$MAX_MODEL_LEN" ]] && echo "  Max model len: ${MAX_MODEL_LEN}"
[[ -n "$GUIDED_DECODING" ]] && echo "  Guided decoding: enabled"
[[ -n "$CONFIG_NAME" ]] && echo "  HF config: ${CONFIG_NAME}"

hf jobs uv run \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    --timeout 2h \
    "$SCRIPT" \
    $SCRIPT_ARGS
