#!/usr/bin/env bash
set -euo pipefail

# Launch funding extraction benchmark as a HuggingFace Job.
#
# Usage:
#   ./launch_benchmark.sh --stages statements --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b.yaml --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b-thinking.yaml --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config llama-3.1-8b-lora.yaml --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b-lora.yaml --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b-lora-thinking.yaml --flavor a100-large
#   ./launch_benchmark.sh --stages entities --vllm-config llama-3.1-8b.yaml --max-tokens 8192
#   ./launch_benchmark.sh --stages metrics --config-name qwen3-8b-entities-thinking --flavor cpu-basic

STAGES=""
FLAVOR="a100-large"
VLLM_CONFIG=""
WORKERS=""
CONFIG_NAME=""
MAX_TOKENS=""
THINKING_BUDGET=""
EXTRACTION_PASSES=""
GPU_MEM_UTIL=""
MAX_MODEL_LEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)             STAGES="$2"; shift 2 ;;
        --flavor)             FLAVOR="$2"; shift 2 ;;
        --vllm-config)        VLLM_CONFIG="$2"; shift 2 ;;
        --workers)            WORKERS="$2"; shift 2 ;;
        --config-name)        CONFIG_NAME="$2"; shift 2 ;;
        --max-tokens)         MAX_TOKENS="$2"; shift 2 ;;
        --thinking-budget)    THINKING_BUDGET="$2"; shift 2 ;;
        --extraction-passes)  EXTRACTION_PASSES="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)      MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$STAGES" ]]; then
    echo "Error: --stages is required (statements, entities, or metrics)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="cometadata/funding-extraction-harness-benchmark"

case "$STAGES" in
    statements)
        SCRIPT="${SCRIPT_DIR}/run_benchmark_statements.py"
        SCRIPT_ARGS=""
        ;;
    entities)
        SCRIPT="${SCRIPT_DIR}/run_benchmark_entities.py"
        SCRIPT_ARGS="--vllm-config ${VLLM_CONFIG}"
        [[ -n "$WORKERS" ]] && SCRIPT_ARGS+=" --workers ${WORKERS}"
        [[ -n "$CONFIG_NAME" ]] && SCRIPT_ARGS+=" --config-name ${CONFIG_NAME}"
        [[ -n "$MAX_TOKENS" ]] && SCRIPT_ARGS+=" --max-tokens ${MAX_TOKENS}"
        [[ -n "$THINKING_BUDGET" ]] && SCRIPT_ARGS+=" --thinking-budget ${THINKING_BUDGET}"
        [[ -n "$EXTRACTION_PASSES" ]] && SCRIPT_ARGS+=" --extraction-passes ${EXTRACTION_PASSES}"
        [[ -n "$GPU_MEM_UTIL" ]] && SCRIPT_ARGS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
        [[ -n "$MAX_MODEL_LEN" ]] && SCRIPT_ARGS+=" --max-model-len ${MAX_MODEL_LEN}"
        ;;
    metrics)
        if [[ -z "$CONFIG_NAME" ]]; then
            echo "Error: --config-name is required for metrics stage" >&2
            exit 1
        fi
        SCRIPT="${SCRIPT_DIR}/run_benchmark_metrics.py"
        SCRIPT_ARGS="--hf-predictions ${DATASET} --hf-config ${CONFIG_NAME} --split both"
        SCRIPT_ARGS+=" --push-to-hub ${DATASET} --push-config ${CONFIG_NAME}-metrics"
        ;;
    *)
        echo "Error: --stages must be 'statements', 'entities', or 'metrics'" >&2
        exit 1
        ;;
esac

echo "Launching benchmark job:"
echo "  Stage:  ${STAGES}"
echo "  Flavor: ${FLAVOR}"
if [[ "$STAGES" == "entities" ]]; then
    echo "  Config: ${VLLM_CONFIG}"
    [[ -n "$WORKERS" ]] && echo "  Workers: ${WORKERS}"
    [[ -n "$MAX_TOKENS" ]] && echo "  Max tokens: ${MAX_TOKENS}"
    [[ -n "$THINKING_BUDGET" ]] && echo "  Thinking budget: ${THINKING_BUDGET}"
    [[ -n "$EXTRACTION_PASSES" ]] && echo "  Extraction passes: ${EXTRACTION_PASSES}"
    [[ -n "$GPU_MEM_UTIL" ]] && echo "  GPU memory: ${GPU_MEM_UTIL}"
    [[ -n "$MAX_MODEL_LEN" ]] && echo "  Max model len: ${MAX_MODEL_LEN}"
    [[ -n "$CONFIG_NAME" ]] && echo "  HF config: ${CONFIG_NAME}"
fi
if [[ "$STAGES" == "metrics" ]]; then
    echo "  Config: ${CONFIG_NAME}"
    echo "  Push:   ${DATASET} (config: ${CONFIG_NAME}-metrics)"
fi
echo "  Script: $(basename "$SCRIPT")"

hf jobs uv run \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    --timeout 2h \
    "$SCRIPT" \
    $SCRIPT_ARGS
