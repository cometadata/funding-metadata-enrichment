#!/usr/bin/env bash
set -euo pipefail

# Launch funding extraction benchmark as a HuggingFace Job.
#
# Usage:
#   # Entity extraction — test split only (default)
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b.yaml --flavor a100-large
#
#   # Entity extraction — both splits
#   ./launch_benchmark.sh --stages entities --vllm-config qwen3-8b.yaml --split both --flavor a100-large
#
#   # Metrics evaluation (CPU)
#   ./launch_benchmark.sh --stages metrics --config-name qwen3-8b-entities-non-thinking
#
#   # Both stages sequentially
#   ./launch_benchmark.sh --stages entities --stages metrics --vllm-config qwen3-8b.yaml --config-name qwen3-8b-entities-non-thinking --flavor a100-large

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
LOG_LEVEL=""
SPLIT="test"
STAGES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)             STAGES+=("$2"); shift 2 ;;
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
        --log-level)          LOG_LEVEL="$2"; shift 2 ;;
        --split)              SPLIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Default to entities if no stages specified
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=("entities")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for STAGE in "${STAGES[@]}"; do
    case "$STAGE" in
        entities)
            if [[ -z "$VLLM_CONFIG" ]]; then
                echo "Error: --vllm-config is required for entities stage" >&2
                exit 1
            fi

            SCRIPT="${SCRIPT_DIR}/run_benchmark_entities.py"
            SCRIPT_ARGS="--vllm-config ${VLLM_CONFIG}"

            SCRIPT_ARGS+=" --split ${SPLIT}"
            [[ -n "$WORKERS" ]] && SCRIPT_ARGS+=" --workers ${WORKERS}"
            [[ -n "$CONFIG_NAME" ]] && SCRIPT_ARGS+=" --config-name ${CONFIG_NAME}"
            [[ -n "$MAX_TOKENS" ]] && SCRIPT_ARGS+=" --max-tokens ${MAX_TOKENS}"
            [[ -n "$THINKING_BUDGET" ]] && SCRIPT_ARGS+=" --thinking-budget ${THINKING_BUDGET}"
            [[ -n "$EXTRACTION_PASSES" ]] && SCRIPT_ARGS+=" --extraction-passes ${EXTRACTION_PASSES}"
            [[ -n "$GPU_MEM_UTIL" ]] && SCRIPT_ARGS+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
            [[ -n "$MAX_MODEL_LEN" ]] && SCRIPT_ARGS+=" --max-model-len ${MAX_MODEL_LEN}"
            [[ -n "$GUIDED_DECODING" ]] && SCRIPT_ARGS+=" --guided-decoding"
            [[ -n "$LOG_LEVEL" ]] && SCRIPT_ARGS+=" --log-level ${LOG_LEVEL}"

            echo "Launching entities benchmark job:"
            echo "  Config: ${VLLM_CONFIG}"
            echo "  Split: ${SPLIT}"
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
            ;;

        metrics)
            if [[ -z "$CONFIG_NAME" ]]; then
                echo "Error: --config-name is required for metrics stage" >&2
                exit 1
            fi

            SCRIPT="${SCRIPT_DIR}/run_benchmark_metrics.py"
            SCRIPT_ARGS="--hf-predictions cometadata/funding-extraction-harness-benchmark"
            SCRIPT_ARGS+=" --hf-config ${CONFIG_NAME}"
            SCRIPT_ARGS+=" --split ${SPLIT}"
            SCRIPT_ARGS+=" --push-to-hub cometadata/funding-extraction-harness-benchmark"
            SCRIPT_ARGS+=" --push-config ${CONFIG_NAME}-metrics"

            echo "Launching metrics benchmark job:"
            echo "  HF config: ${CONFIG_NAME}"
            echo "  Flavor: cpu-basic"

            hf jobs uv run \
                --flavor cpu-basic \
                --secrets HF_TOKEN \
                --timeout 30m \
                "$SCRIPT" \
                $SCRIPT_ARGS
            ;;

        *)
            echo "Error: unknown stage '$STAGE' (use 'entities' or 'metrics')" >&2
            exit 1
            ;;
    esac
done
