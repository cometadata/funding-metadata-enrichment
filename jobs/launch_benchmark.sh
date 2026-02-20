#!/usr/bin/env bash
set -euo pipefail

# Launch funding extraction benchmark as a HuggingFace Job.
#
# Usage:
#   ./launch_benchmark.sh --stages statements --flavor a100-large
#   ./launch_benchmark.sh --stages entities --flavor a100-large
#   ./launch_benchmark.sh --stages entities --model Qwen/Qwen3-8B --mode online --flavor a100-large
#   ./launch_benchmark.sh --stages entities --model Qwen/Qwen3-8B --lora-path /path --lora-name adapter

STAGES=""
FLAVOR="l4x1"
MODEL="Qwen/Qwen3-8B"
MODE="offline"
LORA_PATH=""
LORA_NAME=""
WORKERS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)     STAGES="$2"; shift 2 ;;
        --flavor)     FLAVOR="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --lora-path)  LORA_PATH="$2"; shift 2 ;;
        --lora-name)  LORA_NAME="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$STAGES" ]]; then
    echo "Error: --stages is required (statements or entities)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$STAGES" in
    statements)
        SCRIPT="${SCRIPT_DIR}/run_benchmark_statements.py"
        SCRIPT_ARGS=""
        ;;
    entities)
        SCRIPT="${SCRIPT_DIR}/run_benchmark_entities.py"
        SCRIPT_ARGS="--model-id ${MODEL} --mode ${MODE} --workers ${WORKERS}"
        [[ -n "$LORA_PATH" ]] && SCRIPT_ARGS+=" --lora-path ${LORA_PATH}"
        [[ -n "$LORA_NAME" ]] && SCRIPT_ARGS+=" --lora-name ${LORA_NAME}"
        ;;
    *)
        echo "Error: --stages must be 'statements' or 'entities'" >&2
        exit 1
        ;;
esac

echo "Launching benchmark extraction job:"
echo "  Stage:  ${STAGES}"
echo "  Flavor: ${FLAVOR}"
if [[ "$STAGES" == "entities" ]]; then
    echo "  Model:  ${MODEL}"
    echo "  Mode:   ${MODE}"
    echo "  Workers: ${WORKERS}"
    [[ -n "$LORA_PATH" ]] && echo "  LoRA:   ${LORA_NAME:-default} (${LORA_PATH})"
fi
echo "  Script: $(basename "$SCRIPT")"

hf jobs uv run \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    --timeout 2h \
    "$SCRIPT" \
    $SCRIPT_ARGS
