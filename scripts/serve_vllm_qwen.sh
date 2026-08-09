#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=========================================="
echo " vLLM Server"
echo "=========================================="

echo
echo "Hostname:"
hostname

echo
echo "Container:"
echo "$VLLM_CONTAINER"

echo
echo "Model:"
echo "$VLLM_MODEL"

echo
echo "HF_HOME:"
echo "$HF_HOME"

echo
echo "Checking GPU..."
nvidia-smi

echo
echo "Checking container..."

if [ ! -f "$VLLM_CONTAINER" ]; then
    echo "ERROR: Container not found:"
    echo "$VLLM_CONTAINER"
    exit 1
fi

echo "Container found."

echo
echo "Starting vLLM..."
echo

apptainer exec --nv \
    "$VLLM_CONTAINER" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --enforce-eager \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
