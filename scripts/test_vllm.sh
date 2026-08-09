#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=========================================="
echo " vLLM API Test"
echo "=========================================="

echo
echo "Testing:"
echo "  http://localhost:$VLLM_PORT/v1/models"

curl -s \
    "http://localhost:$VLLM_PORT/v1/models"

echo
echo
echo "Sending test completion..."

curl -s \
    "http://localhost:$VLLM_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$VLLM_MODEL\",
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Explain KV cache in one sentence.\"
            }
        ],
        \"temperature\": 0
    }"

echo
echo
echo "Test complete."
