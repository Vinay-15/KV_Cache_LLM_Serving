#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=========================================="
echo " Alpine A100 MIG GPU Session"
echo "=========================================="

echo
echo "Project:"
echo "  $PROJECT_DIR"

echo "Container:"
echo "  $VLLM_CONTAINER"

echo
echo "GPU request:"
echo "  Partition : aa100"
echo "  QoS       : gpu-testing"
echo "  GPU       : A100 3g.20gb"
echo "  Nodes     : 1"
echo "  Time      : 1 hour"
echo

sinteractive \
    --partition=aa100 \
    --qos=gpu-testing \
    --gres=gpu:a100_3g.20gb:1 \
    --ntasks=10 \
    --nodes=1 \
    --time=01:00:00
