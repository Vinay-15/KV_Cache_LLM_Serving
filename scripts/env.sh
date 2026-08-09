#!/bin/bash

# ============================================================
# KV Cache / LLM Serving Project - Alpine Configuration
# ============================================================

# Project
export PROJECT_DIR=/projects/$USER/kv-cache-llm-serving

# Persistent storage
export APPTAINER_CACHEDIR=/projects/$USER/apptainer_cache
export HF_HOME=/projects/$USER/hf

# Known-good vLLM container.
# This image is intentionally used because newer vLLM images
# currently have CUDA/NVIDIA driver compatibility issues on Alpine.
export VLLM_CONTAINER=/projects/$USER/vllm-085.sif

# Model
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Server
export VLLM_HOST=0.0.0.0
export VLLM_PORT=8000

# vLLM configuration
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.3
