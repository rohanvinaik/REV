#!/bin/bash

# Resume LLaMA 3.1 405B FP8 download
MODEL_ID="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8"
LOCAL_DIR="/Users/rohanvinaik/LLM_models/llama-3.1-405b-fp8"

echo "================================================================================
Resuming LLaMA 3.1 405B FP8 Download
Model: $MODEL_ID
Directory: $LOCAL_DIR
================================================================================
"

# Create directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Use huggingface-cli to download with resume capability
huggingface-cli download \
    "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo "
Download complete or resumed!"