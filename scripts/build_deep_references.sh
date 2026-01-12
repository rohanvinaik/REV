#!/bin/bash

# Build Deep Behavioral References for REV System
# This generates comprehensive references with full topology extraction

echo "====================================================================="
echo "REV DEEP REFERENCE LIBRARY BUILDER"
echo "====================================================================="
echo "This will build deep behavioral references for all model families"
echo "Expected time: 6-24 hours per family"
echo "====================================================================="

# Function to build deep reference
build_deep_reference() {
    local MODEL_PATH=$1
    local FAMILY=$2
    local OUTPUT_NAME=$3
    local CHALLENGES=${4:-100}
    
    echo ""
    echo "Building deep reference for $FAMILY family..."
    echo "Model: $MODEL_PATH"
    echo "Challenges: $CHALLENGES"
    echo "Output: outputs/${OUTPUT_NAME}_deep_reference.json"
    
    python run_rev.py "$MODEL_PATH" \
        --enable-prompt-orchestration \
        --challenges "$CHALLENGES" \
        --build-reference \
        --claimed-family "$FAMILY" \
        --add-to-library \
        --enable-pot \
        --enable-kdf \
        --enable-evolutionary \
        --enable-dynamic \
        --enable-hierarchical \
        --comprehensive-analysis \
        --save-analysis-report \
        --output "outputs/${OUTPUT_NAME}_deep_reference.json" \
        2>&1 | tee "${OUTPUT_NAME}_deep_build.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully built deep reference for $FAMILY"
    else
        echo "❌ Failed to build deep reference for $FAMILY"
    fi
}

# Build references for each family
echo ""
echo "Starting deep reference builds..."

# Pythia family (smallest, fastest)
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42" \
    "pythia" \
    "pythia_70m" \
    100

# GPT family
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/distilgpt2" \
    "gpt" \
    "distilgpt2" \
    100

# Phi family
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/phi-2" \
    "phi" \
    "phi2" \
    100

# Mistral family
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/mistral_for_colab" \
    "mistral" \
    "mistral" \
    100

# Llama family (larger, longer time)
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/llama-2-7b-hf" \
    "llama" \
    "llama_2_7b" \
    150

# Yi family (very large)
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/yi-34b" \
    "yi" \
    "yi_34b" \
    150

# Falcon family (very large)
build_deep_reference \
    "/Users/rohanvinaik/LLM_models/falcon-40b" \
    "falcon" \
    "falcon_40b" \
    150

echo ""
echo "====================================================================="
echo "DEEP REFERENCE BUILD COMPLETE"
echo "====================================================================="
echo "Check outputs/ directory for deep reference files"
echo "Check log files for detailed processing information"
echo ""
echo "Next steps:"
echo "1. Verify reference library: cat fingerprint_library/reference_library.json"
echo "2. Test with large model: python run_rev.py /path/to/large_model --challenges 100"
echo "3. Monitor speedup (should see 15-20x improvement)"
echo "====================================================================="