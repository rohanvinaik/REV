# REV - Restriction Enzyme Verification System v3.0

## ⚠️ CRITICAL: ONE UNIFIED PIPELINE

**`run_rev.py` is the ONLY entry point** - All features integrated, no other pipeline scripts

## 🚀 QUICK START

### Two Execution Modes

1. **LOCAL FILESYSTEM MODELS** (on disk)
   - Segmented streaming (one layer at a time)
   - 2GB memory cap per process (default)
   - **NEW: Parallel processing up to 36GB total memory**
   - NO API keys needed

2. **CLOUD API MODELS**
   - External API calls
   - Requires API keys

### ⚠️ IMPORTANT: API Mode During Development

**Current Development Setup:** The system shows "API-Only mode" in the logs, but this does NOT mean it's making external API calls. During experimentation and development, the full API pipeline infrastructure is set up but routed to the local filesystem instead of external APIs. This allows testing the complete API workflow without the cost/latency of actual API calls to large models like 405B parameter models.

When development is complete, the routing will be switched back to external APIs. The "API mode" designation refers to the **pipeline architecture**, not the actual data source.

### ✅ CORRECT USAGE - Local Models

```bash
# Use FULL PATH to directory containing config.json

# HuggingFace cache format - USE SNAPSHOT PATH
python run_rev.py /Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42

# Standard format
python run_rev.py /Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct
python run_rev.py /Users/rohanvinaik/LLM_models/yi-34b  # 68GB model on 64GB system WORKS!

# Multi-model comparison
python run_rev.py /path/to/model1 /path/to/model2 --challenges 10

# With prompt orchestration (7 systems)
python run_rev.py /path/to/model --enable-prompt-orchestration --challenges 20
```

### Finding Model Paths

```bash
# Find HuggingFace cache models
find ~/LLM_models -name "config.json" | grep pythia

# Example output:
# /Users/.../models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42/config.json
# Use the directory containing config.json!
```

### ❌ COMMON MISTAKES

```bash
# WRONG: Model ID instead of path
python run_rev.py EleutherAI/pythia-70m  # Tries API

# WRONG: --local flag (REMOVED)
python run_rev.py /path --local  # ERROR

# WRONG: Missing snapshot path
python run_rev.py /Users/.../models--EleutherAI--pythia-70m  # Need snapshot

# CORRECT: Full path with snapshot
python run_rev.py /Users/.../pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
```

## 📚 KEY WORKFLOWS

### Building Reference Library (One-Time Setup)

```bash
# Build deep behavioral reference for model family (6-24 hours)
# Use smallest model in family

# Pythia family
python run_rev.py /Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/xxx \
    --enable-prompt-orchestration --challenges 20

# Llama family  
python run_rev.py /Users/rohanvinaik/LLM_models/llama-2-7b \
    --enable-prompt-orchestration --challenges 20
```

### Using References for Large Models (15-20x Faster)

```bash
# After reference exists, large models run much faster

# Pythia-12B (uses pythia-70m reference)
python run_rev.py /Users/rohanvinaik/LLM_models/pythia-12b --challenges 50

# Llama-70B (uses llama-7b reference)  
python run_rev.py /Users/rohanvinaik/LLM_models/llama-3.3-70b-instruct --challenges 100

# Yi-34B (auto-detects family reference)
python run_rev.py /Users/rohanvinaik/LLM_models/yi-34b --challenges 100
```

### Advanced Options

```bash
# Full orchestration with all 7 prompt systems
python run_rev.py /path/to/model \
    --enable-prompt-orchestration \
    --enable-pot --enable-kdf --enable-evolutionary \
    --enable-dynamic --enable-hierarchical \
    --challenges 100 --debug

# Adversarial testing
python run_rev.py /path/to/model \
    --adversarial --adversarial-ratio 0.5 \
    --adversarial-types jailbreak alignment_faking \
    --challenges 50

# Multi-stage orchestration with time budget
python run_rev.py /path/to/model \
    --orchestrate --time-budget 2.5 \
    --claimed-family llama --add-to-library

# Unified fingerprinting
python run_rev.py /path/to/model \
    --unified-fingerprints --fingerprint-dimension 100000 \
    --fingerprint-sparsity 0.001 --save-fingerprints
```

### 🆕 Parallel Processing (36GB Memory Limit)

```bash
# Process multiple models in parallel
python run_rev.py model1/ model2/ model3/ \
    --parallel --parallel-memory-limit 36.0 \
    --challenges 50

# Process many prompts on single model (batch processing)
python run_rev.py /path/to/model \
    --parallel --parallel-batch-size 10 \
    --parallel-memory-limit 36.0 \
    --challenges 100

# Adaptive parallel processing (adjusts to system load)
python run_rev.py model1/ model2/ \
    --parallel --enable-adaptive-parallel \
    --parallel-memory-limit 36.0

# Different parallel modes
python run_rev.py model1/ model2/ model3/ \
    --parallel --parallel-mode cross_product  # Each model × all prompts
    
python run_rev.py model1/ model2/ model3/ \
    --parallel --parallel-mode broadcast      # All models × same prompts
    
python run_rev.py model1/ model2/ model3/ \
    --parallel --parallel-mode paired         # model[i] × prompt[i]
```

## 🎯 PROMPT ORCHESTRATION

### Seven Specialized Systems

1. **PoT** - Behavioral probes for restriction sites (30%)
2. **KDF** - Security/adversarial testing (20%) 
3. **Evolutionary** - Genetic optimization (20%)
4. **Dynamic** - Template-based synthesis (20%)
5. **Hierarchical** - Taxonomical exploration (10%)
6. **Response Predictor** - Effectiveness prediction
7. **Behavior Profiler** - Pattern analysis

### Usage

```bash
# Enable all (recommended)
python run_rev.py <model> --enable-prompt-orchestration --challenges 100

# Specific systems
python run_rev.py <model> \
    --enable-pot --enable-kdf --enable-evolutionary \
    --enable-dynamic --enable-hierarchical \
    --prompt-analytics --challenges 100

# Strategies
python run_rev.py <model> --enable-prompt-orchestration \
    --prompt-strategy [balanced|adversarial|behavioral|comprehensive]
```

## 🔬 DEEP BEHAVIORAL ANALYSIS

Profiles ALL layers to extract:
- **Restriction Sites**: High-divergence boundaries
- **Stable Regions**: Parallelization opportunities  
- **Behavioral Phases**: Architecture stages
- **Optimization Hints**: Critical layers, memory requirements

### Triggers Automatically When:
- Unknown model (confidence < 0.5)
- `--build-reference` flag
- `--profiler` flag

### Performance Impact
- **Small Model (70M-125M)**: 15-20 minutes for complete reference
- **Medium Model (7B)**: 6-24h once → Complete reference  
- **Large Model (405B)**: 37h → 2h using reference (18.5x speedup!)

### Reference Library Build Settings (CRITICAL)
When running `--build-reference`, the system:
1. **Automatically generates 400+ behavioral probes** (NOT manually selected)
2. **Profiles ALL layers** comprehensively (6 for pythia-70m, 12 for GPT-2, etc.)
3. **Takes appropriate time** (18 min for pythia-70m, 41 min for GPT-2)
4. **Uses cryptographic challenge generation** via PoTChallengeGenerator

#### Tested Reference Builds:
```bash
# Pythia-70m (6 layers, 388 probes, ~18 minutes)
python run_rev.py /Users/rohanvinaik/LLM_models/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42 --build-reference

# GPT-2 (12 layers, 406 probes, ~41 minutes)  
python run_rev.py /Users/rohanvinaik/LLM_models/gpt2 --build-reference

# DistilGPT2 (6 layers, should use 400+ probes NOT 20)
python run_rev.py /Users/rohanvinaik/LLM_models/distilgpt2 --build-reference
```

⚠️ **IMPORTANT**: The system should NEVER use only 20 hardcoded probes. If analysis completes in <5 minutes for reference building, something is wrong!

## 🏗️ ARCHITECTURE

### Core Components
- **REVUnified** (run_rev.py) - Main orchestrator
- **REVPipeline** (src/rev_pipeline.py) - Segmented execution engine
- **MetalAcceleratedInference** - Apple Silicon GPU support
- **SegmentRunner** - Layer-by-layer execution
- **UnifiedInferenceManager** - Model loading coordinator

### Key Features
- Memory-bounded execution (2-4GB for 70B+ models)
- Hyperdimensional behavioral fingerprinting  
- Merkle tree computation verification
- Dual library system (Reference + Active)
- Multi-stage orchestrated testing

## 📂 PROJECT STRUCTURE

```
src/
├── core/           # SPRT statistical testing
├── executor/       # Memory-bounded execution
├── hdc/           # Hyperdimensional computing
├── hypervector/   # Vector operations
├── fingerprint/   # Model fingerprinting
├── verifier/      # Model verification
├── privacy/       # ZK proofs, homomorphic ops
└── orchestration/ # Prompt orchestration
```

## 🧬 DUAL LIBRARY SYSTEM

### Reference Library
- **Location**: `fingerprint_library/reference_library.json`
- **Purpose**: Baseline fingerprints with deep behavioral topology
- **Updates**: Rarely, only for new model families

### Active Library  
- **Location**: `fingerprint_library/active_library.json`
- **Purpose**: Continuously updated with all model runs
- **Updates**: Automatic after each successful run

## 📊 TESTING STRATEGIES

| Architecture | Confidence | Strategy | Time | Focus |
|-------------|------------|----------|------|-------|
| Known (Llama 70B) | >85% | Targeted | 2h | Layers 15,35,55 |
| Variant | 60-85% | Adaptive | 3h | Every 8th layer |
| Novel | <60% | Exploratory | 4h | Every 5th layer |

## 🔬 VALIDATION SUITE

### Running Full Validation
```bash
# Run complete validation suite with all experiments
python run_rev.py --run-validation --generate-validation-plots

# Run specific experiments only
python run_rev.py --run-validation --validation-experiments empirical adversarial

# Custom output directory
python run_rev.py --run-validation --validation-output my_results/

# Specify model families to test
python run_rev.py --run-validation --validation-families gpt llama mistral yi

# Adjust sample size for faster/more thorough testing
python run_rev.py --run-validation --validation-samples 200
```

### Collecting Validation Data During Normal Runs
```bash
# Collect metrics during normal pipeline execution
python run_rev.py /path/to/model --collect-validation-data --export-validation-data metrics.json

# Multiple models with validation collection
python run_rev.py model1 model2 model3 --collect-validation-data \
    --export-validation-data validation_batch.json
```

### Validation Outputs
- **ROC Curves**: Model family classification performance with AUC scores
- **Stopping Time Histograms**: SPRT efficiency analysis showing 50-70% sample reduction
- **Adversarial Attack Results**: Success rates for 5 attack types (stitching, spoofing, gradient, poisoning, collision)
- **Performance Dashboard**: Comprehensive 8-panel metrics visualization
- **HTML Report**: Combined report with all plots

Results saved to `experiments/results/`:
- `empirical_metrics.json` - Classification metrics
- `adversarial_results.json` - Attack experiment data  
- `stopping_time_report.json` - SPRT analysis
- `complete_validation_results.json` - All results combined
- `validation_summary.json` - High-level summary
- `plots/` - Publication-ready visualizations (300 DPI)
- `report.html` - Combined HTML report

## 🔐 SECURITY FEATURES

### Attestation Server
```bash
# Start attestation server for fingerprint verification
python run_rev.py --attestation-server --attestation-port 8080

# With Trusted Execution Environment (TEE) support
python run_rev.py --attestation-server --enable-tee

# With Hardware Security Module (HSM) for signing
python run_rev.py --attestation-server --enable-hsm

# Full security configuration
python run_rev.py --attestation-server \
    --attestation-port 8443 \
    --enable-tee \
    --enable-hsm \
    --debug
```

Server endpoints:
- `GET /health` - Health check
- `POST /attest/fingerprint` - Create attestation
- `GET /verify/attestation/<id>` - Verify attestation
- `POST /prove/distance` - ZK distance proof
- `POST /prove/range` - Bulletproof range proof
- `POST /register/fingerprint` - Register fingerprint (auth required)
- `GET /audit/log` - Audit log (admin only)

### Zero-Knowledge Proofs
```bash
# Enable ZK proofs for fingerprint comparisons
python run_rev.py /path/to/model --enable-zk-proofs

# Combined with other features
python run_rev.py /path/to/model \
    --enable-zk-proofs \
    --enable-prompt-orchestration \
    --challenges 50
```

ZK proof types:
- **Distance proofs**: Prove distance between fingerprints without revealing them
- **Range proofs**: Prove similarity score is in range [0,1] using Bulletproofs
- **Membership proofs**: Prove fingerprint is in Merkle tree without revealing it

### Rate Limiting
```bash
# Enable API rate limiting
python run_rev.py /path/to/model --enable-rate-limiting --rate-limit 20.0

# Hierarchical rate limiting (user → model → global)
python run_rev.py /path/to/model \
    --enable-rate-limiting \
    --rate-limit 10.0
```

Rate limiting features:
- Token bucket algorithm with configurable refill rate
- Exponential backoff with jitter for repeated failures
- Per-model and per-user quota management
- Redis backend support for distributed systems
- Adaptive rate limiting based on system load

### Complete Security Setup
```bash
# Enable all security features
python run_rev.py /path/to/model \
    --enable-security \
    --enable-zk-proofs \
    --enable-rate-limiting \
    --rate-limit 15.0 \
    --enable-hsm

# With attestation server running separately
# Terminal 1:
python run_rev.py --attestation-server --enable-tee --enable-hsm

# Terminal 2:
python run_rev.py /path/to/model --enable-security --enable-zk-proofs
```

### Security Testing
```bash
# Run security test suite
pytest tests/test_security.py -v

# Specific test categories
pytest tests/test_security.py::TestZKAttestation -v
pytest tests/test_security.py::TestRateLimiter -v
pytest tests/test_security.py::TestMerkleTrees -v
pytest tests/test_security.py::TestAttestationServer -v
```

### Performance Targets
- **ZK Proof Generation**: < 200ms per proof ✅
- **Merkle Proof Verification**: < 10ms ✅
- **Rate Limiting Check**: < 1ms ✅
- **Batch Verification**: 10-20% speedup ✅
- **Attestation Creation**: < 50ms ✅

## 🧬 PRINCIPLED FEATURE EXTRACTION

### Overview
REV now includes a principled feature extraction system that replaces hand-picked features with automatically discovered, interpretable features across four hierarchical categories.

### Feature Categories

1. **Syntactic Features** (9 features)
   - Token distributions and type-token ratios
   - Zipf distribution parameters
   - N-gram entropy (1-3 grams)
   - Lexical complexity metrics
   - Sentence structure analysis

2. **Semantic Features** (20 features)
   - Embedding space statistics (mean, std, skew, kurtosis)
   - Cosine similarity distributions
   - Principal component analysis (top 10 components)
   - Attention entropy and focus patterns

3. **Behavioral Features** (9 features)
   - Response consistency metrics
   - Uncertainty quantification (entropy, confidence)
   - Refusal behavior analysis
   - Sentiment indicators
   - Temperature-like diversity estimates

4. **Architectural Features** (18 features)
   - Layer-wise activation statistics
   - Gradient flow patterns (vanishing/exploding detection)
   - Sparsity analysis across layers
   - Model capacity indicators
   - Transformer-specific features (heads, dimensions)

### Running with Principled Features

```bash
# Basic usage with principled feature extraction
python run_rev.py /path/to/model --enable-principled-features --challenges 50

# Full configuration with all options
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-selection-method ensemble \
    --feature-reduction-method umap \
    --num-features-select 100 \
    --enable-learned-features \
    --feature-analysis-report

# Compare multiple models with feature analysis
python run_rev.py model1 model2 model3 \
    --enable-principled-features \
    --feature-analysis-report \
    --challenges 100

# With prompt orchestration and principled features
python run_rev.py /path/to/model \
    --enable-prompt-orchestration \
    --enable-principled-features \
    --enable-learned-features \
    --challenges 200
```

### Feature Selection Methods

```bash
# Mutual information (best for classification)
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-selection-method mutual_info

# LASSO (sparse linear features)
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-selection-method lasso

# Elastic Net (balanced L1/L2)
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-selection-method elastic_net

# Ensemble (combines all methods - recommended)
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-selection-method ensemble
```

### Dimensionality Reduction Options

```bash
# PCA (linear, preserves variance)
--feature-reduction-method pca

# t-SNE (non-linear, 2D visualization)
--feature-reduction-method tsne

# UMAP (non-linear, preserves structure - recommended)
--feature-reduction-method umap

# No reduction (use all selected features)
--feature-reduction-method none
```

### Learned Features

Enable contrastive learning and autoencoders for adaptive feature discovery:

```bash
# Enable all learning methods
python run_rev.py /path/to/model \
    --enable-principled-features \
    --enable-learned-features \
    --challenges 100

# Learned features improve with more models
python run_rev.py model1 model2 model3 model4 model5 \
    --enable-principled-features \
    --enable-learned-features
```

### Feature Analysis Report

Generate comprehensive analysis with visualizations:

```bash
# Generate full analysis report
python run_rev.py /path/to/model \
    --enable-principled-features \
    --feature-analysis-report

# Output includes:
# - Feature correlation matrices with clustering
# - Feature importance rankings across methods
# - Ablation study results
# - Model family-specific feature distributions
# - LaTeX report for publication
# - All saved to: experiments/feature_analysis_results/
```

### Integration with Existing Features

The principled features seamlessly integrate with:
- **HDC Encoding**: Features weighted by importance before hypervector encoding
- **Prompt Orchestration**: Enhanced behavioral features from diverse prompts
- **Deep Behavioral Analysis**: Architectural features from layer profiling
- **Unified Fingerprints**: Principled features included in fingerprint data

### Performance Impact

- **Feature Extraction**: ~100ms per model
- **Feature Selection**: ~500ms for 100 features
- **Learned Features**: ~2s for contrastive learning (improves over time)
- **Analysis Report**: ~10s for full visualization suite

### Advanced Usage Examples

```bash
# Reference library building with principled features
python run_rev.py /path/to/small_model \
    --build-reference \
    --enable-principled-features \
    --enable-learned-features \
    --feature-analysis-report

# Adversarial testing with feature analysis
python run_rev.py /path/to/model \
    --adversarial \
    --enable-principled-features \
    --feature-selection-method mutual_info \
    --challenges 50

# Multi-model comparison with feature importance
python run_rev.py gpt-model llama-model mistral-model \
    --enable-principled-features \
    --feature-analysis-report \
    --output comparison_report.json

# Full pipeline with all advanced features
python run_rev.py /path/to/model \
    --enable-prompt-orchestration \
    --enable-principled-features \
    --enable-learned-features \
    --unified-fingerprints \
    --comprehensive-analysis \
    --feature-analysis-report \
    --challenges 500 \
    --debug
```

### Feature Data Storage

Principled features are stored in the pipeline results:

```json
{
  "stages": {
    "behavioral_analysis": {
      "metrics": {
        "principled_features": {
          "syntactic": [...],
          "semantic": [...],
          "behavioral": [...],
          "architectural": [...],
          "feature_importance": [
            ["response_consistency", 0.92],
            ["embedding_mean", 0.87],
            ["attention_entropy", 0.84],
            ...
          ],
          "learned_features": [...]
        }
      }
    }
  }
}
```

### Interpreting Results

The feature analysis report provides:

1. **Top Important Features**: Ranked list of most discriminative features
2. **Feature Correlations**: Identify redundant or complementary features
3. **Ablation Results**: Impact of each feature category on performance
4. **Family Distributions**: How features vary across model families
5. **LaTeX Report**: Publication-ready analysis documentation

## 🔧 DEVELOPMENT

### Running Tests
```bash
make install-dev
make test                # All tests
make test-unit          # Unit tests
make test-integration   # Integration
make test-performance   # Benchmarks
make test-coverage      # Coverage report
```

### Performance Targets Met
- ✅ Hamming distance: <1ms for 10K dimensions
- ✅ HDC encoding: <50ms per 100K-dim sample
- ✅ Sequential test: <100ms for 1000 samples
- ✅ ZK proof generation: <300ms
- ✅ Feature extraction: <100ms per model
- ✅ Feature selection: <500ms for 100 features

### Code Guidelines
- Type hints for all functions
- Docstrings with Args/Returns
- Black formatting (line length: 100)
- Update CLAUDE.md with changes

## 💡 KEY INNOVATION

REV enables verification of massive LLMs (68GB Yi-34B, 405B models) that EXCEED available memory through intelligent segmented execution. Models are processed layer-by-layer at restriction sites (attention boundaries) while maintaining cryptographic verification through Merkle trees.

---
*Repository: https://github.com/rohanvinaik/REV*