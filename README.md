# REV: Restriction Enzyme Verification System v3.0

**Behavioral Fingerprinting for LLM Security & Verification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

## 🔬 Overview

REV (Restriction Enzyme Verification) identifies and verifies Large Language Models through deep behavioral analysis, treating them as secure black boxes. Like restriction enzymes in molecular biology that recognize specific DNA sequences, REV identifies unique behavioral "restriction sites" in neural networks where significant behavioral changes occur.

### Key Innovations
- **Segmented Streaming**: Process 70B+ models with only 2-4GB RAM by streaming weights layer-by-layer
- **Dual Library System**: Reference library (deep topology) + Active library (runtime fingerprints)
- **Prompt Orchestration**: 7 specialized systems generate 250-400+ targeted behavioral probes
- **Enhanced Matching**: 95% accuracy across 80x size differences using topological similarity

### Security First
- Model weights remain completely secret - never fully loaded into memory
- Behavioral patterns are unforgeable (emerge from billions of parameters)
- Robust against metadata spoofing, weight pruning, quantization, and fine-tuning
- Works with both local filesystem models and cloud APIs

## 🚀 Quick Start

```bash
git clone https://github.com/rohanvinaik/REV.git
cd REV
pip install -r requirements.txt
```

## 🧬 How REV Actually Works

### Phase 1: Light Behavioral Probe
Quick topology scan (~10-15% of layers) to determine model family:
```
Layer 0:  Divergence: 0.290  ← Initial behavioral baseline
Layer 7:  Divergence: 0.320  ← Attention layer boundary
Layer 14: Divergence: 0.285  ← Stable region
Layer 21: Divergence: 0.314  ← Behavioral shift
...
Layer 79: Divergence: 0.350  ← Output layer characteristics
```

### Phase 2: Enhanced Dual Library Matching
When initial confidence is low (e.g., 20%), automatically invokes:
- **Cosine Similarity** (20%): Shape matching of variance profiles
- **DTW Matching** (10%): Dynamic time warping for pattern alignment
- **Topology Signatures** (10%): Structural feature comparison
- **Fourier Analysis** (5%): Periodic pattern detection
- **Adaptive Thresholds**: Cross-size normalization (32 vs 80 layers)

### Phase 3: Prompt Orchestration System
Generates 250-400+ specialized prompts across 7 systems:

1. **PoT (Proof-of-Training)** - 30% of prompts
   - Cryptographically pre-committed challenges
   - Tests model-specific training artifacts
   - Verifies behavioral consistency

2. **KDF Adversarial** - 20% of prompts
   - Security boundary testing
   - Jailbreak resistance probes
   - Alignment verification

3. **Evolutionary** - 20% of prompts
   - Genetically optimized discriminative prompts
   - Mutation and crossover for prompt diversity
   - Fitness-based selection

4. **Dynamic Synthesis** - 20% of prompts
   - Real-time template combination
   - Context-aware prompt generation
   - Adaptive complexity

5. **Hierarchical Taxonomy** - 10% of prompts
   - Systematic capability exploration
   - Nested behavioral testing
   - Cross-domain verification

6. **Response Predictor** - Effectiveness scoring
7. **Behavior Profiler** - Pattern analysis

### Phase 4: Deep Behavioral Fingerprinting
Creates 10,000-dimensional hypervector fingerprints from:
- Response diversity patterns at each restriction site
- Layer-wise behavioral transitions
- Attention pattern signatures
- Sparsity distributions
- Entropy measurements

## 📚 Dual Library System

### Reference Library
**Purpose**: Deep behavioral topology maps for model families

**Contains**:
- Complete restriction site mappings
- Variance profiles across all layers
- Behavioral phase transitions
- Architecture-specific patterns

**Build Process** (one-time per architecture family):
```bash
# Use SMALLEST model in family as reference
# Generates 400+ comprehensive probes across ALL layers

# Pythia family (use 70M model as reference)
python run_rev.py /path/to/pythia-70m \
  --build-reference --enable-prompt-orchestration

# Llama family (use 7B model as reference)
python run_rev.py /path/to/llama-2-7b-hf \
  --build-reference --enable-prompt-orchestration

# GPT family (use DistilGPT2 as reference)
python run_rev.py /path/to/distilgpt2 \
  --build-reference --enable-prompt-orchestration
```

**Time Investment**:
- Small models (70M-1B): 20-30 minutes
- Medium models (7B): 2-3 hours
- Worth it: Enables 15-20x speedup for all larger models in family

### Active Library
**Purpose**: Runtime fingerprints for all tested models

**Contains**:
- Model fingerprints from regular runs
- Confidence scores and family matches
- Testing timestamps and parameters
- Cross-model comparison data

**Automatic Updates**:
- Every successful run adds to active library
- No manual intervention needed
- Used for cross-model verification

### How Libraries Work Together

```
First Time Testing Llama-70B (no reference):
  1. Light probe → 20% confidence (unknown family)
  2. Enhanced matching → Still uncertain
  3. Deep analysis → 37+ hours of comprehensive probing
  4. Result saved to active library

After Building Llama-7B Reference:
  1. Light probe → 20% confidence initially
  2. Enhanced matching with reference → 95% confidence!
  3. Targeted probing at known restriction sites
  4. Complete in 2-4 hours (15-20x faster!)
```

## 📋 Core Usage

### Standard Testing (with Orchestration)
```bash
# ALWAYS use --enable-prompt-orchestration
# Without it, only 0-7 probes generated (broken!)

# Standard model test
python run_rev.py /path/to/llama-3.3-70b-instruct \
  --enable-prompt-orchestration

# With comprehensive analysis
python run_rev.py /path/to/model \
  --enable-prompt-orchestration \
  --comprehensive-analysis \
  --unified-fingerprints

# With debug output
python run_rev.py /path/to/model \
  --enable-prompt-orchestration \
  --debug
```

### Cloud API Models
```bash
# OpenAI
python run_rev.py gpt-4 --api-key sk-...

# Anthropic
python run_rev.py claude-3-opus --provider anthropic --api-key ...
```

## 📊 Performance Metrics

| Model | Memory | Without Reference | With Reference | Speedup |
|-------|--------|------------------|----------------|---------|
| 7B | 2-3GB | N/A (is reference) | N/A | - |
| 70B | 3-4GB | 37+ hours | 2-4 hours | 15-20x |
| 405B | 4GB | 72+ hours | 4-6 hours | 18x |

## 🎯 Why This Matters

### For Security
- **Verify model identity** without access to weights
- **Detect trojaned/backdoored** models
- **Prove model provenance** cryptographically
- **Identify fine-tuned variants** of base models

### For Efficiency
- **Run 70B models on laptops** with 8GB RAM
- **15-20x faster testing** with reference library
- **Cross-architecture verification** without re-testing

### For Research
- **Understand model families** through behavioral analysis
- **Map architectural evolution** across versions
- **Discover behavioral boundaries** (restriction sites)

## ⚠️ Common Issues & Solutions

### Issue: "Only 7 challenges generated"
**Cause**: Missing `--enable-prompt-orchestration` flag
**Solution**: Always include this flag for proper operation

### Issue: "Model confidence 0.0%"
**Cause**: No reference library for this architecture
**Solution**: Build reference using smallest model in family

### Issue: Probe times <10ms
**Cause**: Mock responses (not real computation)
**Solution**: Verify model path and weights exist

### Issue: Taking 37+ hours for large model
**Cause**: No reference library available
**Solution**: Build reference first (one-time investment)

## 🛠️ Advanced Features

```bash
# Parallel processing for multiple models
python run_rev.py model1 model2 model3 \
  --parallel --parallel-memory-limit 36.0

# Adversarial testing
python run_rev.py /path/to/model \
  --adversarial \
  --adversarial-types jailbreak alignment_faking

# Full validation suite
python run_rev.py /path/to/model \
  --enable-prompt-orchestration \
  --comprehensive-analysis \
  --unified-fingerprints \
  --collect-validation-data
```

## 📚 Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Detailed architecture explanations
- Complete troubleshooting guide
- Prompt orchestration details
- Security considerations
- Development guidelines

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with support from the open-source community. Special thanks to contributors advancing LLM security and verification.

---
*For bugs/issues: https://github.com/rohanvinaik/REV/issues*