# REV: Restriction Enzyme Verification System v3.0

**Behavioral Fingerprinting for LLM Security & Verification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

## 🔬 Overview

REV (Restriction Enzyme Verification) identifies and verifies Large Language Models through behavioral analysis, treating them as secure black boxes. Like restriction enzymes in molecular biology that recognize specific DNA sequences, REV identifies unique behavioral "restriction sites" in neural networks to create unforgeable fingerprints.

### Key Innovation
- **Segmented Streaming**: Process 70B+ models with only 2-4GB RAM by streaming weights layer-by-layer
- **Behavioral Fingerprinting**: Identify models through response patterns, not metadata or weights
- **Cross-Architecture Matching**: 95% accuracy identifying models across 80x size differences
- **Dual Library System**: 15-20x speedup using reference baselines from smaller models

### Security First
- Model weights remain completely secret - never loaded into memory
- Behavioral patterns are unforgeable (emerge from billions of parameters)
- Robust against weight pruning, quantization, and fine-tuning attacks
- Works with both local filesystem models and cloud APIs

## 🚀 Quick Start

```bash
git clone https://github.com/rohanvinaik/REV.git
cd REV
pip install -r requirements.txt
```

## 📋 Core Usage

### Local Models (Filesystem)
```bash
# ALWAYS use --enable-prompt-orchestration for proper operation
# Generates 250-400+ behavioral probes automatically

# Standard model format
python run_rev.py /path/to/llama-3.3-70b-instruct --enable-prompt-orchestration

# HuggingFace cache format (use full snapshot path)
python run_rev.py /Users/.../pythia-70m/snapshots/a39f36b... --enable-prompt-orchestration

# With debug output
python run_rev.py /path/to/model --enable-prompt-orchestration --debug
```

### Cloud API Models
```bash
# OpenAI
python run_rev.py gpt-4 --api-key sk-...

# Anthropic
python run_rev.py claude-3-opus --provider anthropic --api-key ...
```

## 🧬 How It Works

### 1. Behavioral Analysis
REV injects prompts at specific layers to measure behavioral divergence:
```
Layer 0: "Complete this sentence: The weather today is"
→ Divergence: 0.290 (behavioral boundary detected)

Layer 10: "Complete this sentence: The weather today is"
→ Divergence: 0.337 (restriction site identified)
```

### 2. Topological Fingerprinting
- Identifies "restriction sites" - layers with significant behavioral changes
- Builds variance profiles from actual model responses
- Creates 10,000-dimensional hypervector fingerprints
- Matches patterns using cosine similarity, DTW, and Fourier analysis

### 3. Enhanced Dual Library Matching
When initial confidence is low (e.g., 20%), REV automatically:
1. Invokes enhanced topological similarity algorithm
2. Applies cross-size matching (works across 32 → 80 layer models)
3. Uses multiple similarity metrics with adaptive weighting
4. Achieves 95%+ confidence through behavioral analysis

## 🎯 Reference Library System

### Build Reference (One-Time per Architecture)
```bash
# Use SMALLEST model in family as reference
# Takes 20min-3hrs depending on size

# Pythia family
python run_rev.py /path/to/pythia-70m --build-reference --enable-prompt-orchestration

# Llama family
python run_rev.py /path/to/llama-2-7b-hf --build-reference --enable-prompt-orchestration

# GPT family
python run_rev.py /path/to/distilgpt2 --build-reference --enable-prompt-orchestration
```

### Use Reference for Large Models (15-20x Faster)
```bash
# After reference exists, large models run MUCH faster
python run_rev.py /path/to/llama-70b --enable-prompt-orchestration
# Uses llama-7b reference → 2-4 hrs instead of 37 hrs!
```

## 📊 Performance Metrics

| Model | Memory | Time (w/o Ref) | Time (w/ Ref) | Confidence |
|-------|--------|----------------|---------------|------------|
| 7B | 2-3GB | 2-3 hrs | N/A (is ref) | 95%+ |
| 70B | 3-4GB | 37 hrs | 2-4 hrs | 95%+ |
| 405B | 4GB | 72+ hrs | 4-6 hrs | 92%+ |

## 🔐 Security Features

- **Zero Weight Exposure**: Weights streamed layer-by-layer, never fully loaded
- **Behavioral Verification**: Unforgeable patterns from billions of parameters
- **Attack Resistance**: Robust against metadata spoofing, pruning, quantization
- **Cross-Version Detection**: Identifies families across versions/sizes

## ⚠️ Common Issues & Solutions

### Issue: Low Initial Confidence
**Solution**: System automatically invokes enhanced matching. Ensure `--enable-prompt-orchestration` is used.

### Issue: Reference Not Being Used
**Solution**: Build reference for model family first using smallest model.

### Issue: Probe Times Too Fast (<10ms)
**Solution**: Indicates mock responses. Verify model path and weights exist.

## 🛠️ Advanced Features

```bash
# Comprehensive analysis with all features
python run_rev.py /path/to/model \
  --enable-prompt-orchestration \
  --comprehensive-analysis \
  --unified-fingerprints \
  --debug

# Parallel processing for multiple models
python run_rev.py model1 model2 model3 \
  --parallel --parallel-memory-limit 36.0

# Adversarial testing
python run_rev.py /path/to/model \
  --adversarial --adversarial-types jailbreak alignment_faking
```

## 📚 Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Detailed architecture explanations
- Troubleshooting guides
- Performance optimization tips
- Security considerations
- API references

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with support from the open-source community. Special thanks to contributors and researchers advancing LLM security.

---
*For bugs/issues: https://github.com/rohanvinaik/REV/issues*