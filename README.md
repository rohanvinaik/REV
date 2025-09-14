# REV: Restriction Enzyme Verification Framework

**Pure API-Based LLM Behavioral Fingerprinting & Family Recognition**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![API-Only](https://img.shields.io/badge/Mode-API--Only-blue.svg)]()

## 🚀 Overview

REV enables model family recognition and behavioral fingerprinting through pure API access - no weights, downloads, or local compute required.

### Key Capabilities
- Process 644GB models with <1GB RAM
- Recognize families across 58x size differences (7B → 405B)
- Compare proprietary models behind APIs
- Build reference libraries automatically
- Zero model downloads required

### Validated Results
Successfully identified Llama-2-7B and Llama-3.1-405B as same family with 97% confidence using <700MB RAM.

## 🌟 Core Technology

### Pattern-Based Identification (v3.2)
- **60%** variance delta patterns - behavioral change magnitudes
- **25%** relative positions - normalized for different model depths
- **15%** adaptive features - smart interpolation & confidence-based expansion

### Prompt Orchestration System
- **PoT (Proof-of-Training)**: Cryptographically pre-committed challenges for identity verification
- **KDF Adversarial**: Security boundary testing via jailbreak/alignment probes
- **Evolutionary**: Genetic optimization for discriminative prompts
- **Dynamic Synthesis**: Real-time template combination
- **Hierarchical Taxonomy**: Systematic capability exploration

### Behavioral Metrics
10,000-dimensional hypervector fingerprints capturing:
- Response diversity & entropy patterns
- Sparsity distributions
- Architectural signatures
- Training-specific behaviors

## 🚀 Quick Start

```bash
git clone https://github.com/rohanvinaik/REV.git
cd REV
pip install -r requirements.txt
```

## 📋 Usage

### Local Models (Filesystem)
```bash
# Standard format
python run_rev.py /path/to/llama-3.3-70b-instruct

# HuggingFace cache (use snapshot path!)
python run_rev.py /Users/.../pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42

# With orchestration
python run_rev.py /path/to/model --enable-prompt-orchestration --challenges 20
```

### Cloud APIs
```bash
# OpenAI/Anthropic
python run_rev.py gpt-4 --api-key sk-...
python run_rev.py claude-3-opus --provider anthropic --api-key ...
```

### Common Mistakes
```bash
# ❌ WRONG: Model ID instead of path
python run_rev.py EleutherAI/pythia-70m  # Tries API!

# ❌ WRONG: Missing snapshot path
python run_rev.py /path/models--EleutherAI--pythia-70m  # Incomplete!

# ✅ CORRECT: Full path with snapshot
python run_rev.py /path/models--EleutherAI--pythia-70m/snapshots/xxx
```

## 🎯 Applications

1. **Duplicate Detection**: Identify rebranded/modified models
2. **Architecture Analysis**: Understand model lineage
3. **Security Verification**: Detect trojaned variants
4. **API Authentication**: Verify claimed model families

## 📊 Performance

| Model Size | Memory Usage | Processing Time | Confidence |
|------------|--------------|-----------------|------------|
| 7B | <500MB | 15 min | 95%+ |
| 70B | <1GB | 2-3 hrs | 92%+ |
| 405B | <1GB | 3-4 hrs | 90%+ |

## 🔧 Advanced Features

### Reference Library Building
```bash
# Build once with small model
python run_rev.py /path/to/pythia-70m --build-reference --enable-prompt-orchestration

# Accelerates all future models in family (15-20x speedup)
python run_rev.py /path/to/pythia-12b --challenges 100
```

### Security Features
- Zero-knowledge proofs for verification
- Rate limiting & attestation server
- Merkle tree computation verification
- TEE/HSM support

### Parallel Processing
```bash
# Process multiple models (up to 36GB total memory)
python run_rev.py model1/ model2/ model3/ --parallel --parallel-memory-limit 36.0
```

## 📚 Documentation

- **User Guide**: See [CLAUDE.md](CLAUDE.md) for detailed instructions
- **API Reference**: Available in `docs/api/`
- **Paper**: [arXiv:2024.xxxxx](https://arxiv.org)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on research from:
- Proof-of-Training (PoT) framework
- Hyperdimensional computing
- Sequential hypothesis testing (SPRT)
- Zero-knowledge cryptography

---
*Repository: https://github.com/rohanvinaik/REV*