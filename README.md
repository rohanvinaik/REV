# REV - Restriction Enzyme Verification

A framework for memory-bounded, black-box LLM comparison using restriction enzyme verification techniques combined with hyperdimensional computing and privacy-preserving infrastructure.

## Overview

REV (Restriction Enzyme Verification) provides a comprehensive framework for comparing and verifying Large Language Models (LLMs) under memory-bounded constraints. The system combines cutting-edge techniques from statistical testing, hyperdimensional computing, and cryptography to enable efficient and privacy-preserving model verification.

## Key Features

### 🧬 Sequential Testing Framework
- **Anytime-valid statistical tests** with Empirical-Bernstein bounds
- **SPRT (Sequential Probability Ratio Test)** implementation with numerical stability
- **Welford's algorithm** for online mean/variance computation
- **Early stopping** with configurable error bounds (α, β)

### 🎯 Challenge Generation
- **Deterministic prompt generation** with seeded synthesis
- **Reproducible probes** for consistent model evaluation
- **Domain-specific challenge templates** with variable difficulty
- **Commit-reveal protocols** for verification integrity

### 🔬 Model Verification Pipeline
- **Enhanced sequential testing** with streaming execution
- **Verdict system** with confidence intervals
- **Adaptive sampling** based on statistical power
- **Resource-bounded verification** for memory efficiency

### 🧠 Hyperdimensional Computing Core
- **8K-100K dimensional vectors** for model signatures
- **Sparse and dense encoding** strategies
- **Orthogonal random projections** for dimensionality preservation
- **Numerical stability** with fp32/fp64 precision control

### ⚡ Hamming Distance LUTs
- **16-bit lookup tables** for 10-20× speedup
- **Batch processing** with vectorized operations
- **CPU/GPU acceleration** support
- **Memory-efficient packed binary representations**

### 🔒 Privacy-Preserving Infrastructure
- **Zero-knowledge proofs** using Halo2-compatible circuits
- **Differential privacy** mechanisms with calibrated noise
- **Merkle tree commitments** for verifiable computation
- **No trusted setup** required for ZK verification

## Installation

### Requirements
- Python 3.8+
- NumPy, PyTorch, Transformers
- Cryptography libraries (pycryptodome)
- Optional: CUDA for GPU acceleration

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/REV.git
cd REV

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Verification

```python
from rev import sequential_verify, DeterministicPromptGenerator

# Generate challenges
generator = DeterministicPromptGenerator(seed=42)
challenges = generator.generate_batch(n=100)

# Run sequential verification
result = sequential_verify(
    model_a=model_a,
    model_b=model_b,
    challenges=challenges,
    alpha=0.05,
    beta=0.10
)

print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence:.3f}")
```

### Hyperdimensional Encoding

```python
from rev import HypervectorEncoder, HypervectorConfig

# Configure encoder
config = HypervectorConfig(
    dimension=10000,
    sparse_density=0.01,
    dtype="float32"
)

# Encode model outputs
encoder = HypervectorEncoder(config)
signature = encoder.encode_sequence(model_outputs)
```

### Privacy-Preserving Verification

```python
from rev import ZKProofSystem, DifferentialPrivacyMechanism

# Setup zero-knowledge proof
zk_system = ZKProofSystem()
proof = zk_system.prove_verification(result)

# Add differential privacy
dp_mechanism = DifferentialPrivacyMechanism(epsilon=1.0)
private_result = dp_mechanism.privatize(result)
```

## Architecture

```
src/
├── core/                 # Core statistical testing
│   ├── sequential.py     # SPRT and sequential testing
│   ├── boundaries.py     # Empirical-Bernstein bounds
│   └── stats.py          # Statistical utilities
├── challenges/           # Challenge generation
│   ├── prompt_generator.py
│   └── templates.py
├── verifier/            # Model verification
│   ├── decision.py
│   └── pipeline.py
├── hdc/                 # Hyperdimensional computing
│   ├── encoder.py
│   └── projections.py
├── hypervector/         # Hypervector operations
│   ├── hamming.py       # Hamming distance
│   ├── operations/      # Vector operations
│   └── packed.py        # Packed representations
├── crypto/              # Cryptographic primitives
│   ├── zk_proofs.py     # Zero-knowledge proofs
│   ├── merkle.py        # Merkle trees
│   └── commit.py        # Commitment schemes
└── privacy/             # Privacy mechanisms
    └── differential_privacy.py
```

## Performance

### Benchmarks

| Component | Operation | Performance |
|-----------|-----------|-------------|
| Hamming LUT | 10K-dim distance | 10-20× faster than naive |
| HDC Encoder | 100K-dim encoding | ~50ms per sample |
| Sequential Test | 1000 samples | <100ms decision |
| ZK Proof | Generation | ~200ms |
| Merkle Tree | 1M leaves | ~1s construction |

### Memory Usage

- **Hypervectors**: 40KB for 10K-dim float32
- **Hamming LUT**: 512KB for 16-bit table
- **ZK Circuits**: ~10MB compiled
- **Merkle Proofs**: O(log n) storage

## Advanced Usage

### Custom Challenge Templates

```python
from rev.challenges import ChallengeTemplate

template = ChallengeTemplate(
    pattern="Complete the sequence: {prefix}",
    difficulty="hard",
    domain="mathematical"
)

challenges = generator.generate_from_template(template, n=50)
```

### Streaming Verification

```python
from rev.verifier import StreamingVerifier

verifier = StreamingVerifier(buffer_size=32)
for batch in data_stream:
    decision = verifier.update(batch)
    if decision.is_conclusive:
        break
```

### GPU Acceleration

```python
from rev.hypervector import hamming_distance_gpu

# Compute on GPU if available
import torch
if torch.cuda.is_available():
    distance = hamming_distance_gpu(vec_a, vec_b)
```

## Components Origin

This framework integrates battle-tested components from:

- **PoT_Experiments**: Sequential testing, challenge generation, and model verification pipeline
- **GenomeVault**: Hyperdimensional computing, Hamming distance LUTs, and privacy-preserving infrastructure

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Citation

If you use REV in your research, please cite:

```bibtex
@software{rev2024,
  title = {REV: Restriction Enzyme Verification for LLMs},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/REV}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sequential testing algorithms adapted from PoT_Experiments
- Hyperdimensional computing core from GenomeVault
- Empirical-Bernstein bounds implementation based on statistical literature

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.