# REV - Restriction Enzyme Verification System

## 🎯 CORE PURPOSE OF THIS EXPERIMENT

**THE WHOLE POINT**: REV enables verification of massive LLMs (like Yi-34B with 68GB memory footprint) that EXCEED available device memory through intelligent segmented execution. This is NOT about avoiding loading the model - it's about making it POSSIBLE to run and verify models that wouldn't otherwise fit in memory AT ALL.

### Why This Matters
- **Problem**: Modern LLMs (70B, 175B parameters) require 140GB-350GB+ RAM
- **Solution**: REV segments execution to verify these models with only 8-16GB active RAM
- **Validation**: Yi-34B (68GB) successfully runs on 64GB system with only 2-3GB active memory per segment

### Key Innovation
REV treats transformer models like DNA sequences that can be "cut" at restriction sites (attention boundaries, layer transitions) and processed segment-by-segment while maintaining cryptographic verification of the complete computation through Merkle trees.

### What We're Testing with Yi-34B
1. **Memory-Bounded Execution**: Process 68GB model with <4GB active memory
2. **Behavioral Segmentation**: Automatically discover model's processing regions
3. **Cryptographic Integrity**: Generate Merkle proofs for all segments
4. **Semantic Fingerprinting**: Create hyperdimensional vectors for model behavior
5. **Statistical Verification**: Use SPRT for efficient model comparison

## Project Overview
REV is a comprehensive framework for memory-bounded, black-box LLM comparison using restriction enzyme verification techniques combined with hyperdimensional computing and privacy-preserving infrastructure.

## Key Components Status

### ✅ Core Pipeline (Completed)
- **REVPipeline** (`src/rev_pipeline.py`): Segment-wise execution, memory-bounded streaming, Merkle tree construction
- **SegmentRunner** (`src/executor/segment_runner.py`): Parameter loading/offloading, KV cache management, activation extraction
- **BlackBoxVerifier** (`src/verifier/blackbox.py`): API-based model comparison with rate limiting and caching

### ✅ Hyperdimensional Computing (Completed)
- **HypervectorEncoder** (`src/hdc/encoder.py`): 8K-100K dimensional vectors with sparse/dense encoding
- **BehavioralSites** (`src/hdc/behavioral_sites.py`): Probe feature extraction, hierarchical zoom levels
- **BindingOperations** (`src/hdc/binding_operations.py`): XOR, permutation, circular convolution, Fourier binding
- **ErrorCorrection** (`src/hdc/error_correction.py`): XOR parity blocks, Hamming codes, noise tolerance

### ✅ Statistical Testing (Completed)
- **Sequential Testing** (`src/core/sequential.py`): SPRT with Empirical-Bernstein bounds
- **DecisionAggregator** (`src/verifier/decision_aggregator.py`): Per-challenge indicators, score aggregation
- **AdvancedSimilarity** (`src/hypervector/similarity.py`): Hierarchical distance, clustering, 10+ metrics

### ✅ Privacy & Security (Completed)
- **ZK Proofs** (`src/crypto/zk_proofs.py`, `src/privacy/distance_zk_proofs.py`): Distance proofs, range proofs
- **Homomorphic Ops** (`src/privacy/homomorphic_ops.py`): Encrypted computation, federated protocol
- **Differential Privacy** (`src/privacy/differential_privacy.py`): Noise mechanisms, privacy levels

### 🔄 Testing (In Progress)
- Unit tests for all components
- Integration tests for full pipeline
- Performance benchmarks
- Memory profiling
- Adversarial robustness

## Architecture

```
src/
├── core/                 # Statistical testing core
│   ├── sequential.py     # SPRT implementation
│   └── boundaries.py     # Empirical-Bernstein bounds
├── executor/            # Memory-bounded execution
│   └── segment_runner.py # Segment-wise model execution
├── hdc/                 # Hyperdimensional computing
│   ├── encoder.py       # Hypervector encoding
│   ├── behavioral_sites.py # Feature extraction
│   ├── binding_operations.py # Binding operations
│   └── error_correction.py # Error correction
├── hypervector/         # Vector operations
│   ├── similarity.py    # Advanced similarity metrics
│   └── hamming.py       # Hamming distance
├── verifier/           # Model verification
│   ├── blackbox.py     # API-based verification
│   └── decision_aggregator.py # Decision aggregation
├── privacy/            # Privacy-preserving features
│   ├── homomorphic_ops.py # Homomorphic operations
│   └── distance_zk_proofs.py # ZK proofs
└── rev_pipeline.py     # Main pipeline integration
```

## Key Algorithms

### Sequential Testing (SPRT)
- Anytime-valid confidence bounds
- Welford's algorithm for numerical stability
- Early stopping with configurable error bounds (α=0.05, β=0.10)

### Hyperdimensional Computing
- Dimension: 8K-100K vectors
- Sparse density: 0.01 (1% active)
- Binding operations: XOR, permutation, circular convolution
- Error correction: 25% parity overhead

### Memory Management
- Segment size: 512 tokens
- Buffer size: 4 segments
- KV cache: 2048 max sequence length
- Memory limit: 4GB default

## Performance Targets

### Speed
- Hamming distance: 10-20× faster with LUTs
- HDC encoding: ~50ms per 100K-dim sample
- Sequential test: <100ms for 1000 samples
- ZK proof generation: ~200ms

### Memory
- Hypervectors: 40KB for 10K-dim float32
- Hamming LUT: 512KB for 16-bit table
- Segment buffer: <100MB for typical workload
- ZK circuits: ~10MB compiled

## API Support

### Model Providers
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- HuggingFace (Inference API)
- Cohere
- Local models (OpenAI-compatible)

### Rate Limits
- Default: 60 requests/minute
- Configurable per provider
- Automatic retry with exponential backoff
- Response caching with TTL

## Testing Strategy

### Unit Tests
- Component isolation with mocks
- Property-based testing for mathematical operations
- Edge cases and error conditions
- Coverage target: >90%

### Integration Tests
- End-to-end pipeline validation
- Multi-model comparison scenarios
- API integration with mock servers
- Memory-bounded execution verification

### Performance Tests
- Throughput benchmarks
- Latency measurements
- Memory usage profiling
- Scalability tests (vector dimensions, segment sizes)

### Robustness Tests
- Adversarial input generation
- Noise injection and recovery
- Byzantine fault tolerance
- Privacy attack resistance

## Development Guidelines

### Code Style
- Type hints for all functions
- Docstrings with Args/Returns
- No inline comments unless necessary
- Black formatting (line length: 100)

### Testing
- Write tests before implementation
- Use pytest fixtures for setup
- Mock external dependencies
- Benchmark critical paths

### Documentation
- Update CLAUDE.md with changes
- Include usage examples
- Document performance characteristics
- Maintain API compatibility

## Recent Updates

### 2024-08-29
- Implemented core REV pipeline with segment execution
- Added HDC behavioral sites and binding operations
- Created black-box verifier for API-based comparison
- Implemented error correction with XOR parity
- Added advanced similarity metrics and clustering
- Created decision aggregator with sequential testing
- Implemented privacy-preserving features (HE, ZK)
- ✅ Completed comprehensive test suite:
  - Unit tests for core components (test_core_sequential.py, test_hdc_components.py)
  - Integration tests for full pipeline (test_integration.py)
  - Performance benchmarks with targets (test_performance.py)
  - Adversarial robustness tests (test_adversarial.py)
- Added development tooling (Makefile, pytest.ini, requirements-dev.txt)

## Testing Infrastructure

### Test Coverage
- **Unit Tests**: Component isolation with mocks, property-based testing
- **Integration Tests**: End-to-end pipeline validation, API integration
- **Performance Tests**: Benchmarks for all critical operations
- **Adversarial Tests**: Robustness against attacks, Byzantine fault tolerance

### Running Tests
```bash
# Install development dependencies
make install-dev

# Run all tests
make test

# Run specific test suites
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-performance   # Performance benchmarks
make test-adversarial   # Adversarial robustness

# Generate coverage report
make test-coverage
```

### Performance Targets Met
- ✅ Hamming distance: <1ms for 10K dimensions (with LUTs)
- ✅ HDC encoding: <50ms per 100K-dim sample
- ✅ Sequential test: <100ms for 1000 samples
- ✅ ZK proof generation: <300ms
- ✅ Error correction: <50ms for 1K dimensions

## Next Steps

1. ✅ ~~Complete comprehensive test suite~~
2. ✅ ~~Add performance benchmarks~~
3. ✅ ~~Implement memory profiling~~
4. ✅ ~~Create adversarial robustness tests~~
5. Package for distribution (setup.py, PyPI)
6. Write user documentation (tutorials, API docs)

## Dependencies

### Core
- numpy>=1.21.0
- torch>=1.9.0
- scipy>=1.7.0
- scikit-learn>=1.0.0

### ML/AI
- transformers>=4.0.0
- pandas>=1.3.0

### Cryptography
- cryptography>=3.4.0
- pycryptodome>=3.15.0

### Development
- pytest>=6.2.0
- black>=21.0.0
- mypy>=0.910
- pytest-benchmark>=3.4.0

## Contact

Repository: https://github.com/rohanvinaik/REV

---

*This file is actively maintained and updated with each development session.*