# REV - Restriction Enzyme Verification System

## 🚀 QUICK START - RUNNING THE PIPELINE

### Default Mode: API-Only (Recommended)
```bash
# Single model verification (uses API, no local loading)
python run_rev.py meta-llama/Llama-3.3-70B-Instruct

# Compare multiple models
python run_rev.py gpt-4 claude-3-opus meta-llama/Llama-3.3-70B

# With more challenges and debug output
python run_rev.py meta-llama/Llama-3.3-70B-Instruct --challenges 10 --debug

# Specify API provider explicitly
python run_rev.py gpt-4 --provider openai --api-key sk-...
```

### Local Mode (Only for Small Models)
```bash
# WARNING: Only use --local for models <20GB. Larger models will exhaust memory!
python run_rev.py meta-llama/Llama-2-7b-hf --local --device cpu --quantize 4bit

# With memory limit (still may fail for very large models)
python run_rev.py /path/to/small-model --local --memory-limit 20
```

### Important Notes:
- **DEFAULT IS API-ONLY**: The pipeline uses APIs by default, no local model loading
- **DO NOT USE --local FOR LARGE MODELS**: Models >20GB will crash your system
- **The 70B/405B models should ONLY be run via API**
- Use `run_rev.py` - this is the ONLY unified pipeline script

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

## Recent Updates (September 2025)

### Unified Pipeline (v3.0)
- **CONSOLIDATED**: All pipeline scripts merged into single `run_rev.py`
- **API-FIRST**: Default mode is now API-only (no local model loading)
- **BEHAVIORAL ANALYSIS**: Integrated sophisticated PoT behavioral probing
- **DIVERGENCE METRICS**: Information-theoretic divergence calculation
- **MEMORY SAFETY**: Added --memory-limit flag for local mode
- **MULTI-MODEL**: Support for comparing multiple models in one run

### Deleted Old Scripts
The following scripts have been removed (functionality merged into `run_rev.py`):
- `run_pipeline.py` 
- `run_rev_complete.py`
- `run_rev_e2e.py`

## Multi-Stage Orchestrated Testing (NEW)

REV now includes intelligent multi-stage testing that adapts based on model architecture identification:

### Architecture Identification Workflow

1. **Quick Identification Stage** (5 minutes)
   - Analyzes first 10 layers with basic probes
   - Compares against fingerprint library of known architectures
   - Identifies: Llama, GPT, Mistral, Qwen, Yi families

2. **Adaptive Testing Strategy**
   - **Known Architecture** (>85% confidence): Targeted testing on known vulnerabilities
   - **Variant/Fine-tuned** (60-85% confidence): Adaptive approach with expanded testing  
   - **Novel Architecture** (<60% confidence): Full exploratory analysis

3. **Cassette Loading**
   - Automatically selects appropriate test cassettes based on identified architecture
   - Focus layers determined by known behavioral patterns
   - Optimizes testing time by skipping stable regions

### Command Examples

```bash
# Standard testing (backward compatible)
python run_rev.py /path/to/model

# Orchestrated multi-stage testing
python run_rev.py /path/to/model --orchestrate

# With claimed architecture verification
python run_rev.py /path/to/model --orchestrate --claimed-family llama

# With time budget (in hours)
python run_rev.py /path/to/model --orchestrate --time-budget 2.5

# List known architectures in library
python run_rev.py --list-known-architectures

# Add discovered fingerprint to library
python run_rev.py /path/to/model --orchestrate --add-to-library

# Export/Import fingerprints
python run_rev.py /path/to/model --orchestrate --export-fingerprint model.fp
python run_rev.py --import-fingerprint model.fp
```

### Fingerprint Library

The system maintains a library of base model fingerprints at `./fingerprint_library/`:

```
fingerprint_library/
├── fingerprint_library.json  # Main registry
├── llama/                    # Llama family fingerprints
├── gpt/                      # GPT family fingerprints
├── mistral/                  # Mistral family fingerprints
└── novel/                    # Discovered novel architectures
```

### Testing Strategies by Architecture

| Architecture | Confidence | Strategy | Time | Focus |
|-------------|------------|----------|------|-------|
| Known (Llama 70B) | >85% | Targeted | 2h | Layers 15,35,55 (vulnerable) |
| Variant | 60-85% | Adaptive | 3h | Every 8th layer |
| Novel | <60% | Exploratory | 4h | Every 5th layer |

### Strategic Benefits

1. **Time Efficiency**: 70% reduction for known architectures
2. **Accuracy**: Focus on vulnerable layers increases detection rate
3. **Adaptability**: Automatically adjusts to novel architectures
4. **Knowledge Base**: Library grows with each novel architecture

### INCORRECT - Don't use these (deleted):
python run_rev_complete.py ...  # DELETED
python run_pipeline.py ...      # DELETED
python run_rev_e2e.py ...       # DELETED
```

## Contact

Repository: https://github.com/rohanvinaik/REV

---

*This file is actively maintained and updated with each development session.*