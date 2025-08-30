# REV - Restriction Enzyme Verification

A memory-bounded, black-box LLM comparison framework using restriction enzyme verification techniques with semantic hypervector behavioral sites, inspired by genomic analysis and hyperdimensional computing.

## 📚 Abstract

REV (Restriction Enzyme Verification) provides a method for comparing large language models whose parameter sizes exceed available device memory. REV treats a transformer as a composition of functional segments separated by **restriction sites**—architecturally or behaviorally meaningful cut points. Using pre-committed challenges and streamed, segment-wise execution, REV emits compact **segment signatures** derived from activations (or output logits in black-box cases). These signatures are committed in a **Merkle tree** per challenge and aggregated with anytime-valid sequential tests to reach **SAME/DIFFERENT/UNDECIDED** conclusions under controlled error rates.

The system extends verification with a **Semantic Hypervector (HDC) layer** inspired by GenomeVault's architecture, enabling **robust, spoof-resistant behavioral sites**. Prompts and model responses are embedded into high-dimensional vectors (8K–100K dims) using binding/permutation operations, fast Hamming-distance LUTs, and optional KAN-HD compression, lifting REV from brittle probes to a **high-dimensional semantic space** where model similarity is measured as **behavioral proximity**.

## 🎯 Motivation & Goals

REV addresses three critical gaps in LLM evaluation:

1. **Memory-bounded verification**: Stream inference through segments to compare models larger than RAM/VRAM
2. **Modular equivalence**: Evaluate **where** two models agree or diverge, not just whether final outputs match
3. **Auditability & tamper-resistance**: Produce cryptographic commitments and reproducible transcripts compatible with Proof-of-Tests (PoT) workflows

### Key Innovation: Semantic Hypervector Behavioral Sites
- Replace brittle probe families with **semantic hypervectors** that encode challenge features and model responses
- Enable **black-box** behavioral sites that are robust, scalable, and privacy-preserving
- Measure model similarity as **behavioral proximity** in high-dimensional space, not merely output token equality

## 🏗️ Core Architecture

### Restriction Site Policies

#### Architectural Sites (White/Gray-box)
- After attention layers
- After MLP layers
- End-of-block boundaries
- Overlapping block windows (e.g., layers 1–8, 5–12, 9–16)

#### Behavioral Sites (HDC-based, Black-box friendly)
- Sites defined by **semantic hypervectors** from challenge features and response distributions
- Multi-resolution "zoom levels" for hierarchical analysis
- Points/tiles in HDC space with distance-based matching

### Segment Processing Pipeline

```python
# Core REV workflow
1. Generate challenges with HMAC-based seeds
2. Stream model execution segment-by-segment
3. Extract signatures at restriction sites
4. Build Merkle tree commitments per challenge
5. Compare models using sequential testing
6. Reach SAME/DIFFERENT/UNDECIDED decision
```

## ⚡ Key Features

### 📊 Memory-Bounded Execution
- **Segment-wise streaming** for models exceeding device memory
- **Single-segment working set** constraint
- **KV cache management** with overlap support
- **Checkpoint/resume** capability for long-running verifications

### 🧬 Hyperdimensional Computing (HDC)
- **8K–100K dimensional vectors** with sparse/dense encoding
- **Binding operations**: XOR, permutation, circular convolution, Fourier
- **Multi-modal feature encoding** from prompts and responses
- **Hierarchical zoom levels** for multi-scale analysis

### 🔐 Cryptographic Commitments
- **Merkle tree construction** for verifiable computation chains
- **Collision-resistant hashing** of sketches/metadata
- **Public transcript generation** with seeds and policies
- **Zero-knowledge friendly** verification support

### ⚡ Performance Optimizations
- **Hamming distance LUTs** for 10–20× speedup
- **16-bit popcount tables** with SIMD acceleration
- **Error-correcting codes** (25% parity overhead)
- **KAN-HD compression** for 50–100× reduction

### 📈 Statistical Testing
- **Sequential Probability Ratio Test (SPRT)** with anytime-valid bounds
- **Empirical-Bernstein confidence intervals**
- **Early stopping** with controlled error rates (α=0.05, β=0.10)
- **Per-challenge indicators** and distance aggregation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rohanvinaik/REV.git
cd REV

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Basic Usage

```python
from src.rev_pipeline import REVPipeline, REVConfig
from src.challenges.kdf_prompts import EnhancedKDFPromptGenerator
from src.hdc.encoder import UnifiedHDCEncoder

# Configure REV
config = REVConfig(
    hdc_config=HDCConfig(
        dimension=16384,          # Hypervector dimension
        use_sparse=True,          # Sparse encoding
        sparse_density=0.01       # 1% active bits
    ),
    segment_config=SegmentConfig(
        segment_size=512,         # Tokens per segment
        buffer_size=1024          # Streaming buffer
    ),
    sequential_config=SequentialConfig(
        alpha=0.05,               # Type I error
        beta=0.10,                # Type II error
        d_threshold=0.08          # Distance threshold
    )
)

# Initialize pipeline
pipeline = REVPipeline(config)

# Generate challenges with HMAC seeding
generator = EnhancedKDFPromptGenerator(seed=b"verification_key")
challenges = generator.generate_diverse_batch(
    n_challenges=100,
    diversity_weight=0.3
)

# Run verification
result = pipeline.verify_models(
    model_a="model_a_endpoint",
    model_b="model_b_endpoint",
    challenges=challenges,
    policy=ExecutionPolicy(
        temperature=0.0,          # Deterministic
        max_tokens=100,
        dtype="fp16"
    )
)

print(f"Decision: {result.verdict}")      # SAME/DIFFERENT/UNDECIDED
print(f"Confidence: {result.confidence:.3f}")
print(f"First divergence: {result.first_divergence_site}")
```

### Semantic Hypervector Encoding

```python
from src.hdc.encoder import UnifiedHDCEncoder
from src.hdc.behavioral_sites import BehavioralSiteExtractor

# Create HDC encoder
encoder = UnifiedHDCEncoder(config.hdc_config)

# Extract behavioral sites from model response
extractor = BehavioralSiteExtractor()
sites = extractor.extract_sites(
    prompt=challenge.prompt,
    response=model_response,
    zoom_level=1  # Prompt-level granularity
)

# Encode to hypervector
probe_hv = encoder.encode_probe(challenge.features)
response_hv = encoder.encode_response(model_response.logits)

# Compute behavioral similarity
similarity = encoder.compute_similarity(probe_hv, response_hv)
```

### Segment-wise Execution (Memory-bounded)

```python
from src.executor.segment_runner import SegmentRunner

# Initialize segment runner
runner = SegmentRunner(config.segment_config)

# Stream through model segments
for segment in model.segments:
    # Load segment parameters (offload-aware)
    runner.load_segment(segment)
    
    # Run forward pass
    activations = runner.forward(input_states)
    
    # Build signature
    signature = runner.build_signature(
        activations,
        segment_id=segment.id,
        projector_seed=segment.seed
    )
    
    # Release segment memory
    runner.release_segment(segment)
    
    # Commit to Merkle tree
    merkle_tree.add_leaf(signature.hash)
```

## 🔬 Technical Details

### Challenge Generation (PoT-aligned)
```python
# Deterministic challenge synthesis
seed_i = HMAC(key, f"{run_id}:{i}")
prompt = synthesize_prompt(seed_i)
```

### Signature Construction
```python
# Per-segment signature building
def build_signature(activations, segment_site, policy):
    a = select_and_pool(activations)           # Fixed pooling
    R = seeded_random_matrix(segment.seed)     # Random projection
    z = quantize(clip(R @ a, tau), q)         # Quantization
    sigma = binarize(z)                        # Binary sketch
    leaf = hash(encode({
        "seg": segment.id,
        "sigma": sigma,
        "policy": policy
    }))
    return Signature(segment.id, sigma, {"leaf": leaf})
```

### HDC Behavioral Encoding
```python
# Map prompts/responses to hypervectors
def probe_to_hypervector(features, dims=16384):
    vec = rand_hv(dims, seed=0xBEEF)
    for k, v in features.items():
        hv_k = rand_hv(dims, seed=hash32(k))
        hv_v = rand_hv(dims, seed=hash32(v))
        vec ^= permute(hv_k, shift=hash32(k+v) % 257) ^ hv_v
    return vec

def response_to_hypervector(logits, dims=16384):
    vec = rand_hv(dims, seed=0xF00D)
    for rank, (tok_id, p) in enumerate(topk(logits, K=16)):
        hv_tok = rand_hv(dims, seed=tok_id)
        hv_rnk = rand_hv(dims, seed=rank)
        vec ^= weighted_bind(hv_tok, hv_rnk, weight=p)
    return vec
```

### Sequential Decision Logic
```python
def sequential_decision(stream, alpha=0.01, beta=0.01):
    S_match = init_seq_test(alpha)
    S_dist = init_seq_test(beta)
    
    for t, result in enumerate(stream, 1):
        update(S_match, result["merkle_match"])
        update(S_dist, result["hdc_distance"])
        
        if accept_same(S_match, S_dist):
            return "SAME", t
        if accept_diff(S_match, S_dist):
            return "DIFFERENT", t
        if t >= max_challenges:
            break
    
    return "UNDECIDED", t
```

## 📊 Performance Characteristics

### Computational Efficiency

| Operation | Traditional | HDC-Optimized | Speedup |
|-----------|------------|---------------|---------|
| Similarity Search (1M sites) | 10–30s | 10–50ms | ~1,500–3,000× |
| Hamming Distance (10K-dim) | 50–100µs | 5–10µs | ~10–20× |
| Segment Signature | 200ms | 50ms | 4× |
| Merkle Tree (1K leaves) | 100ms | 20ms | 5× |

### Memory Usage

- **Per-segment working set**: <2GB for 70B models
- **Hypervector storage**: 2KB for 16K-dim binary vector
- **Merkle tree**: O(log n) proof size
- **KV cache**: Configurable, typically 2048 tokens

## 🛡️ Security Properties

### Threat Model
- Black-box or gray-box model access
- Potential adversarial control over runtime
- Model-switching or wrapper orchestration attacks

### Mitigations
- **Overlapping windows** resist stitching attacks
- **Multi-resolution zoom** prevents single-point spoofing
- **Distributed HDC representation** thwarts bit manipulation
- **Merkle commitments** ensure auditability
- **Pre-committed challenges** prevent adaptive attacks

## 🔧 Advanced Features

### Privacy-Preserving Comparison
```python
from src.privacy.homomorphic_ops import HomomorphicComputation
from src.privacy.distance_zk_proofs import DistanceZKProofSystem

# Homomorphic similarity computation
he = HomomorphicComputation()
encrypted_hv_a = he.encrypt(hypervector_a)
encrypted_hv_b = he.encrypt(hypervector_b)
encrypted_distance = he.compute_distance(encrypted_hv_a, encrypted_hv_b)

# Zero-knowledge proof of behavioral closeness
zk = DistanceZKProofSystem(max_distance=1000)
proof = zk.prove_distance_threshold(hv_a, hv_b, threshold=100)
valid = zk.verify_proof(commitment_a, commitment_b, threshold, proof)
```

### Hierarchical Zoom Levels
```python
# Multi-scale behavioral analysis
zoom_levels = {
    0: {},  # Corpus/site-wide prototypes
    1: {},  # Prompt-level hypervectors
    2: {},  # Span/tile-level hypervectors
}

# Analyze at different granularities
for level in range(3):
    sites = extract_sites_at_zoom(response, level)
    hv = encode_sites_to_hypervector(sites, level)
    zoom_levels[level][challenge_id] = hv
```

### Error Correction
```python
# Add XOR parity blocks for robustness
def add_error_correction(hypervector, parity_rate=0.25):
    n_parity = int(len(hypervector) * parity_rate)
    parity_blocks = []
    
    for i in range(n_parity):
        block_indices = hash_to_indices(i, len(hypervector))
        parity = xor_reduce(hypervector[block_indices])
        parity_blocks.append(parity)
    
    return hypervector, parity_blocks
```

## 📋 Configuration

### Execution Policy
```python
class ExecutionPolicy:
    temperature: float = 0.0      # Deterministic decoding
    top_p: float = 1.0
    max_tokens: int = 100
    dtype: str = "fp16"           # Precision
    seed: int = 42                # Fixed seed
    attn_impl: str = "paged"      # Memory-efficient attention
```

### HDC Policy
```python
class HDCPolicy:
    dimension: int = 16384        # Hypervector dimension
    binding_ops: List[str] = ["xor", "permutation"]
    top_k: int = 16              # Top-K tokens for response encoding
    use_ecc: bool = True         # Error correction
    use_kan_hd: bool = False     # KAN-HD compression
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/test_unified_system.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run performance benchmarks
pytest tests/test_performance.py -v --benchmark-only

# Run adversarial tests
pytest tests/test_adversarial.py -v
```

## 📈 Evaluation Results

### Sanity Checks
- Model A vs Model A: **SAME** (100% match rate)
- Model A vs Quantized(A): **SAME** (>95% match with HDC)
- Model A vs Distilled(A): **DIFFERENT** at specific layers
- Model A vs Model B (different family): **DIFFERENT** immediately

### Operating Characteristics
- False Accept Rate (FAR): <0.01 at α=0.05
- False Reject Rate (FRR): <0.10 at β=0.10
- Average stopping time: 742 challenges for 95% confidence
- Localization accuracy: 98% for injected edits

## 🚢 Deployment

### Docker Support
```bash
# Build and run with Docker
docker build -t rev-verifier .
docker run -p 8000:8000 rev-verifier

# With GPU support
docker run --gpus all -p 8000:8000 rev-verifier
```

### Production Configuration
```yaml
# docker-compose.yml
services:
  rev-verifier:
    image: rev-verifier:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4GB
    environment:
      - MAX_SEGMENT_SIZE=512
      - HDC_DIMENSION=16384
      - USE_GPU=true
```

## 📚 Documentation

- [Architecture Details](docs/architecture.md)
- [HDC/GenomeVault Integration](docs/hdc_integration.md)
- [Security Model](docs/security.md)
- [API Reference](docs/api.md)

## 🔬 Research Foundation

This implementation is based on the paper:
**"Restriction Enzyme Verification (REV) for Memory-Bounded, Black-Box LLM Comparison with Semantic Hypervector Behavioral Sites"**

Key concepts adapted from:
- GenomeVault's semantic hypervector architecture
- Proof-of-Tests (PoT) sequential testing framework
- Transformer mechanistic interpretability research
- Vector-symbolic/hyperdimensional computing

## 🤝 Contributing

Contributions welcome! Please ensure:
1. Tests pass (`pytest tests/`)
2. Code is formatted (`black src/ tests/`)
3. Types are correct (`mypy src/`)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

This work integrates:
- **GenomeVault** semantic hypervector architecture (HDC encoding, binding ops, LUT acceleration)
- **PoT** anytime-valid sequential testing framework
- Transformer interpretability research from mechanistic circuits literature
- Authenticated data structures and zero-knowledge proof systems

## 📞 Contact

- **Repository**: https://github.com/rohanvinaik/REV
- **Paper**: [docs/REV_paper.md](docs/Restriction%20Enzyme%20Verification%20(REV)%20for%20Memory-Bounded,%20Black-Box%20LLM%20Comparison.md)

---

**Version**: 1.0.0 | **Status**: Research Implementation | **Last Updated**: August 2024