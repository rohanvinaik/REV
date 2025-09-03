# REV API Reference

## Table of Contents

1. [Core Pipeline](#core-pipeline)
2. [Feature Extraction](#feature-extraction)
3. [Hyperdimensional Computing](#hyperdimensional-computing)
4. [Statistical Testing](#statistical-testing)
5. [Model Verification](#model-verification)
6. [Prompt Orchestration](#prompt-orchestration)
7. [Error Handling](#error-handling)
8. [REST API](#rest-api)
9. [Utilities](#utilities)

---

## Core Pipeline

### REVUnified

Main orchestrator for the entire REV system.

```python
class REVUnified:
    """
    Unified REV system orchestrator.
    
    Integrates all components for model verification using
    hyperdimensional computing and restriction enzyme metaphor.
    """
```

#### `__init__(self, **kwargs)`

Initialize REV system with configuration.

**Parameters:**
- `memory_limit_gb` (float, default=4.0): Maximum memory usage per segment (1.0-16.0)
- `enable_prompt_orchestration` (bool, default=False): Enable 7-system prompt orchestration
- `enable_principled_features` (bool, default=False): Use automated feature extraction
- `debug` (bool, default=False): Enable debug logging
- `device` (str, default='auto'): Compute device ('cpu', 'cuda', 'mps', 'auto')
- `checkpoint_dir` (str, optional): Directory for checkpoints
- `cache_dir` (str, optional): Directory for caching

**Returns:**
- `REVUnified`: Initialized REV system

**Example:**
```python
from run_rev import REVUnified

rev = REVUnified(
    memory_limit_gb=2.0,
    enable_prompt_orchestration=True,
    enable_principled_features=True,
    debug=True
)
```

#### `process_model(self, model_path: str, challenges: int = 30) -> ModelAnalysisResult`

Process and verify a model.

**Parameters:**
- `model_path` (str): Path to model directory containing config.json
- `challenges` (int, default=30): Number of verification challenges (5-500)

**Returns:**
- `ModelAnalysisResult`: Contains:
  - `fingerprint` (UnifiedFingerprint): Model's hyperdimensional fingerprint
  - `confidence` (float): Verification confidence [0, 1]
  - `model_family` (str): Detected model family
  - `restriction_sites` (List[RestrictionSite]): Behavioral boundaries
  - `metrics` (Dict): Performance and accuracy metrics

**Raises:**
- `ModelLoadError`: If model cannot be loaded
- `InvalidModelPath`: If path doesn't contain valid model

**Example:**
```python
result = rev.process_model(
    "/path/to/llama-70b",
    challenges=50
)
print(f"Confidence: {result.confidence:.2%}")
print(f"Model family: {result.model_family}")
```

---

## Feature Extraction

### HierarchicalFeatureTaxonomy

Implements principled feature extraction across 4 categories.

```python
class HierarchicalFeatureTaxonomy:
    """
    56 principled features organized hierarchically:
    - Syntactic (9 features)
    - Semantic (20 features)
    - Behavioral (9 features)
    - Architectural (18 features)
    """
```

#### `extract_all_features(self, model_output: Any, **kwargs) -> Dict[str, np.ndarray]`

Extract all features from model output.

**Parameters:**
- `model_output` (Any): Raw model output (text, logits, or hidden states)
- `**kwargs`: Additional context:
  - `prompt` (str): Input prompt used
  - `layer_idx` (int): Layer index for architectural features
  - `attention_weights` (Tensor): Attention matrices

**Returns:**
- `Dict[str, np.ndarray]`: Feature vectors by category:
  - `"syntactic"`: Shape (9,)
  - `"semantic"`: Shape (20,)
  - `"behavioral"`: Shape (9,)
  - `"architectural"`: Shape (18,)

**Example:**
```python
taxonomy = HierarchicalFeatureTaxonomy()
features = taxonomy.extract_all_features(
    model_output="Generated text response",
    prompt="What is machine learning?",
    layer_idx=15
)
print(f"Total features: {sum(len(v) for v in features.values())}")
```

#### `get_concatenated_features(self, features: Dict[str, np.ndarray]) -> np.ndarray`

Concatenate all feature categories into single vector.

**Parameters:**
- `features` (Dict[str, np.ndarray]): Feature dictionary from extract_all_features

**Returns:**
- `np.ndarray`: Concatenated feature vector, shape (56,)

**Example:**
```python
concat_features = taxonomy.get_concatenated_features(features)
assert concat_features.shape == (56,)
```

### AutomaticFeaturizer

Automatic feature discovery and selection.

```python
class AutomaticFeaturizer:
    """
    Discovers and selects features using:
    - Mutual information
    - LASSO regularization
    - Elastic net
    - Ensemble methods
    """
```

#### `discover_features(self, X: np.ndarray, y: np.ndarray, method: str = 'ensemble') -> np.ndarray`

Discover important features from data.

**Parameters:**
- `X` (np.ndarray): Feature matrix, shape (n_samples, n_features)
- `y` (np.ndarray): Target labels, shape (n_samples,)
- `method` (str): Discovery method:
  - `'mutual_info'`: Mutual information ranking
  - `'lasso'`: LASSO regularization
  - `'elastic_net'`: Elastic net regularization
  - `'ensemble'`: Combination of all methods

**Returns:**
- `np.ndarray`: Selected feature indices, shape (n_selected,)

**Example:**
```python
featurizer = AutomaticFeaturizer(n_features_to_select=100)
selected_indices = featurizer.discover_features(
    X_train, y_train, method='ensemble'
)
X_selected = X_train[:, selected_indices]
```

---

## Hyperdimensional Computing

### HDCEncoder

Encodes features into high-dimensional binary vectors.

```python
class HDCEncoder:
    """
    Hyperdimensional computing encoder.
    Maps features to binary vectors in {0,1}^d.
    """
```

#### `__init__(self, dimension: int = 10000, sparsity: float = 0.01)`

Initialize HDC encoder.

**Parameters:**
- `dimension` (int): Hypervector dimension (1000-100000)
- `sparsity` (float): Sparsity level (0.001-0.15)

**Returns:**
- `HDCEncoder`: Initialized encoder

#### `encode_vector(self, features: np.ndarray) -> np.ndarray`

Encode feature vector to hypervector.

**Parameters:**
- `features` (np.ndarray): Feature vector, any shape

**Returns:**
- `np.ndarray`: Binary hypervector, shape (dimension,), dtype=uint8

**Example:**
```python
encoder = HDCEncoder(dimension=10000, sparsity=0.01)
features = np.random.randn(56)
hypervector = encoder.encode_vector(features)
assert hypervector.shape == (10000,)
assert np.mean(hypervector) < 0.02  # Sparse
```

#### `bundle(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray`

Bundle (XOR) two hypervectors.

**Parameters:**
- `hv1` (np.ndarray): First hypervector
- `hv2` (np.ndarray): Second hypervector

**Returns:**
- `np.ndarray`: Bundled hypervector

**Mathematical Operation:**
```
h_bundled = h1 ⊕ h2
```

#### `bind(self, hv: np.ndarray, shift: int) -> np.ndarray`

Bind hypervector with rotation.

**Parameters:**
- `hv` (np.ndarray): Hypervector to bind
- `shift` (int): Rotation amount

**Returns:**
- `np.ndarray`: Bound hypervector

**Mathematical Operation:**
```
h_bound = rotate(h, shift)
```

### UnifiedFingerprint

Complete model fingerprint with multiple pathways.

```python
class UnifiedFingerprint:
    """
    Unified fingerprint combining:
    - Behavioral pathway (response patterns)
    - Structural pathway (architecture)
    - Semantic pathway (understanding)
    - Syntactic pathway (language use)
    """
```

#### `generate(self, responses: List[str], features: Dict) -> UnifiedFingerprint`

Generate unified fingerprint from model responses.

**Parameters:**
- `responses` (List[str]): Model responses to challenges
- `features` (Dict): Extracted features by category

**Returns:**
- `UnifiedFingerprint`: Complete fingerprint object

**Example:**
```python
generator = UnifiedFingerprintGenerator()
fingerprint = generator.generate(
    responses=["Response 1", "Response 2"],
    features={"syntactic": np.array([...]), ...}
)
print(f"Fingerprint dimension: {fingerprint.dimension}")
```

---

## Statistical Testing

### SequentialTest

Implements Sequential Probability Ratio Test (SPRT).

```python
class SequentialTest:
    """
    Wald's Sequential Probability Ratio Test.
    Provides optimal stopping with guaranteed error bounds.
    """
```

#### `__init__(self, alpha: float = 0.05, beta: float = 0.05, theta_0: float = 0.5, theta_1: float = 0.7)`

Initialize SPRT.

**Parameters:**
- `alpha` (float): Type I error rate (false positive), range [0.01, 0.1]
- `beta` (float): Type II error rate (false negative), range [0.01, 0.1]
- `theta_0` (float): Null hypothesis parameter, range [0.3, 0.7]
- `theta_1` (float): Alternative hypothesis parameter, range [0.5, 0.9]

**Returns:**
- `SequentialTest`: Initialized test

**Mathematical Boundaries:**
```
A = (1-β)/α  (upper boundary)
B = β/(1-α)  (lower boundary)
```

#### `add_sample(self, value: float) -> TestDecision`

Add sample and update decision.

**Parameters:**
- `value` (float): Sample value, range [0, 1]

**Returns:**
- `TestDecision`: One of:
  - `CONTINUE`: Need more samples
  - `ACCEPT_H0`: Accept null hypothesis
  - `ACCEPT_H1`: Accept alternative hypothesis

**Example:**
```python
test = SequentialTest(alpha=0.05, beta=0.05)
for similarity_score in scores:
    decision = test.add_sample(similarity_score)
    if decision != TestDecision.CONTINUE:
        break
print(f"Decision: {decision}, Samples: {len(test.samples)}")
```

---

## Model Verification

### ModelVerifier

Core verification engine.

```python
class ModelVerifier:
    """
    Verifies model identity using fingerprint comparison.
    """
```

#### `verify(self, fingerprint: UnifiedFingerprint, reference_library: ModelLibrary) -> VerificationResult`

Verify model against reference library.

**Parameters:**
- `fingerprint` (UnifiedFingerprint): Model fingerprint to verify
- `reference_library` (ModelLibrary): Reference fingerprint library

**Returns:**
- `VerificationResult`: Contains:
  - `is_verified` (bool): Verification success
  - `confidence` (float): Confidence score [0, 1]
  - `matched_family` (str): Best matching model family
  - `similarity_scores` (Dict[str, float]): Scores per family

**Example:**
```python
verifier = ModelVerifier()
result = verifier.verify(fingerprint, library)
if result.is_verified:
    print(f"Verified as {result.matched_family}")
    print(f"Confidence: {result.confidence:.2%}")
```

### RestrictionSite

Behavioral divergence point in model.

```python
@dataclass
class RestrictionSite:
    """
    High-divergence layer marking behavioral boundary.
    Analogous to restriction enzyme recognition sites.
    """
    layer_idx: int  # Layer index (0-based)
    behavioral_divergence: float  # Divergence score [0, 1]
    confidence_score: float  # Statistical confidence [0, 1]
    site_type: str  # 'attention', 'mlp', 'norm'
```

---

## Prompt Orchestration

### PromptOrchestrator

Manages 7 specialized prompt generation systems.

```python
class PromptOrchestrator:
    """
    Orchestrates prompt generation across:
    1. PoT (30%) - Behavioral probes
    2. KDF (20%) - Security/adversarial
    3. Evolutionary (20%) - Genetic optimization
    4. Dynamic (20%) - Template synthesis
    5. Hierarchical (10%) - Taxonomical
    6. Predictor - Effectiveness scoring
    7. Profiler - Pattern analysis
    """
```

#### `generate_prompts(self, n: int, strategy: str = 'balanced') -> List[str]`

Generate orchestrated prompts.

**Parameters:**
- `n` (int): Number of prompts to generate (1-500)
- `strategy` (str): Generation strategy:
  - `'balanced'`: Default weighted distribution
  - `'adversarial'`: Focus on security testing
  - `'behavioral'`: Focus on behavioral boundaries
  - `'comprehensive'`: All systems equally

**Returns:**
- `List[str]`: Generated prompts

**Example:**
```python
orchestrator = PromptOrchestrator()
prompts = orchestrator.generate_prompts(
    n=100,
    strategy='adversarial'
)
```

### PoTChallengeGenerator

Proof of Thought challenge generation.

```python
class PoTChallengeGenerator:
    """
    Generates behavioral probe challenges.
    Targets restriction sites for maximum divergence.
    """
```

#### `generate_challenges(self, difficulty: str = 'medium', n: int = 10) -> List[Challenge]`

Generate PoT challenges.

**Parameters:**
- `difficulty` (str): Challenge difficulty:
  - `'easy'`: Basic probes
  - `'medium'`: Standard testing
  - `'hard'`: Edge cases
  - `'extreme'`: Adversarial
- `n` (int): Number of challenges (1-100)

**Returns:**
- `List[Challenge]`: Challenge objects with prompts and expected behaviors

---

## Error Handling

### CircuitBreaker

Prevents cascade failures.

```python
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    States: closed -> open -> half_open -> closed
    """
```

#### `call(self, func: Callable, *args, **kwargs) -> Any`

Execute function with circuit breaker protection.

**Parameters:**
- `func` (Callable): Function to execute
- `*args`: Positional arguments
- `**kwargs`: Keyword arguments

**Returns:**
- `Any`: Function result

**Raises:**
- `CircuitOpenError`: If circuit is open
- `Exception`: Original exception if circuit is closed

**Example:**
```python
breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0
)
result = breaker.call(api_call, endpoint="/verify")
```

### retry_with_backoff

Decorator for automatic retries.

```python
@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2,
    jitter=True
)
def unstable_operation():
    """Operation that might fail transiently."""
    pass
```

**Parameters:**
- `max_retries` (int): Maximum retry attempts (0-10)
- `initial_delay` (float): Initial backoff delay in seconds
- `max_delay` (float): Maximum backoff delay
- `exponential_base` (float): Backoff multiplier
- `jitter` (bool): Add random jitter to prevent thundering herd

---

## REST API

### FastAPI Endpoints

#### `POST /analyze`

Analyze and verify a model.

**Request Body:**
```json
{
  "model_path": "/path/to/model",
  "challenges": 50,
  "enable_principled_features": true,
  "enable_prompt_orchestration": true,
  "strategy": "balanced"
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "status": "completed",
  "model_info": {
    "name": "llama-70b",
    "family": "llama",
    "architecture": "LlamaForCausalLM",
    "parameters": 70000000000
  },
  "verification": {
    "is_verified": true,
    "confidence": 0.95,
    "matched_family": "llama"
  },
  "fingerprint": {
    "dimension": 10000,
    "sparsity": 0.01,
    "pathways": ["behavioral", "structural", "semantic", "syntactic"]
  },
  "metrics": {
    "processing_time": 120.5,
    "memory_peak_gb": 3.2,
    "samples_used": 35
  }
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `404`: Model not found
- `429`: Rate limited
- `500`: Internal error

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "model_path": "/models/llama-70b",
        "challenges": 50
    }
)
result = response.json()
```

#### `POST /compare`

Compare two fingerprints.

**Request Body:**
```json
{
  "fingerprint1": {...},
  "fingerprint2": {...},
  "method": "hamming"
}
```

**Response:**
```json
{
  "similarity": 0.87,
  "distance": 1300,
  "method": "hamming",
  "is_same_family": true,
  "confidence": 0.92
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "3.0.0",
  "uptime_seconds": 3600,
  "memory_usage_gb": 2.1,
  "active_requests": 3
}
```

### WebSocket API

#### `/ws/analysis`

Real-time analysis updates.

**Client Message:**
```json
{
  "type": "start_analysis",
  "model_path": "/path/to/model",
  "challenges": 50
}
```

**Server Messages:**
```json
{
  "type": "progress",
  "progress": 0.45,
  "current_step": "extracting_features",
  "eta_seconds": 60
}
```

```json
{
  "type": "completed",
  "result": {...}
}
```

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    console.log(`Progress: ${data.progress * 100}%`);
  }
};
```

---

## Utilities

### HammingDistance

Optimized Hamming distance computation.

```python
class HammingDistanceOptimized:
    """
    SIMD-accelerated Hamming distance.
    10-20x faster than naive implementation.
    """
```

#### `distance(self, hv1: np.ndarray, hv2: np.ndarray) -> int`

Compute Hamming distance between binary vectors.

**Parameters:**
- `hv1` (np.ndarray): First binary vector
- `hv2` (np.ndarray): Second binary vector

**Returns:**
- `int`: Hamming distance (number of differing bits)

**Performance:**
- 10,000 dimensions: <1ms
- 100,000 dimensions: <10ms

**Example:**
```python
hamming = HammingDistanceOptimized()
dist = hamming.distance(fingerprint1.vector, fingerprint2.vector)
similarity = 1.0 - (dist / len(fingerprint1.vector))
```

### CheckpointManager

Manages checkpoint saving and loading.

```python
class CheckpointManager:
    """
    Checkpoint management with automatic pruning.
    """
```

#### `save(self, state: Dict, metrics: Dict) -> Path`

Save checkpoint.

**Parameters:**
- `state` (Dict): State to checkpoint
- `metrics` (Dict): Associated metrics

**Returns:**
- `Path`: Checkpoint file path

**Example:**
```python
manager = CheckpointManager(
    checkpoint_dir="checkpoints/",
    max_checkpoints=5,
    save_best=True,
    metric_name="confidence"
)
path = manager.save(
    state={"fingerprint": fp, "progress": 0.5},
    metrics={"confidence": 0.92}
)
```

### SeedManager

Ensures reproducibility across runs.

```python
class SeedManager:
    """
    Manages random seeds for all libraries.
    """
```

#### `set_all_seeds(self, seed: int = 42)`

Set seeds for all random number generators.

**Parameters:**
- `seed` (int): Random seed (0-2^32)

**Affected Libraries:**
- Python `random`
- NumPy
- PyTorch
- TensorFlow (if available)

**Example:**
```python
seed_mgr = SeedManager()
seed_mgr.set_all_seeds(42)
# All random operations now reproducible
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Feature extraction | O(n) | 50-100ms |
| HDC encoding | O(d) | <50ms |
| Hamming distance | O(d) | <1ms |
| SPRT update | O(1) | <0.1ms |
| Fingerprint generation | O(n*d) | 100-200ms |
| Library search | O(m*d) | 10-50ms |

Where:
- n = number of features
- d = hypervector dimension
- m = library size

### Memory Requirements

| Component | Memory Usage | Notes |
|-----------|-------------|--------|
| Hypervector (10K dim) | 10KB | Binary packed |
| Hypervector (100K dim) | 100KB | Binary packed |
| Feature cache | 100MB | Per 1000 samples |
| Model segment | 2-4GB | Configurable cap |
| Reference library | 50-200MB | ~100 models |
| Active library | Unlimited | Auto-pruned |

---

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| E001 | ModelLoadError | Cannot load model from path |
| E002 | InvalidModelPath | Path doesn't contain valid model |
| E003 | OutOfMemoryError | Memory limit exceeded |
| E004 | CircuitOpenError | Circuit breaker is open |
| E005 | RateLimitError | Rate limit exceeded |
| E006 | FingerprintMismatchError | Fingerprint validation failed |
| E007 | CheckpointError | Cannot save/load checkpoint |
| E008 | FeatureExtractionError | Feature extraction failed |
| E009 | VerificationTimeoutError | Verification timed out |
| E010 | InvalidParameterError | Invalid parameter value |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REV_MEMORY_LIMIT_GB` | Memory limit per segment | 4.0 |
| `REV_CACHE_DIR` | Cache directory | ~/.rev/cache |
| `REV_CHECKPOINT_DIR` | Checkpoint directory | ./checkpoints |
| `REV_LOG_LEVEL` | Logging level | INFO |
| `REV_DEBUG` | Enable debug mode | false |
| `REV_DEVICE` | Compute device | auto |
| `REV_API_TIMEOUT` | API timeout (seconds) | 300 |
| `REV_RATE_LIMIT` | Requests per minute | 60 |

---

## Thread Safety

All public API methods are thread-safe except where noted:

**Thread-Safe:**
- `REVUnified.process_model()`
- `HammingDistanceOptimized.distance()`
- `SequentialTest.add_sample()`
- `CircuitBreaker.call()`

**Not Thread-Safe:**
- `CheckpointManager.save()` (use lock)
- `ModelLibrary.add_fingerprint()` (use lock)
- `PromptOrchestrator` state modifications

**Example:**
```python
from threading import Lock

library_lock = Lock()

def add_to_library(fingerprint):
    with library_lock:
        library.add_fingerprint(fingerprint)
```

---

*For more examples and tutorials, see the [examples/](../examples/) directory.*