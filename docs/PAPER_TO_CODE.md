# Paper to Code Mapping

## Overview
This document maps concepts from the REV paper "Restriction Enzyme Verification of Large Language Models" to their implementation in the codebase.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Mathematical Notation](#mathematical-notation)
3. [Algorithms](#algorithms)
4. [Figures and Visualizations](#figures-and-visualizations)
5. [Experimental Results](#experimental-results)

## Core Concepts

### Paper Section → Code Module Mapping

| Paper Section | Concept | Implementation | File Location |
|--------------|---------|----------------|---------------|
| **Section 2: Biological Metaphor** | | | |
| 2.1 | Restriction Enzymes | `RestrictionSite` class | `src/hdc/behavioral_sites.py` |
| 2.2 | Recognition Sites | Layer divergence detection | `src/models/true_segment_execution.py:identify_all_restriction_sites()` |
| 2.3 | Cleavage Patterns | Behavioral boundaries | `src/hdc/behavioral_sites.py:BehavioralSites` |
| **Section 3: Hyperdimensional Computing** | | | |
| 3.1 | HDC Encoding | Binary hypervector generation | `src/hdc/encoder.py:HypervectorEncoder` |
| 3.2 | Bundling Operation | XOR combination | `src/hdc/encoder.py:bundle()` |
| 3.3 | Binding Operation | Rotation/permutation | `src/hdc/encoder.py:bind()` |
| 3.4 | Similarity Metrics | Hamming distance | `src/hypervector/hamming.py:HammingDistanceOptimized` |
| **Section 4: Model Fingerprinting** | | | |
| 4.1 | Feature Extraction | Principled feature taxonomy | `src/features/taxonomy.py:HierarchicalFeatureTaxonomy` |
| 4.2 | Prompt Generation | Orchestrated prompting | `src/orchestration/prompt_orchestrator.py` |
| 4.3 | Fingerprint Generation | Unified fingerprints | `src/hdc/unified_fingerprint.py:UnifiedFingerprintGenerator` |
| 4.4 | Dual Library System | Reference + Active libraries | `src/fingerprint/dual_library_system.py` |
| **Section 5: Statistical Verification** | | | |
| 5.1 | Sequential Testing | SPRT implementation | `src/core/sequential.py:SequentialTest` |
| 5.2 | Hypothesis Testing | Likelihood ratios | `src/core/sequential.py:compute_likelihood_ratio()` |
| 5.3 | Error Bounds | Type I/II errors | `src/core/sequential.py:alpha, beta` |
| **Section 6: Memory-Efficient Execution** | | | |
| 6.1 | Segmented Loading | Layer-wise execution | `src/models/true_segment_execution.py:LayerSegmentExecutor` |
| 6.2 | Memory Bounds | 2GB cap enforcement | `src/executor/memory_bounded_executor.py` |
| 6.3 | Checkpoint Recovery | Resumable execution | `src/utils/run_rev_recovery.py:CheckpointManager` |

## Mathematical Notation

### Variable Mappings

| Paper Notation | Mathematical Definition | Code Variable | Location |
|---------------|------------------------|---------------|----------|
| **Hypervectors** | | | |
| $\mathbf{h} \in \{0,1\}^d$ | Binary hypervector | `hypervector: np.ndarray` | `HDCEncoder.encode_vector()` |
| $d$ | Dimension (10,000-100,000) | `self.dimension` | `HDCEncoder.__init__()` |
| $s$ | Sparsity level | `self.sparsity` | `AdaptiveSparsityEncoder` |
| $\oplus$ | XOR bundling | `np.logical_xor()` | `HDCEncoder.bundle()` |
| $\otimes$ | Binding operation | `np.roll()` | `HDCEncoder.bind()` |
| **Similarity** | | | |
| $d_H(\mathbf{h}_1, \mathbf{h}_2)$ | Hamming distance | `hamming.distance(hv1, hv2)` | `HammingDistanceOptimized` |
| $\text{sim}(\mathbf{h}_1, \mathbf{h}_2)$ | Cosine similarity | `1 - (d_H / d)` | `AdvancedSimilarity.cosine_binary()` |
| **SPRT** | | | |
| $\Lambda_n$ | Likelihood ratio | `self.likelihood_ratio` | `SequentialTest` |
| $\alpha$ | Type I error | `self.alpha = 0.05` | `SequentialTest.__init__()` |
| $\beta$ | Type II error | `self.beta = 0.05` | `SequentialTest.__init__()` |
| $\theta_0$ | Null hypothesis | `self.theta_0 = 0.5` | `SequentialTest` |
| $\theta_1$ | Alternative hypothesis | `self.theta_1 = 0.7` | `SequentialTest` |
| $A = \frac{1-\beta}{\alpha}$ | Upper boundary | `self.upper_threshold` | `SequentialTest._calculate_thresholds()` |
| $B = \frac{\beta}{1-\alpha}$ | Lower boundary | `self.lower_threshold` | `SequentialTest._calculate_thresholds()` |
| **Features** | | | |
| $\mathbf{f}_s \in \mathbb{R}^9$ | Syntactic features | `syntactic_features` | `SyntacticFeatures.extract()` |
| $\mathbf{f}_e \in \mathbb{R}^{20}$ | Semantic features | `semantic_features` | `SemanticFeatures.extract()` |
| $\mathbf{f}_b \in \mathbb{R}^9$ | Behavioral features | `behavioral_features` | `BehavioralFeatures.extract()` |
| $\mathbf{f}_a \in \mathbb{R}^{18}$ | Architectural features | `architectural_features` | `ArchitecturalFeatures.extract()` |
| **Restriction Sites** | | | |
| $L_i$ | Layer $i$ | `layer_idx` | `RestrictionSite.layer_idx` |
| $\delta_i$ | Divergence at layer $i$ | `behavioral_divergence` | `RestrictionSite.behavioral_divergence` |
| $\tau$ | Divergence threshold | `divergence_threshold = 0.3` | `identify_restriction_sites()` |

## Algorithms

### Algorithm 1: Restriction Site Identification

**Paper Pseudocode:**
```
Algorithm 1: IdentifyRestrictionSites
Input: Model M, Probes P
Output: Sites S

1: for each layer L_i in M do
2:    responses ← []
3:    for each probe p in P do
4:       r ← forward(M, p, stop_at=L_i)
5:       responses.append(r)
6:    divergence ← compute_divergence(responses)
7:    if divergence > τ then
8:       S.add(RestrictionSite(L_i, divergence))
9: return S
```

**Implementation:**
```python
# src/models/true_segment_execution.py
def identify_all_restriction_sites(self, probe_prompts: List[str]) -> List[RestrictionSite]:
    """Identify all restriction sites by profiling every layer"""
    sites = []
    
    for layer_idx in range(self.n_layers):
        # Get responses at this layer
        responses = []
        for prompt in probe_prompts:
            response = self._forward_to_layer(prompt, layer_idx)
            responses.append(response)
        
        # Calculate behavioral divergence
        divergence = self._calculate_divergence(responses)
        
        # Check if this is a restriction site
        if divergence > self.divergence_threshold:
            site = RestrictionSite(
                layer_idx=layer_idx,
                behavioral_divergence=divergence,
                confidence_score=self._calculate_confidence(divergence),
                site_type=self._classify_site_type(layer_idx)
            )
            sites.append(site)
    
    return sites
```

### Algorithm 2: HDC Encoding

**Paper Pseudocode:**
```
Algorithm 2: EncodeToHypervector
Input: Features F, Dimension d
Output: Hypervector h

1: h ← random_binary_vector(d)
2: for each feature f_i in F do
3:    h_i ← feature_to_hypervector(f_i, d)
4:    h ← h ⊕ h_i  // Bundle
5: h ← apply_sparsity(h, s)
6: return h
```

**Implementation:**
```python
# src/hdc/encoder.py
def encode_vector(self, features: np.ndarray) -> np.ndarray:
    """Encode feature vector to hypervector"""
    # Initialize random hypervector
    hypervector = self._generate_random_hypervector()
    
    # Encode each feature
    for i, feature_value in enumerate(features):
        # Map feature to hypervector
        feature_hv = self._encode_feature(i, feature_value)
        
        # Bundle with main hypervector
        hypervector = self.bundle(hypervector, feature_hv)
    
    # Apply sparsity
    hypervector = self._apply_sparsity(hypervector, self.sparsity)
    
    return hypervector
```

### Algorithm 3: SPRT Decision

**Paper Pseudocode:**
```
Algorithm 3: SequentialTest
Input: Samples X, Thresholds A, B
Output: Decision D

1: Λ ← 1  // Initialize likelihood ratio
2: for each sample x_i in X do
3:    Λ ← Λ × (p(x_i|θ₁) / p(x_i|θ₀))
4:    if Λ ≥ A then
5:       return ACCEPT_H1
6:    else if Λ ≤ B then
7:       return ACCEPT_H0
8: return CONTINUE
```

**Implementation:**
```python
# src/core/sequential.py
def add_sample(self, value: float) -> TestDecision:
    """Add sample and update test decision"""
    self.samples.append(value)
    
    # Update likelihood ratio
    lr_increment = self._compute_likelihood_ratio(value)
    self.likelihood_ratio *= lr_increment
    
    # Check decision boundaries
    if self.likelihood_ratio >= self.upper_threshold:
        self.decision = TestDecision.ACCEPT_H1
    elif self.likelihood_ratio <= self.lower_threshold:
        self.decision = TestDecision.ACCEPT_H0
    else:
        self.decision = TestDecision.CONTINUE
    
    return self.decision
```

## Figures and Visualizations

### Figure Generation Scripts

| Paper Figure | Description | Generation Code | Output Location |
|--------------|-------------|-----------------|-----------------|
| Figure 1 | Restriction site distribution | `visualize_restriction_sites()` | `experiments/plots/restriction_sites.png` |
| Figure 2 | HDC similarity heatmap | `plot_similarity_matrix()` | `experiments/plots/similarity_heatmap.png` |
| Figure 3 | SPRT stopping times | `plot_stopping_time_distributions()` | `experiments/plots/sprt_stopping.png` |
| Figure 4 | ROC curves | `generate_roc_curves()` | `experiments/plots/roc_curves.png` |
| Figure 5 | Memory usage over time | `plot_memory_profile()` | `experiments/plots/memory_usage.png` |
| Figure 6 | Feature importance | `plot_feature_importance()` | `experiments/plots/feature_importance.png` |

### Visualization Code Examples

**Restriction Site Visualization:**
```python
# experiments/visualization.py
def visualize_restriction_sites(sites: List[RestrictionSite], model_name: str):
    """Generate Figure 1 from paper"""
    import matplotlib.pyplot as plt
    
    layers = [s.layer_idx for s in sites]
    divergences = [s.behavioral_divergence for s in sites]
    
    plt.figure(figsize=(12, 6))
    plt.bar(layers, divergences, color='red', alpha=0.7)
    plt.axhline(y=0.3, color='k', linestyle='--', label='Threshold τ')
    plt.xlabel('Layer Index')
    plt.ylabel('Behavioral Divergence δ')
    plt.title(f'Restriction Sites in {model_name}')
    plt.legend()
    plt.savefig('experiments/plots/restriction_sites.png', dpi=300)
```

**SPRT Visualization:**
```python
# src/validation/stopping_time_analysis.py
def plot_stopping_time_distributions(self):
    """Generate Figure 3 from paper"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Empirical distribution
    ax1.hist(self.stopping_times, bins=30, alpha=0.7, color='blue')
    ax1.axvline(x=np.mean(self.stopping_times), color='red', 
                linestyle='--', label=f'Mean: {np.mean(self.stopping_times):.1f}')
    ax1.set_xlabel('Stopping Time (samples)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('SPRT Stopping Time Distribution')
    ax1.legend()
    
    # Comparison with fixed sample size
    fixed_size = 100
    ax2.bar(['SPRT', 'Fixed'], 
            [np.mean(self.stopping_times), fixed_size],
            color=['green', 'orange'])
    ax2.set_ylabel('Average Samples Required')
    ax2.set_title(f'Efficiency: {(1 - np.mean(self.stopping_times)/fixed_size)*100:.1f}% reduction')
    
    plt.tight_layout()
    plt.savefig('experiments/plots/sprt_stopping.png', dpi=300)
```

## Experimental Results

### Table Reproductions

| Paper Table | Description | Data Generation | Analysis Code |
|-------------|-------------|-----------------|---------------|
| Table 1 | Model families tested | `MODEL_CONFIGS` dict | `src/fingerprint/model_library.py` |
| Table 2 | Feature importance scores | `feature_importance` | `src/features/taxonomy.py:get_top_features()` |
| Table 3 | Classification accuracy | `classification_metrics` | `src/validation/empirical_metrics.py` |
| Table 4 | Memory usage comparison | `memory_profile` | `src/executor/memory_bounded_executor.py` |
| Table 5 | Inference time speedup | `timing_results` | `experiments/run_validation_suite.py` |

### Reproducing Paper Results

```bash
# Reproduce Table 3: Classification Accuracy
python experiments/run_validation_suite.py \
    --validation-experiments empirical \
    --validation-families gpt llama mistral yi \
    --validation-samples 1000

# Reproduce Figure 4: ROC Curves
python experiments/run_validation_suite.py \
    --validation-experiments empirical \
    --generate-validation-plots

# Reproduce Table 5: Performance Comparison
python experiments/benchmarking.py \
    --models llama-7b llama-70b llama-405b \
    --compare-with-baseline \
    --output results/table5_reproduction.csv
```

## Key Innovations Mapped

### 1. Biological Metaphor Implementation

Paper Concept → Code Reality:
- **Restriction Enzymes** → Layer analyzers that detect behavioral boundaries
- **Recognition Sequences** → Prompt patterns that trigger divergent responses  
- **Cleavage Sites** → Optimal segmentation points for memory-efficient loading
- **DNA Fingerprinting** → Hyperdimensional binary vectors unique to each model

### 2. Memory-Bounded Execution

Paper Theory → Implementation:
- **Theoretical Bound**: O(1) memory for any model size
- **Practical Implementation**: 2-4GB cap via `MemoryBoundedExecutor`
- **Segmentation Strategy**: Process at attention boundaries (`LayerSegmentExecutor`)
- **Checkpoint System**: Resume from any layer (`CheckpointManager`)

### 3. Statistical Guarantees

Paper Claims → Code Validation:
- **False Positive Rate**: α = 0.05 → `TestType.TYPE_I_ERROR`
- **False Negative Rate**: β = 0.05 → `TestType.TYPE_II_ERROR`  
- **Sample Efficiency**: 50-70% reduction → Validated in `SPRTAnalyzer`
- **Convergence**: Guaranteed by Wald's theorem → Implemented in `SequentialTest`

---

This mapping ensures complete traceability from theoretical concepts in the paper to practical implementation in the codebase.