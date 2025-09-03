# REV System Architecture

## Table of Contents
- [Overview](#overview)
- [Module Dependency Graph](#module-dependency-graph)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Class Hierarchy](#class-hierarchy)
- [Key Workflows](#key-workflows)
- [Component Details](#component-details)

## Overview

The REV (Restriction Enzyme Verification) System implements a novel approach to LLM verification through biological-inspired restriction site analysis. The architecture consists of five main layers:

1. **API Layer**: REST/WebSocket/GraphQL interfaces
2. **Orchestration Layer**: Prompt generation and strategy management
3. **Processing Layer**: Model inference and fingerprint extraction
4. **Analysis Layer**: Statistical testing and verification
5. **Storage Layer**: Dual library system and caching

## Module Dependency Graph

```mermaid
graph TB
    subgraph "API Layer"
        REST[REST API<br/>src/api/rest_service.py]
        WS[WebSocket<br/>Real-time Updates]
        GQL[GraphQL<br/>Flexible Queries]
    end

    subgraph "Main Pipeline"
        REV[REVUnified<br/>run_rev.py]
        PIPE[REVPipeline<br/>src/rev_pipeline.py]
    end

    subgraph "Orchestration"
        ORCH[PromptOrchestrator<br/>src/orchestration/]
        POT[PoT Generator<br/>src/challenges/pot_challenge_generator.py]
        KDF[KDF Prompts<br/>src/challenges/kdf_prompts.py]
        EVO[Evolutionary<br/>src/orchestration/evolutionary_prompts.py]
    end

    subgraph "Feature Extraction"
        TAX[FeatureTaxonomy<br/>src/features/taxonomy.py]
        AUTO[AutomaticFeaturizer<br/>src/features/automatic_featurizer.py]
        LEARN[LearnedFeatures<br/>src/features/learned_features.py]
    end

    subgraph "HDC Processing"
        HDC[HDC Encoder<br/>src/hdc/encoder.py]
        ADAPT[AdaptiveSparsity<br/>src/hdc/adaptive_encoder.py]
        UNIFIED[UnifiedFingerprint<br/>src/hdc/unified_fingerprint.py]
    end

    subgraph "Analysis"
        SPRT[Sequential Test<br/>src/core/sequential.py]
        HAMMING[Hamming Distance<br/>src/hypervector/hamming.py]
        SIM[Similarity Metrics<br/>src/hypervector/similarity.py]
    end

    subgraph "Storage"
        LIB[Model Library<br/>src/fingerprint/model_library.py]
        DUAL[Dual Library<br/>src/fingerprint/dual_library_system.py]
        CACHE[Redis Cache<br/>External]
    end

    subgraph "Infrastructure"
        ERROR[Error Handling<br/>src/utils/error_handling.py]
        LOG[Logging<br/>src/utils/logging_config.py]
        REPRO[Reproducibility<br/>src/utils/reproducibility.py]
    end

    REST --> REV
    WS --> REV
    GQL --> REV
    
    REV --> PIPE
    REV --> ORCH
    
    ORCH --> POT
    ORCH --> KDF
    ORCH --> EVO
    
    PIPE --> TAX
    TAX --> AUTO
    TAX --> LEARN
    
    AUTO --> HDC
    LEARN --> HDC
    
    HDC --> ADAPT
    HDC --> UNIFIED
    
    UNIFIED --> SPRT
    UNIFIED --> HAMMING
    UNIFIED --> SIM
    
    SPRT --> LIB
    HAMMING --> LIB
    SIM --> LIB
    
    LIB --> DUAL
    DUAL --> CACHE
    
    REV --> ERROR
    REV --> LOG
    REV --> REPRO
    
    style REST fill:#f9f,stroke:#333,stroke-width:2px
    style REV fill:#bbf,stroke:#333,stroke-width:4px
    style HDC fill:#bfb,stroke:#333,stroke-width:2px
    style SPRT fill:#fbf,stroke:#333,stroke-width:2px
```

## Data Flow Pipeline

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Pipeline
    participant Orchestrator
    participant Model
    participant HDC
    participant Analyzer
    participant Library

    User->>API: Submit model path
    API->>Pipeline: Initialize analysis
    
    Pipeline->>Library: Check existing fingerprint
    Library-->>Pipeline: Return confidence & strategy
    
    alt Confidence < 0.5 (Unknown model)
        Pipeline->>Pipeline: Enable deep analysis
        Note over Pipeline: 6-24 hour profiling
    end
    
    Pipeline->>Orchestrator: Generate prompts
    Orchestrator->>Orchestrator: Select prompt strategy
    Note over Orchestrator: 7 specialized systems
    
    loop For each challenge
        Orchestrator->>Model: Send prompt
        Model-->>Pipeline: Return response
        Pipeline->>Pipeline: Extract features
        Note over Pipeline: 56 principled features
    end
    
    Pipeline->>HDC: Encode to hypervector
    Note over HDC: 10,000-100,000 dimensions
    HDC->>HDC: Apply adaptive sparsity
    
    HDC->>Analyzer: Statistical testing
    Note over Analyzer: SPRT, Hamming distance
    
    Analyzer->>Library: Store fingerprint
    Library->>Library: Update dual libraries
    
    Analyzer-->>API: Return results
    API-->>User: Model verification complete
```

## Class Hierarchy

```mermaid
classDiagram
    class REVUnified {
        +process_model(path, challenges)
        +cleanup()
        -_initialize_components()
        -_collect_validation_metrics()
    }
    
    class REVPipeline {
        +run(model_path, config)
        +segment_execution()
        -_load_segments()
        -_process_segment()
    }
    
    class PromptOrchestrator {
        +generate_prompts(strategy)
        +get_next_prompt()
        -_select_system()
    }
    
    class FeatureTaxonomy {
        +extract_all_features()
        +get_concatenated_features()
        +update_importance_scores()
    }
    
    class HDCEncoder {
        +encode_vector(features)
        +decode_vector(hypervector)
        -_apply_bundling()
        -_apply_binding()
    }
    
    class UnifiedFingerprint {
        +generate(responses, pathways)
        +compare(other)
        +to_dict()
    }
    
    class SequentialTest {
        +add_sample(value)
        +get_decision()
        -_compute_likelihood_ratio()
    }
    
    class ModelLibrary {
        +add_fingerprint(fingerprint)
        +search(query)
        +get_reference(family)
    }
    
    REVUnified --> REVPipeline
    REVUnified --> PromptOrchestrator
    REVPipeline --> FeatureTaxonomy
    FeatureTaxonomy --> HDCEncoder
    HDCEncoder --> UnifiedFingerprint
    UnifiedFingerprint --> SequentialTest
    SequentialTest --> ModelLibrary
```

## Key Workflows

### 1. Model Analysis Workflow

```mermaid
flowchart LR
    Start([Start]) --> Load[Load Model]
    Load --> Check{Cached?}
    Check -->|Yes| UseCache[Use Cached Fingerprint]
    Check -->|No| Identify[Identify Architecture]
    
    Identify --> Strategy{Confidence?}
    Strategy -->|High >85%| Targeted[Targeted Testing<br/>15-20 challenges]
    Strategy -->|Medium 60-85%| Adaptive[Adaptive Testing<br/>30-50 challenges]
    Strategy -->|Low <60%| Deep[Deep Analysis<br/>100+ challenges]
    
    Targeted --> Generate[Generate Prompts]
    Adaptive --> Generate
    Deep --> Generate
    
    Generate --> Process[Process Responses]
    Process --> Extract[Extract Features]
    Extract --> Encode[HDC Encoding]
    Encode --> Test[Statistical Testing]
    Test --> Store[Store in Library]
    
    UseCache --> Result([Result])
    Store --> Result
```

### 2. Feature Extraction Workflow

```mermaid
flowchart TB
    Response[Model Response] --> Categories{Feature Categories}
    
    Categories --> Syntactic[Syntactic Features<br/>9 features]
    Categories --> Semantic[Semantic Features<br/>20 features]
    Categories --> Behavioral[Behavioral Features<br/>9 features]
    Categories --> Architectural[Architectural Features<br/>18 features]
    
    Syntactic --> Concat[Concatenate<br/>56 total features]
    Semantic --> Concat
    Behavioral --> Concat
    Architectural --> Concat
    
    Concat --> Selection{Feature Selection}
    Selection --> MI[Mutual Information]
    Selection --> LASSO[LASSO]
    Selection --> Elastic[Elastic Net]
    
    MI --> Ensemble[Ensemble Scoring]
    LASSO --> Ensemble
    Elastic --> Ensemble
    
    Ensemble --> Top[Select Top 100]
    Top --> Weight[Weight by Importance]
    Weight --> HDC[HDC Encoding]
```

### 3. Error Recovery Workflow

```mermaid
stateDiagram-v2
    [*] --> Normal
    
    Normal --> Error: Exception
    Error --> Analyze: Analyze Error Type
    
    Analyze --> MemoryError: OOM
    Analyze --> NetworkError: Timeout
    Analyze --> GPUError: CUDA Error
    
    MemoryError --> ReduceSize: Reduce Batch/Segment
    ReduceSize --> Retry
    
    NetworkError --> UseCache: Switch to Cache
    UseCache --> Retry
    
    GPUError --> SwitchCPU: Fallback to CPU
    SwitchCPU --> Retry
    
    Retry --> Normal: Success
    Retry --> CircuitOpen: Too Many Failures
    
    CircuitOpen --> Wait: Recovery Timeout
    Wait --> HalfOpen: Test Recovery
    HalfOpen --> Normal: Success
    HalfOpen --> CircuitOpen: Failure
```

## Component Details

### 1. Restriction Sites (Biological Metaphor)

The core innovation maps LLM layers to restriction enzymes:

```python
# Restriction Site = High behavioral divergence point
RestrictionSite:
  - layer_idx: int          # Layer number (0-based)
  - divergence: float        # Behavioral divergence score [0,1]
  - confidence: float        # Statistical confidence [0,1]
  - site_type: str          # "attention", "mlp", "norm"
```

### 2. Hyperdimensional Computing (HDC)

Encodes features into high-dimensional binary vectors:

```python
# HDC Operations
Bundling: XOR operation for combining vectors
Binding: Rotation for associating vectors
Similarity: Hamming distance for comparison

Dimensions: 10,000 - 100,000
Sparsity: 0.5% - 15% (adaptive)
```

### 3. Sequential Testing (SPRT)

Implements Wald's Sequential Probability Ratio Test:

```python
# SPRT Parameters
α = 0.05  # Type I error (false positive)
β = 0.05  # Type II error (false negative)
θ₀ = 0.5  # Null hypothesis
θ₁ = 0.7  # Alternative hypothesis

# Decision boundaries
A = (1-β)/α     # Upper boundary (accept H₁)
B = β/(1-α)     # Lower boundary (accept H₀)
```

### 4. Dual Library System

Maintains two fingerprint libraries:

```python
Reference Library:
  - Location: fingerprint_library/reference_library.json
  - Purpose: Deep behavioral baseline
  - Updates: Rare (new families only)
  - Size: ~100 reference models

Active Library:
  - Location: fingerprint_library/active_library.json  
  - Purpose: Continuous learning
  - Updates: Every successful run
  - Size: Unlimited (auto-pruned)
```

### 5. Prompt Orchestration

Seven specialized prompt generation systems:

| System | Weight | Focus | Prompts/Min |
|--------|--------|-------|------------|
| PoT | 30% | Behavioral boundaries | 10 |
| KDF | 20% | Security/adversarial | 5 |
| Evolutionary | 20% | Genetic optimization | 3 |
| Dynamic | 20% | Template synthesis | 15 |
| Hierarchical | 10% | Taxonomical | 8 |
| Predictor | - | Effectiveness scoring | - |
| Profiler | - | Pattern analysis | - |

### 6. Memory Management

Segmented execution for massive models:

```python
Segment Execution:
  - Load one layer at a time
  - Process at attention boundaries  
  - Memory cap: 2-4GB per segment
  - Supports 405B+ parameter models
  
Memory Recovery:
  - Automatic GPU cache clearing
  - Dynamic segment size adjustment
  - CPU fallback on GPU OOM
  - Checkpoint-based resumption
```

## Performance Characteristics

| Component | Latency | Memory | GPU Required |
|-----------|---------|--------|--------------|
| Feature Extraction | <100ms | 100MB | No |
| HDC Encoding | <50ms | 500MB | No |
| SPRT Testing | <10ms | 10MB | No |
| Hamming Distance | <1ms | 50MB | No |
| Deep Analysis | 6-24h | 2-4GB | Yes |
| API Response | <500ms | 200MB | No |

## Scalability

The system scales across multiple dimensions:

```mermaid
graph LR
    subgraph Horizontal
        API1[API Instance 1] 
        API2[API Instance 2]
        API3[API Instance 3]
        LB[Load Balancer] --> API1
        LB --> API2
        LB --> API3
    end
    
    subgraph Vertical
        Small[7B Model<br/>2GB RAM]
        Medium[70B Model<br/>4GB RAM]
        Large[405B Model<br/>4GB RAM]
    end
    
    subgraph Storage
        Redis[(Redis<br/>Cache)]
        Postgres[(PostgreSQL<br/>Metadata)]
        S3[(S3<br/>Models)]
    end
```

## Security Architecture

```mermaid
flowchart TB
    Client[Client] --> WAF[Web Application Firewall]
    WAF --> LB[Load Balancer/TLS]
    
    LB --> Auth{Authentication}
    Auth -->|Invalid| Reject[Reject 401]
    Auth -->|Valid| RateLimit{Rate Limiting}
    
    RateLimit -->|Exceeded| Throttle[Throttle 429]
    RateLimit -->|OK| API[API Gateway]
    
    API --> Authz{Authorization}
    Authz -->|Denied| Forbid[Forbid 403]
    Authz -->|Allowed| Process[Process Request]
    
    Process --> Audit[Audit Log]
    Process --> Monitor[Security Monitor]
```

## Deployment Architecture

```mermaid
graph TB
    subgraph Production
        K8S[Kubernetes Cluster]
        K8S --> Pods[REV Pods<br/>3-10 replicas]
        K8S --> GPU[GPU Nodes<br/>For inference]
        K8S --> Storage[Persistent Storage<br/>500GB+]
    end
    
    subgraph Monitoring
        Prom[Prometheus<br/>Metrics]
        Graf[Grafana<br/>Dashboards]
        Jaeger[Jaeger<br/>Tracing]
        ELK[ELK Stack<br/>Logging]
    end
    
    subgraph Data
        Primary[(PostgreSQL<br/>Primary)]
        Replica[(PostgreSQL<br/>Replica)]
        RedisCluster[(Redis Cluster<br/>3 nodes)]
    end
    
    Pods --> Prom
    Pods --> Jaeger
    Pods --> ELK
    
    Pods --> Primary
    Primary --> Replica
    Pods --> RedisCluster
```

---

This architecture enables REV to verify models 15-20x faster than traditional methods while maintaining cryptographic integrity through Merkle trees and supporting models that exceed available memory.