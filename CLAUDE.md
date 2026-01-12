# REV - Restriction Enzyme Verification System

## CRITICAL MISTAKES TO AVOID ("Sin Bucket") - READ THIS FIRST!

### 1. ALWAYS Use --enable-prompt-orchestration
**WRONG:**
```bash
python run_rev.py /path/to/model  # Generates only 0-7 challenges - USELESS
python run_rev.py /path/to/model --challenges 50  # Still broken without orchestration
```

**CORRECT:**
```bash
python run_rev.py /path/to/model --enable-prompt-orchestration  # Generates 250+ challenges
```

Without orchestration, the pipeline generates insufficient challenges for meaningful fingerprinting.

### 2. Use FULL PATH to Model Directory
**WRONG:**
```bash
python run_rev.py EleutherAI/pythia-70m  # Tries cloud API, not local
python run_rev.py ~/models/pythia-70m  # May fail on relative path
```

**CORRECT:**
```bash
python run_rev.py /Users/rohanvinaik/LLM_models/pythia-70m --enable-prompt-orchestration
```

The path MUST contain `config.json` for the model to be recognized.

### 3. For HuggingFace Cache, Use SNAPSHOT Path
**WRONG:**
```bash
python run_rev.py /path/to/models--EleutherAI--pythia-70m  # Parent directory
```

**CORRECT:**
```bash
python run_rev.py /path/to/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42 --enable-prompt-orchestration
```

Find the snapshot path with: `find ~/LLM_models -name "config.json" | grep pythia`

### 4. Reference Builds Require BOTH Flags
**WRONG:**
```bash
python run_rev.py /path/to/model --build-reference  # Generates 0 challenges!
```

**CORRECT:**
```bash
python run_rev.py /path/to/model --build-reference --enable-prompt-orchestration
```

### 5. Build References with SMALLEST Model in Family
**WRONG:**
```bash
python run_rev.py /path/to/llama-70b --build-reference --enable-prompt-orchestration  # Takes days
```

**CORRECT:**
```bash
python run_rev.py /path/to/llama-2-7b-hf --build-reference --enable-prompt-orchestration  # Reference for all Llama
```

### 6. Never Run Multiple Reference Builds Concurrently
Reference builds are resource-intensive. Run ONE at a time.

---

## Overview

**REV (Restriction Enzyme Verification)** identifies and verifies neural network models through behavioral analysis, inspired by restriction enzymes that cut DNA at specific sites. REV identifies "behavioral boundaries" (restriction sites) where significant behavioral changes occur between transformer layers.

**Key Innovation**: Stream 68GB+ models on 64GB systems by loading weights layer-by-layer from disk (never fully in memory). Model weights remain secret throughout - only behavioral fingerprints are extracted.

---

## Core Architecture

```
Light Probe (10-20% layers)  →  Reference Match  →  Targeted Execution
         ↓                            ↓                     ↓
   Variance Profile              Dual Library          REV Sites Only
   Family Detection           (Ref + Active)         (15-20x speedup)
```

**Key Components:**
- `run_rev.py` - Single entry point (users run ONLY this)
- `fingerprint_library/reference_library.json` - Deep behavioral topology (one per model family)
- `fingerprint_library/active_library.json` - All other run fingerprints
- `src/models/true_segment_execution.py` - Layer-by-layer streaming engine (2GB memory cap)

---

## Quick Start Commands

### Standard Model Verification
```bash
python run_rev.py /full/path/to/model --enable-prompt-orchestration
```

### Build Reference (One-Time Per Family)
```bash
# Use SMALLEST model in family (e.g., pythia-70m, not pythia-12b)
python run_rev.py /path/to/smallest-model --build-reference --enable-prompt-orchestration
```

### Multi-Model Comparison
```bash
python run_rev.py /path/model1 /path/model2 --enable-prompt-orchestration
```

### Verify Reference Library
```bash
python -c "import json; print({k: v.get('challenges_processed', 0) for k,v in json.load(open('fingerprint_library/reference_library.json'))['fingerprints'].items()})"
```

---

## Key Concepts

### Restriction Sites
Behavioral boundaries in transformer layers where significant changes occur. Analogous to restriction enzyme cut sites in DNA. Identified through variance analysis, not architecture inspection.

### Dual Library System
| Library | Purpose | Updates |
|---------|---------|---------|
| Reference | Deep behavioral topology from smallest model in family | Once per family |
| Active | All runtime fingerprints | Every run |

Reference enables 15-20x speedup: instead of probing all layers, probe only known restriction sites.

### Light Probe → Match → Target Workflow
1. **Light Probe**: Sample 10-20% of layers, measure variance profile
2. **Reference Match**: Compare against known families (Llama, GPT, Pythia, etc.)
3. **Targeted Execution**: If match >30% confidence, probe only restriction sites

### 7 Prompt Orchestration Systems
The pipeline auto-generates 250-400+ prompts using:
- PoT (30%) - Proof-of-Training behavioral probes
- KDF (20%) - Adversarial/security testing
- Evolutionary (20%) - Genetic optimization
- Dynamic (20%) - Template synthesis
- Hierarchical (10%) - Taxonomy exploration

---

## Quick Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| <250 challenges generated | Missing orchestration flag | Add `--enable-prompt-orchestration` |
| Probe times <10ms | Mock responses (not real execution) | Check for `[SEGMENT-EXEC]` in logs |
| Identical divergence values | Deep analysis failing | Verify model path has config.json |
| "NoneType" error | Invalid model path | Use absolute path to directory with config.json |
| No reference speedup | Reference missing or incomplete | Rebuild with both flags |
| System slowdown | Concurrent builds | Run ONE reference build at a time |

### Verify Real Execution
Look for these log patterns (real execution):
```
[SEGMENT-EXEC] Loaded weights for layer X
[DIVERGENCE-DEBUG] Computing signature for layer X
Probe times: 50-150ms (not <10ms)
Divergence values: varying (not all identical)
```

---

## Expected Results

| Model Size | Reference Build Time | Challenges | Use Case |
|------------|---------------------|------------|----------|
| 70M (pythia-70m) | 20 min | 260+ | Reference baseline |
| 7B (llama-7b) | 2-3 hours | 280+ | Reference baseline |
| 160M (with ref) | 8-12 min | ~50 | Uses reference speedup |
| 70B (with ref) | 2-4 hours | ~50 | Uses reference speedup |

---

## Project Structure

```
run_rev.py                     # ONLY entry point
fingerprint_library/
├── reference_library.json     # Deep behavioral topology
└── active_library.json        # All runtime fingerprints
src/
├── models/                    # Streaming execution engine
├── fingerprint/               # Dual library system
├── challenges/                # 7 prompt orchestration systems
├── hdc/                       # Hyperdimensional computing
├── core/                      # SPRT statistical testing
└── analysis/                  # Behavioral analysis
```

---

## Dependencies

```bash
pip install torch transformers safetensors numpy scipy
```

---

## Running Tests

```bash
make test                # All tests
make test-unit          # Unit tests only
python run_rev.py --help  # Verify installation
```

---

## REV vs HBT_Paper

These are **separate but related** projects:
- **REV** (`/Users/rohanvinaik/REV`): Practical tool for verifying local models on disk
- **HBT_Paper** (`/Users/rohanvinaik/HBT_Paper`): Academic research with additional theory (VMCI, Holographic Construction)

REV handles execution; HBT handles interpretation. Both can be used independently.
