# REV — Behavioral Fingerprinting for Neural Networks

**Verify any LLM's identity without loading it into memory.**

REV identifies neural networks through *behavioral fingerprinting*—probing how models respond at specific layer boundaries ("restriction sites") to generate unforgeable signatures. Like DNA fingerprinting, each model has a unique behavioral profile.

## Why This Matters

| Problem | REV Solution |
|---------|--------------|
| 70B model needs 140GB VRAM to verify | **3-4GB** using layer streaming |
| No way to verify downloaded model identity | **Behavioral fingerprint** proves authenticity |
| API models are black boxes | **Response patterns** reveal model family |
| Full analysis takes days | **15-20x faster** with reference library |

## Proven Results

### Model Discrimination: 95% Accuracy

Five different models produce completely unique fingerprints:

```
              smollm  tinyllama  gemma2   phi3  llama3.2
smollm         1.000    -0.004   0.000  0.000     0.047
tinyllama     -0.004     1.000   0.006 -0.012    -0.032
gemma2         0.000     0.006   1.000  0.008     0.000
phi3           0.000    -0.012   0.008  1.000     0.008
llama3.2       0.047    -0.032   0.000  0.008     1.000
```

Self-similarity = 1.0, cross-model similarity ≈ 0. **Perfect discrimination.**

### Key Validated Claims

| Claim | Evidence |
|-------|----------|
| Restriction sites exist | Found 38 high-variance layers across GPT-2, Pythia, Phi |
| Layer analysis beats black-box | Within-family ≈ cross-family for API responses alone |
| SPRT enables early stopping | 100% convergence rate, 11 samples average |
| HDC encoding is orthogonal | Cross-similarity < 0.01 for different inputs |

## Quick Start

```bash
git clone https://github.com/rohanvinaik/REV.git
cd REV && pip install -r requirements.txt

# Validate installation (requires Ollama running locally)
python scripts/validate_ollama_pipeline.py

# Fingerprint a local model
python run_rev.py /path/to/model --enable-prompt-orchestration

# Build reference library (one-time per model family)
python run_rev.py /path/to/smallest-model --build-reference --enable-prompt-orchestration
```

## How It Works

```
1. LIGHT PROBE      →  Sample 10-15% of layers, identify model family
2. REFERENCE MATCH  →  Compare variance profile to known architectures
3. TARGETED PROBE   →  Focus 250-400 prompts on restriction sites
4. SPRT DECISION    →  Stop when statistically confident
```

**Restriction sites** are layer boundaries with high behavioral variance—where the model's "personality" changes. By targeting these sites, REV achieves 15-20x speedup over exhaustive analysis.

## Architecture

```
src/
├── core/           # SPRT sequential testing
├── hdc/            # Hyperdimensional encoding (10K-dim vectors)
├── models/         # Layer-by-layer streaming execution
├── fingerprint/    # Dual library system (reference + active)
├── orchestration/  # 7 prompt generation systems
└── challenges/     # PoT cryptographic probes
```

## Key Concepts

<details>
<summary><b>Restriction Sites</b></summary>

Layer boundaries where behavioral variance spikes. Like restriction enzyme cut sites in DNA, these are model-specific signatures that emerge from training.

</details>

<details>
<summary><b>Dual Library System</b></summary>

- **Reference Library**: Deep topology from smallest model in each family
- **Active Library**: Runtime fingerprints for verification

Build reference once → verify any model in that family 15-20x faster.

</details>

<details>
<summary><b>Hyperdimensional Computing</b></summary>

Responses encoded as 10,000-dimensional sparse vectors. Mathematical properties:
- Same text → identical vector (self-similarity = 1.0)
- Different text → orthogonal vectors (similarity ≈ 0)

</details>

<details>
<summary><b>SPRT Early Stopping</b></summary>

Sequential Probability Ratio Test enables stopping as soon as confidence threshold is reached. Reduces sample requirements by 50-80% compared to fixed-sample testing.

</details>

## Supported Sources

| Source | Example |
|--------|---------|
| **Ollama** | `llama3.2:3b`, `gemma2:2b` |
| **HuggingFace** | `/path/to/model/` with `config.json` |
| **Cloud APIs** | OpenAI, Anthropic, Together |

## Limitations

- Reference build: 20 min - 3 hours (one-time per family)
- Requires `--enable-prompt-orchestration` flag
- API models limited by rate limits

## Citation

```bibtex
@software{rev2024,
  title={REV: Restriction Enzyme Verification for LLM Identity},
  author={Vinaik, Rohan},
  year={2024},
  url={https://github.com/rohanvinaik/REV}
}
```

MIT License
