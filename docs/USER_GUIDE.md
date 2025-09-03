# REV System User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Model Compatibility](#model-compatibility)
4. [Common Use Cases](#common-use-cases)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)
7. [FAQ](#faq)

## Installation

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10 WSL2 | macOS 14+, Ubuntu 22.04+ | Native Apple Silicon support |
| **RAM** | 16 GB | 64 GB | 2-4GB per model segment |
| **Storage** | 100 GB | 500 GB | For model storage |
| **GPU** | None (CPU mode) | NVIDIA RTX 3090+, Apple M1 Pro+ | CUDA 12.1+ or Metal |
| **Python** | 3.9+ | 3.10+ | 3.11 recommended |

### macOS Installation (Apple Silicon)

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python and dependencies
brew install python@3.11
brew install git git-lfs
brew install redis  # For caching

# 3. Clone repository
git clone https://github.com/rohanvinaik/REV.git
cd REV

# 4. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 5. Install PyTorch with Metal support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# 6. Install REV dependencies
pip install -r requirements.txt

# 7. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'Metal available: {torch.backends.mps.is_available()}')"
```

### Linux Installation (CUDA)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and essentials
sudo apt install python3.10 python3.10-venv python3-pip git git-lfs

# 3. Install CUDA (if not installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-12-1

# 4. Clone repository
git clone https://github.com/rohanvinaik/REV.git
cd REV

# 5. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 6. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Install REV dependencies
pip install -r requirements.txt

# 8. Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Windows Installation (WSL2)

```powershell
# 1. Enable WSL2 (PowerShell as Administrator)
wsl --install
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04

# 2. Install NVIDIA drivers for WSL2
# Download from: https://developer.nvidia.com/cuda/wsl

# 3. In WSL2 Ubuntu terminal:
# Follow Linux installation steps above
```

### Docker Installation (All Platforms)

```bash
# 1. Install Docker Desktop
# macOS: https://docs.docker.com/desktop/mac/install/
# Linux: https://docs.docker.com/engine/install/
# Windows: https://docs.docker.com/desktop/windows/install/

# 2. Pull and run REV container
docker pull revhub/rev-system:latest
docker run -it --gpus all -v $(pwd)/models:/app/models revhub/rev-system:latest

# Or build locally:
docker build -f deployment/Dockerfile -t rev-system .
docker run -it --gpus all rev-system
```

## Quick Start

### Basic Model Verification

```bash
# 1. Verify a local model (most common use case)
python run_rev.py /path/to/your/model --challenges 50

# 2. Verify with principled features (recommended)
python run_rev.py /path/to/model \
    --enable-principled-features \
    --challenges 100

# 3. Multiple models comparison
python run_rev.py model1/ model2/ model3/ \
    --challenges 50 \
    --output comparison.json

# 4. Quick diagnostic (5 challenges)
python run_rev.py /path/to/model --quick
```

### Finding Your Models

#### HuggingFace Cache Models

```bash
# Find downloaded models
find ~/.cache/huggingface -name "config.json" -type f | head -5

# Example paths:
# ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/XXX/
# ~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/XXX/

# Use the snapshot directory containing config.json:
python run_rev.py ~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
```

#### Standard Model Directories

```bash
# Common locations
~/models/              # Personal models
/opt/models/          # System-wide models
/mnt/models/          # Mounted storage

# Verify directory structure
ls -la /path/to/model/
# Should contain: config.json, pytorch_model.bin or model.safetensors
```

## Model Compatibility

### Fully Supported Models

| Model Family | Versions | Optimal Settings | Notes |
|--------------|----------|------------------|-------|
| **Llama** | 1, 2, 3, 3.1, 3.2, 3.3 | `--challenges 50` | All sizes: 7B-405B |
| **GPT** | GPT-2, GPT-Neo, GPT-J | `--challenges 30` | API or local |
| **Mistral** | 7B, 8x7B, 8x22B | `--challenges 40` | Mixture of Experts |
| **Yi** | 6B, 34B | `--challenges 60` | Extended context |
| **Falcon** | 7B, 40B, 180B | `--challenges 50` | RefinedWeb trained |
| **Pythia** | 70M-12B | `--challenges 20` | Research models |
| **Gemma** | 2B, 7B | `--challenges 30` | Google models |
| **Qwen** | 0.5B-72B | `--challenges 40` | Alibaba models |

### Architecture Requirements

```python
# Model must have:
1. config.json with architecture info
2. PyTorch weights (.bin or .safetensors)
3. tokenizer files

# Supported architectures:
- LlamaForCausalLM
- GPT2LMHeadModel  
- GPTNeoXForCausalLM
- MistralForCausalLM
- FalconForCausalLM
- BloomForCausalLM
- Any AutoModelForCausalLM compatible
```

### Adding Custom Model Support

```python
# 1. Create model adapter in src/models/adapters/
class CustomModelAdapter:
    def __init__(self, model_path):
        self.config = load_config(model_path)
        
    def get_num_layers(self):
        return self.config.num_hidden_layers
        
    def forward_to_layer(self, input_ids, layer_idx):
        # Custom forward implementation
        pass

# 2. Register in model factory
ADAPTER_REGISTRY['custom_arch'] = CustomModelAdapter

# 3. Use with REV
python run_rev.py /path/to/custom/model --model-adapter custom_arch
```

## Common Use Cases

### 1. Building Reference Library (First Time Setup)

```bash
# Build reference for a model family (do once per family)
# Use smallest model in family, takes 6-24 hours

# Llama family
python run_rev.py ~/models/llama-2-7b \
    --build-reference \
    --enable-prompt-orchestration \
    --challenges 100

# GPT family  
python run_rev.py ~/models/gpt2 \
    --build-reference \
    --enable-prompt-orchestration \
    --challenges 100
```

### 2. Fast Verification Using Reference

```bash
# After reference exists, large models run 15-20x faster

# Verify Llama 70B (uses 7B reference)
python run_rev.py ~/models/llama-2-70b --challenges 50
# Runtime: ~2 hours instead of 37 hours

# Verify Llama 405B
python run_rev.py ~/models/llama-3.1-405b-fp8 --challenges 100
# Runtime: ~3 hours with 4GB RAM
```

### 3. Adversarial Testing

```bash
# Test model robustness
python run_rev.py /path/to/model \
    --adversarial \
    --adversarial-ratio 0.3 \
    --adversarial-types jailbreak prompt_injection \
    --challenges 100
```

### 4. Production Deployment

```bash
# Start API server
python -m uvicorn src.api.rest_service:app --host 0.0.0.0 --port 8000

# Verify via API
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/llama-70b", "challenges": 50}'
```

### 5. Batch Processing

```bash
# Process multiple models
for model in /models/*; do
    python run_rev.py "$model" \
        --challenges 30 \
        --output "results/$(basename $model).json"
done

# Parallel processing
parallel -j 4 python run_rev.py {} --challenges 30 ::: /models/*
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
Killed (system OOM killer)
```

**Solutions:**
```bash
# Reduce memory usage
python run_rev.py /path/to/model \
    --memory-limit 2.0 \
    --challenges 10

# Force CPU mode
python run_rev.py /path/to/model --device cpu

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Monitor memory
watch -n 1 nvidia-smi  # GPU
watch -n 1 free -h     # System
```

#### 2. Model Loading Errors

**Symptoms:**
```
FileNotFoundError: config.json not found
KeyError: 'model.embed_tokens.weight'
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**
```bash
# Verify model structure
ls -la /path/to/model/
# Must have: config.json, pytorch_model.bin or model.safetensors

# Use correct path (snapshot for HF cache)
find ~/.cache/huggingface -name "config.json" | grep model_name

# Convert model format if needed
python scripts/convert_model.py --input old_model/ --output new_model/

# Verify config
python -c "import json; print(json.load(open('/path/to/model/config.json'))['architectures'])"
```

#### 3. Slow Performance

**Symptoms:**
```
Processing: 0.1 samples/sec
ETA: 48 hours remaining
High CPU, low GPU usage
```

**Solutions:**
```bash
# Use reference library (15-20x speedup)
python run_rev.py model --use-reference

# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0
python run_rev.py model --device cuda:0

# Reduce challenge count
python run_rev.py model --challenges 10  # Instead of 100

# Profile bottlenecks
python run_rev.py model --profile --debug

# Use optimized settings
python run_rev.py model \
    --batch-size 8 \
    --num-workers 4 \
    --pin-memory
```

#### 4. API Connection Issues

**Symptoms:**
```
ConnectionRefusedError: [Errno 111] Connection refused
requests.exceptions.Timeout
Circuit breaker is OPEN
```

**Solutions:**
```bash
# Check service status
curl http://localhost:8000/health

# Restart services
docker-compose restart rev-api

# Check logs
docker logs rev-api --tail 100

# Increase timeout
export REV_API_TIMEOUT=300

# Reset circuit breaker
curl -X POST http://localhost:8000/admin/reset-circuit-breaker
```

## Performance Tuning

### Memory Optimization

```bash
# 1. Optimal segment size based on RAM
16GB RAM: --segment-size 32 --memory-limit 2.0
32GB RAM: --segment-size 64 --memory-limit 4.0
64GB RAM: --segment-size 128 --memory-limit 8.0

# 2. Enable memory mapping for large models
python run_rev.py huge_model/ --enable-mmap

# 3. Use checkpoint recovery
python run_rev.py model/ \
    --checkpoint-dir checkpoints/ \
    --checkpoint-interval 10
```

### GPU Optimization

```bash
# 1. Multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_rev.py model/ --data-parallel

# 2. Mixed precision (faster, less memory)
python run_rev.py model/ --mixed-precision fp16

# 3. Optimize for specific GPU
# A100/H100
python run_rev.py model/ --torch-compile --use-flash-attention

# RTX 3090/4090
python run_rev.py model/ --batch-size 4 --gradient-checkpointing

# Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
python run_rev.py model/ --device mps
```

### Network/API Optimization

```bash
# 1. Enable caching
redis-server &
python run_rev.py model/ --enable-cache --cache-ttl 3600

# 2. Batch requests
python run_rev.py model/ --batch-size 32

# 3. Connection pooling
export REV_MAX_CONNECTIONS=100
export REV_CONNECTION_TIMEOUT=30

# 4. Use local models instead of API
python run_rev.py /local/path/to/model  # Not API
```

### Profiling and Monitoring

```bash
# 1. Enable profiling
python run_rev.py model/ --profile --profile-output profile.json

# 2. Memory profiling
python -m memory_profiler run_rev.py model/

# 3. GPU profiling
nsys profile python run_rev.py model/
nvidia-smi dmon -s mu -d 1

# 4. System monitoring
htop  # CPU/RAM
iotop  # Disk I/O
iftop  # Network
```

## FAQ

### Q: How long does verification take?

**A:** Depends on model size and reference availability:
- With reference: 7B (30min), 70B (2h), 405B (3h)
- Without reference: 7B (2h), 70B (24h), 405B (48h)
- Quick diagnostic: Any size (5min)

### Q: Can I verify models larger than my RAM?

**A:** Yes! REV uses segmented execution:
- 405B model on 16GB RAM: ✅ Works (slower)
- Memory cap: 2-4GB per segment
- Automatic spilling to disk

### Q: What's the difference between reference and active library?

**A:** 
- **Reference**: Deep baseline (built once, 6-24h)
- **Active**: Continuous updates (every run)
- Reference enables 15-20x speedup for family

### Q: Can I use REV without GPU?

**A:** Yes, CPU mode fully supported:
```bash
python run_rev.py model/ --device cpu
```
Slower but works for all features.

### Q: How accurate is the verification?

**A:** Statistical guarantees:
- False positive rate: <5%
- False negative rate: <5%
- Confidence improves with more challenges

### Q: Can I verify fine-tuned models?

**A:** Yes, REV detects:
- Base model family
- Fine-tuning signatures
- Architectural modifications

### Q: What's the minimum number of challenges?

**A:** 
- Quick test: 5 (low confidence)
- Standard: 30-50 (good confidence)
- High confidence: 100+
- Reference building: 100-500

### Q: Can I resume interrupted runs?

**A:** Yes, with checkpointing:
```bash
python run_rev.py model/ --resume-from-checkpoint
```

### Q: How do I update REV?

**A:** 
```bash
git pull
pip install -r requirements.txt --upgrade
```

### Q: Where are results stored?

**A:** 
- JSON reports: `outputs/`
- Checkpoints: `checkpoints/`
- Fingerprints: `fingerprint_library/`
- Logs: `logs/`

---

For more help: [GitHub Issues](https://github.com/rohanvinaik/REV/issues) | [Documentation](https://rev-system.readthedocs.io)