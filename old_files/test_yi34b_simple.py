#!/usr/bin/env python3
"""Simple test to verify Yi-34B loads and runs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("Testing Yi-34B model loading and inference...")
print("="*60)

# Load model
print("Loading model...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    "/Users/rohanvinaik/LLM_models/yi-34b",
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)
print(f"Model loaded in {time.time()-start:.1f}s")

# Load tokenizer  
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/Users/rohanvinaik/LLM_models/yi-34b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

# Model info
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params/1e9:.1f}B")

# Simple test
print("\nRunning inference test...")
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"Input tokens: {inputs['input_ids'].shape}")

# Generate with minimal tokens
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=10,
        do_sample=False,
        temperature=1.0
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
print("\n✅ Yi-34B model test successful!")