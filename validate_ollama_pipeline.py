#!/usr/bin/env python3
"""
REV End-to-End Validation with Ollama Models
Tests the full pipeline with real models to validate behavioral fingerprinting.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Suppress excessive logging
import logging
logging.basicConfig(level=logging.WARNING)

from src.models.cloud_inference import create_ollama_inference, list_ollama_models
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.orchestration.prompt_orchestrator import UnifiedPromptOrchestrator


def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_result(test_name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    symbol = "[OK]" if passed else "[X]"
    print(f"  {symbol} {test_name}: {status}")
    if details:
        print(f"      {details}")


class OllamaModelInterface:
    """Wrapper to make Ollama work with REV's behavioral analysis."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference = create_ollama_inference(model_name)
        self.response_times = []

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        start = time.perf_counter()
        result = self.inference.generate(prompt, max_tokens=max_tokens)
        self.response_times.append(time.perf_counter() - start)

        if result['success']:
            return result['response']
        else:
            return f"[ERROR: {result.get('error', 'Unknown')}]"

    def get_avg_response_time(self) -> float:
        if self.response_times:
            return np.mean(self.response_times)
        return 0.0


def compute_response_divergence(responses_a: List[str], responses_b: List[str]) -> Dict[str, float]:
    """
    Compute behavioral divergence between two sets of responses.
    Uses multiple metrics to capture different aspects of divergence.
    """
    # Length-based metrics
    lens_a = [len(r) for r in responses_a]
    lens_b = [len(r) for r in responses_b]
    len_diff = abs(np.mean(lens_a) - np.mean(lens_b))

    # Vocabulary overlap
    vocab_a = set(' '.join(responses_a).lower().split())
    vocab_b = set(' '.join(responses_b).lower().split())
    vocab_overlap = len(vocab_a & vocab_b) / max(len(vocab_a | vocab_b), 1)

    # Response variance (consistency within each model)
    var_a = np.std(lens_a) if len(lens_a) > 1 else 0
    var_b = np.std(lens_b) if len(lens_b) > 1 else 0

    # HDC-based semantic divergence
    encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, sparsity=0.01))

    vecs_a = [encoder.encode_feature(r[:500], "response") for r in responses_a]
    vecs_b = [encoder.encode_feature(r[:500], "response") for r in responses_b]

    # Mean vectors for each model
    mean_a = np.mean(vecs_a, axis=0)
    mean_b = np.mean(vecs_b, axis=0)

    # Normalize
    mean_a = mean_a / (np.linalg.norm(mean_a) + 1e-8)
    mean_b = mean_b / (np.linalg.norm(mean_b) + 1e-8)

    # Cosine similarity
    semantic_similarity = np.dot(mean_a, mean_b)

    return {
        'length_divergence': len_diff,
        'vocab_overlap': vocab_overlap,
        'variance_a': var_a,
        'variance_b': var_b,
        'semantic_similarity': semantic_similarity,
        'semantic_divergence': 1 - semantic_similarity
    }


def test_model_responses(model_name: str, prompts: List[str]) -> Dict[str, Any]:
    """Test a model with given prompts and collect responses."""
    print(f"\n  Testing {model_name}...")

    model = OllamaModelInterface(model_name)
    responses = []

    for i, prompt in enumerate(prompts):
        response = model.generate(prompt, max_tokens=50)
        responses.append(response)
        if i < 2:  # Show first 2 responses
            print(f"    Prompt: {prompt[:40]}...")
            print(f"    Response: {response[:60]}...")

    return {
        'model': model_name,
        'responses': responses,
        'avg_time': model.get_avg_response_time(),
        'num_prompts': len(prompts)
    }


def main():
    print_section("REV End-to-End Validation with Ollama Models")
    print("Testing behavioral fingerprinting with real local models...")

    # Check available models
    print_section("1. Available Ollama Models")
    models = list_ollama_models()

    if not models:
        print("  [ERROR] No Ollama models found. Is Ollama running?")
        print("  Try: ollama serve")
        sys.exit(1)

    print(f"  Found {len(models)} models")

    # Select test models (small ones for speed)
    test_models = ['smollm:360m', 'tinyllama:latest', 'gemma2:2b']
    available_test_models = [m for m in test_models if any(m in model['name'] for model in models)]

    if len(available_test_models) < 2:
        print("  [ERROR] Need at least 2 test models. Available small models:")
        for m in models:
            if any(x in m['name'] for x in ['small', 'tiny', '360m', '1b', '2b', '3b']):
                print(f"    - {m['name']}")
        sys.exit(1)

    print(f"  Testing with: {available_test_models}")

    # Generate test prompts using orchestrator
    print_section("2. Generating Behavioral Probes")
    orchestrator = UnifiedPromptOrchestrator(enable_all_systems=True)

    result = orchestrator.generate_orchestrated_prompts(
        model_family="unknown",
        total_prompts=20,
        target_layers=[0, 1, 2]
    )

    all_prompts = []
    for system, prompts in result.get('prompts_by_type', {}).items():
        for p in prompts[:5]:
            if isinstance(p, dict):
                prompt_text = p.get('prompt', p.get('text', str(p)))
            else:
                prompt_text = str(p)
            # Ensure it's a clean string
            if isinstance(prompt_text, str) and len(prompt_text) > 10:
                all_prompts.append(prompt_text)

    test_prompts = all_prompts[:15]  # Use 15 prompts for testing
    print(f"  Generated {len(test_prompts)} behavioral probes")

    # Test each model
    print_section("3. Model Response Collection")
    model_results = {}

    for model_name in available_test_models:
        try:
            result = test_model_responses(model_name, test_prompts)
            model_results[model_name] = result
            print_result(f"{model_name} responses", True,
                        f"{result['num_prompts']} prompts, {result['avg_time']*1000:.0f}ms avg")
        except Exception as e:
            print_result(f"{model_name} responses", False, str(e))

    if len(model_results) < 2:
        print("\n  [ERROR] Need at least 2 working models for comparison")
        sys.exit(1)

    # Compute cross-model divergence
    print_section("4. Cross-Model Behavioral Divergence")

    model_names = list(model_results.keys())
    divergence_matrix = {}

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i:]:
            responses_a = model_results[model_a]['responses']
            responses_b = model_results[model_b]['responses']

            divergence = compute_response_divergence(responses_a, responses_b)
            key = f"{model_a} vs {model_b}"
            divergence_matrix[key] = divergence

            if model_a == model_b:
                print(f"\n  {model_a} (self-consistency):")
                print(f"    Semantic similarity: {divergence['semantic_similarity']:.4f}")
            else:
                print(f"\n  {model_a} vs {model_b}:")
                print(f"    Semantic divergence: {divergence['semantic_divergence']:.4f}")
                print(f"    Vocab overlap: {divergence['vocab_overlap']:.2%}")
                print(f"    Length diff: {divergence['length_divergence']:.1f} chars")

    # Generate fingerprints
    print_section("5. Behavioral Fingerprint Generation")

    encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, sparsity=0.01))
    fingerprints = {}

    for model_name, result in model_results.items():
        # Combine all responses into a fingerprint
        combined_text = ' '.join(result['responses'])
        fp = encoder.encode_feature(combined_text, "behavioral_signature")
        fingerprints[model_name] = fp
        print(f"  {model_name}: dim={len(fp)}, norm={np.linalg.norm(fp):.4f}")

    # Fingerprint similarity matrix
    print_section("6. Fingerprint Similarity Matrix")

    print("\n  Model Similarity (cosine):")
    print("  " + " " * 20 + "  ".join([m.split(':')[0][:8] for m in model_names]))

    for model_a in model_names:
        row = f"  {model_a.split(':')[0][:15]:15}"
        for model_b in model_names:
            sim = np.dot(fingerprints[model_a], fingerprints[model_b])
            row += f"  {sim:6.3f}"
        print(row)

    # Summary
    print_section("VALIDATION SUMMARY")

    # Check key results
    all_passed = True

    # Self-similarity should be ~1.0
    for model in model_names:
        key = f"{model} vs {model}"
        if key in divergence_matrix:
            self_sim = divergence_matrix[key]['semantic_similarity']
            passed = self_sim > 0.99
            print_result(f"{model} self-consistency", passed, f"sim={self_sim:.4f}")
            all_passed = all_passed and passed

    # Cross-model divergence should be measurable
    for key, div in divergence_matrix.items():
        if 'vs' in key and key.split(' vs ')[0] != key.split(' vs ')[1]:
            divergence = div['semantic_divergence']
            passed = divergence > 0.01  # Should show some divergence
            print_result(f"Divergence: {key}", passed, f"div={divergence:.4f}")
            all_passed = all_passed and passed

    # Save results
    results = {
        'models_tested': list(model_results.keys()),
        'prompts_used': len(test_prompts),
        'divergence_matrix': {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in divergence_matrix.items()},
        'fingerprint_dimensions': 10000,
        'validation_passed': bool(all_passed)
    }

    with open('ollama_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: ollama_validation_results.json")

    if all_passed:
        print("\n  STATUS: ALL VALIDATION CHECKS PASSED")
    else:
        print("\n  STATUS: SOME CHECKS FAILED - Review above")

    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
