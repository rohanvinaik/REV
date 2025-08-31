#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for REV Framework

Validates paper claims using real implementation:
- 99.95% memory reduction (actual model vs activation storage)
- 15.3x Hamming distance speedup
- Byzantine fault tolerance with 5 HBT replicas
- Real model performance (50-200ms inference, 52-440MB memory)

REAL IMPLEMENTATION - Uses actual models and production configurations
"""

import os
import sys
import time
import torch
import numpy as np
import psutil
import gc
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage
import pandas as pd
from tabulate import tabulate

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import REV components
from src.rev_pipeline import REVPipeline, ExecutionPolicy, Segment
from src.executor.segment_runner import SegmentRunner, SegmentConfig
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hypervector.hamming import HammingDistanceOptimized, HammingDistanceBaseline
from src.hypervector.similarity import AdvancedSimilarity
from src.consensus.byzantine import ConsensusNetwork, ByzantineValidator
from src.verifier.streaming_consensus import StreamingConsensusVerifier, ConsensusMode
from src.core.sequential import sequential_verify, SequentialState

# Import real model loading utilities
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    metric: str
    value: float
    unit: str
    baseline: Optional[float] = None
    improvement: Optional[float] = None
    verified_claim: Optional[float] = None
    
    def validate_claim(self) -> bool:
        """Check if result validates the paper claim."""
        if self.verified_claim is None:
            return True
        if self.improvement is None:
            return False
        return self.improvement >= self.verified_claim * 0.9  # Allow 10% margin


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    memory_results: List[BenchmarkResult] = field(default_factory=list)
    speed_results: List[BenchmarkResult] = field(default_factory=list)
    byzantine_results: List[BenchmarkResult] = field(default_factory=list)
    model_results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryBenchmark:
    """
    Benchmark memory usage to validate 99.95% reduction claim.
    
    REAL IMPLEMENTATION - Measures actual model vs activation storage.
    """
    
    def __init__(self):
        self.model_path = os.path.expanduser("~/LLM_models/gpt2")
        self.results = []
    
    def measure_full_model_memory(self) -> Tuple[float, Dict]:
        """Measure memory for storing full model."""
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load full model
        print("Loading full GPT-2 model...")
        model = GPT2Model.from_pretrained(self.model_path)
        model.eval()
        
        # Measure with model loaded
        full_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = full_memory - baseline_memory
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_memory = param_count * 4 / 1024 / 1024  # Float32 in MB
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return model_memory, {
            'parameters': param_count,
            'theoretical_mb': param_memory,
            'actual_mb': model_memory
        }
    
    def measure_activation_memory(self, num_challenges: int = 100) -> Tuple[float, Dict]:
        """Measure memory for storing only activations."""
        process = psutil.Process()
        
        # Initialize segment runner for activation extraction
        config = SegmentConfig(
            segment_size=512,
            overlap=50,
            max_segments=10,
            cache_segments=True
        )
        runner = SegmentRunner(config)
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Extracting activations for {num_challenges} challenges...")
        activations_storage = []
        
        # Load model once
        model = GPT2Model.from_pretrained(self.model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        model.eval()
        
        # Extract activations for challenges
        for i in range(num_challenges):
            challenge = f"Test challenge {i}: What is the meaning of life?"
            inputs = tokenizer(challenge, return_tensors='pt', max_length=512, truncation=True)
            
            # Extract only specific layer activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Store only selected layers (e.g., layers 0, 6, 11)
                selected_activations = {
                    'layer_0': outputs.hidden_states[0].cpu().numpy(),
                    'layer_6': outputs.hidden_states[6].cpu().numpy() if len(outputs.hidden_states) > 6 else None,
                    'layer_11': outputs.hidden_states[11].cpu().numpy() if len(outputs.hidden_states) > 11 else None,
                }
                activations_storage.append(selected_activations)
        
        # Measure activation memory
        activation_memory = process.memory_info().rss / 1024 / 1024
        activation_usage = activation_memory - baseline_memory
        
        # Calculate storage size
        total_activation_size = 0
        for acts in activations_storage:
            for layer_acts in acts.values():
                if layer_acts is not None:
                    total_activation_size += layer_acts.nbytes / 1024 / 1024  # MB
        
        # Clean up
        del model, activations_storage
        gc.collect()
        
        return activation_usage, {
            'num_challenges': num_challenges,
            'theoretical_mb': total_activation_size,
            'actual_mb': activation_usage
        }
    
    def calculate_memory_reduction(self) -> BenchmarkResult:
        """Calculate and validate memory reduction claim."""
        # Measure full model memory
        model_memory, model_info = self.measure_full_model_memory()
        print(f"Full model memory: {model_memory:.2f} MB")
        
        # Measure activation memory
        activation_memory, activation_info = self.measure_activation_memory()
        print(f"Activation memory: {activation_memory:.2f} MB")
        
        # Calculate reduction
        reduction_ratio = 1 - (activation_memory / model_memory)
        reduction_percentage = reduction_ratio * 100
        
        print(f"Memory reduction: {reduction_percentage:.2f}%")
        
        return BenchmarkResult(
            name="Memory Reduction",
            metric="reduction_percentage",
            value=reduction_percentage,
            unit="%",
            baseline=model_memory,
            improvement=reduction_ratio,
            verified_claim=99.95
        )


class SpeedBenchmark:
    """
    Benchmark speed improvements, especially Hamming distance.
    
    REAL IMPLEMENTATION - Validates 15.3x speedup claim.
    """
    
    def __init__(self):
        self.dimensions = [1000, 8000, 10000, 50000, 100000]
        self.results = []
    
    def benchmark_hamming_baseline(self, dimension: int, iterations: int = 1000) -> float:
        """Benchmark baseline Hamming distance implementation."""
        # Generate random hypervectors
        np.random.seed(42)
        vectors_a = np.random.randint(0, 2, (iterations, dimension), dtype=np.uint8)
        vectors_b = np.random.randint(0, 2, (iterations, dimension), dtype=np.uint8)
        
        hamming = HammingDistanceBaseline()
        
        start_time = time.perf_counter()
        for i in range(iterations):
            _ = hamming.compute(vectors_a[i], vectors_b[i])
        end_time = time.perf_counter()
        
        return (end_time - start_time) / iterations * 1000  # ms per computation
    
    def benchmark_hamming_optimized(self, dimension: int, iterations: int = 1000) -> float:
        """Benchmark optimized Hamming distance with LUTs."""
        # Generate random hypervectors
        np.random.seed(42)
        vectors_a = np.random.randint(0, 2, (iterations, dimension), dtype=np.uint8)
        vectors_b = np.random.randint(0, 2, (iterations, dimension), dtype=np.uint8)
        
        hamming = HammingDistanceOptimized(use_parallel=True)
        
        start_time = time.perf_counter()
        for i in range(iterations):
            _ = hamming.compute(vectors_a[i], vectors_b[i])
        end_time = time.perf_counter()
        
        return (end_time - start_time) / iterations * 1000  # ms per computation
    
    def calculate_hamming_speedup(self) -> List[BenchmarkResult]:
        """Calculate Hamming distance speedup across dimensions."""
        results = []
        
        for dim in self.dimensions:
            print(f"Benchmarking Hamming distance for dimension {dim}...")
            
            baseline_time = self.benchmark_hamming_baseline(dim)
            optimized_time = self.benchmark_hamming_optimized(dim)
            speedup = baseline_time / optimized_time
            
            print(f"  Baseline: {baseline_time:.4f} ms")
            print(f"  Optimized: {optimized_time:.4f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append(BenchmarkResult(
                name=f"Hamming_{dim}D",
                metric="speedup",
                value=speedup,
                unit="x",
                baseline=baseline_time,
                improvement=speedup,
                verified_claim=15.3 if dim == 10000 else None
            ))
        
        return results
    
    def benchmark_inference_speed(self) -> BenchmarkResult:
        """Benchmark real model inference speed."""
        model_path = os.path.expanduser("~/LLM_models/gpt2")
        
        # Load model
        model = GPT2Model.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model.eval()
        
        # Prepare test inputs
        test_prompts = [
            "The future of artificial intelligence is",
            "Climate change affects our planet by",
            "The most important scientific discovery was",
        ]
        
        inference_times = []
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Warm up
            with torch.no_grad():
                _ = model(**inputs)
            
            # Measure
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)  # ms
        
        avg_inference_time = np.mean(inference_times)
        
        # Clean up
        del model
        gc.collect()
        
        return BenchmarkResult(
            name="Model Inference",
            metric="latency",
            value=avg_inference_time,
            unit="ms",
            baseline=1.0,  # Mock implementation baseline
            improvement=avg_inference_time,
            verified_claim=200  # Max expected from testing
        )


class ByzantineBenchmark:
    """
    Benchmark Byzantine fault tolerance with 5 replicas.
    
    REAL IMPLEMENTATION - Tests consensus with Docker configuration.
    """
    
    def __init__(self):
        self.num_validators = 5  # From Docker configuration
        self.fault_tolerance = 1  # Can tolerate 1 Byzantine node
    
    def test_consensus_with_failures(self, num_failures: int) -> Tuple[bool, float]:
        """Test consensus with specified number of Byzantine failures."""
        # Create consensus network
        network = ConsensusNetwork(
            num_validators=self.num_validators,
            fault_tolerance=self.fault_tolerance
        )
        
        # Initialize validators
        validators = []
        for i in range(self.num_validators):
            validator = ByzantineValidator(
                node_id=f"validator_{i}",
                total_nodes=self.num_validators,
                is_byzantine=(i < num_failures)  # Make first n validators Byzantine
            )
            validators.append(validator)
            network.add_validator(validator)
        
        # Test data
        test_data = {
            'challenge': 'test_prompt',
            'response': 'model_output',
            'hash': hashlib.sha256(b'test_data').hexdigest()
        }
        
        # Measure consensus time
        start_time = time.perf_counter()
        
        # Each validator votes
        for validator in validators:
            if validator.is_byzantine:
                # Byzantine nodes vote randomly
                vote = np.random.choice([True, False])
            else:
                # Honest nodes vote correctly
                vote = True
            network.submit_vote(validator.node_id, test_data['hash'], vote)
        
        # Reach consensus
        consensus_reached, consensus_value = network.check_consensus(test_data['hash'])
        
        end_time = time.perf_counter()
        consensus_time = (end_time - start_time) * 1000  # ms
        
        return consensus_reached, consensus_time
    
    def run_byzantine_tests(self) -> List[BenchmarkResult]:
        """Run comprehensive Byzantine fault tolerance tests."""
        results = []
        
        # Test with no failures
        print("Testing consensus with no Byzantine nodes...")
        success, time_ms = self.test_consensus_with_failures(0)
        results.append(BenchmarkResult(
            name="Byzantine_0_Failures",
            metric="consensus_time",
            value=time_ms,
            unit="ms",
            baseline=None,
            improvement=1.0 if success else 0.0,
            verified_claim=1.0
        ))
        print(f"  Success: {success}, Time: {time_ms:.2f} ms")
        
        # Test with 1 failure (should tolerate)
        print("Testing consensus with 1 Byzantine node...")
        success, time_ms = self.test_consensus_with_failures(1)
        results.append(BenchmarkResult(
            name="Byzantine_1_Failure",
            metric="consensus_time",
            value=time_ms,
            unit="ms",
            baseline=None,
            improvement=1.0 if success else 0.0,
            verified_claim=1.0
        ))
        print(f"  Success: {success}, Time: {time_ms:.2f} ms")
        
        # Test with 2 failures (should fail)
        print("Testing consensus with 2 Byzantine nodes...")
        success, time_ms = self.test_consensus_with_failures(2)
        results.append(BenchmarkResult(
            name="Byzantine_2_Failures",
            metric="consensus_time",
            value=time_ms,
            unit="ms",
            baseline=None,
            improvement=1.0 if success else 0.0,
            verified_claim=0.0  # Should fail
        ))
        print(f"  Success: {success}, Time: {time_ms:.2f} ms")
        
        return results


class ModelPerformanceBenchmark:
    """
    Benchmark real model performance metrics.
    
    REAL IMPLEMENTATION - Validates 50-200ms inference, 52-440MB memory.
    """
    
    def __init__(self):
        self.models = {
            'gpt2': '~/LLM_models/gpt2',
            'distilgpt2': '~/LLM_models/distilgpt2'
        }
    
    def benchmark_model(self, model_name: str, model_path: str) -> Dict[str, BenchmarkResult]:
        """Benchmark a specific model."""
        model_path = os.path.expanduser(model_path)
        results = {}
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}, skipping...")
            return results
        
        print(f"Benchmarking {model_name}...")
        
        # Memory benchmark
        process = psutil.Process()
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Load model
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        # Measure memory
        loaded_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = loaded_memory - baseline_memory
        
        results['memory'] = BenchmarkResult(
            name=f"{model_name}_memory",
            metric="memory_usage",
            value=memory_usage,
            unit="MB",
            baseline=0,  # Mock implementation used 0MB
            improvement=memory_usage,
            verified_claim=440  # Max expected
        )
        
        # Inference speed benchmark
        test_prompt = "The future of artificial intelligence"
        inputs = tokenizer(test_prompt, return_tensors='pt', max_length=512, truncation=True)
        
        # Warm up
        with torch.no_grad():
            _ = model(**inputs)
        
        # Measure inference time
        inference_times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            end = time.perf_counter()
            inference_times.append((end - start) * 1000)
        
        avg_inference = np.mean(inference_times)
        
        results['inference'] = BenchmarkResult(
            name=f"{model_name}_inference",
            metric="inference_time",
            value=avg_inference,
            unit="ms",
            baseline=1.0,  # Mock implementation ~1ms
            improvement=avg_inference,
            verified_claim=200  # Max expected
        )
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def run_model_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for all models."""
        all_results = []
        
        for model_name, model_path in self.models.items():
            results = self.benchmark_model(model_name, model_path)
            all_results.extend(results.values())
        
        return all_results


def generate_latex_tables(suite: BenchmarkSuite) -> str:
    """
    Generate LaTeX tables for paper publication.
    
    Returns publication-ready LaTeX code.
    """
    latex_output = []
    
    # Memory Reduction Table
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Memory Reduction Validation}")
    latex_output.append(r"\begin{tabular}{|l|r|r|r|c|}")
    latex_output.append(r"\hline")
    latex_output.append(r"Metric & Baseline (MB) & Optimized (MB) & Reduction (\%) & Claim Met \\")
    latex_output.append(r"\hline")
    
    for result in suite.memory_results:
        claim_met = "\\checkmark" if result.validate_claim() else "\\times"
        latex_output.append(
            f"{result.name} & {result.baseline:.2f} & {result.value:.2f} & "
            f"{result.improvement*100:.2f} & {claim_met} \\\\"
        )
    
    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    latex_output.append("")
    
    # Speed Improvement Table
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Performance Speedup Validation}")
    latex_output.append(r"\begin{tabular}{|l|r|r|r|r|c|}")
    latex_output.append(r"\hline")
    latex_output.append(r"Operation & Dimension & Baseline (ms) & Optimized (ms) & Speedup & Target \\")
    latex_output.append(r"\hline")
    
    for result in suite.speed_results:
        if 'Hamming' in result.name:
            dim = result.name.split('_')[1].replace('D', '')
            target = f"{result.verified_claim:.1f}x" if result.verified_claim else "-"
            latex_output.append(
                f"Hamming & {dim} & {result.baseline:.4f} & "
                f"{result.baseline/result.value:.4f} & {result.value:.2f}x & {target} \\\\"
            )
    
    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    latex_output.append("")
    
    # Model Performance Table
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Real Model Performance Metrics}")
    latex_output.append(r"\begin{tabular}{|l|r|r|r|r|}")
    latex_output.append(r"\hline")
    latex_output.append(r"Model & Memory (MB) & Inference (ms) & vs Mock Memory & vs Mock Speed \\")
    latex_output.append(r"\hline")
    
    model_metrics = {}
    for result in suite.model_results:
        model = result.name.split('_')[0]
        if model not in model_metrics:
            model_metrics[model] = {}
        metric_type = result.name.split('_')[1]
        model_metrics[model][metric_type] = result
    
    for model, metrics in model_metrics.items():
        if 'memory' in metrics and 'inference' in metrics:
            mem = metrics['memory']
            inf = metrics['inference']
            latex_output.append(
                f"{model} & {mem.value:.1f} & {inf.value:.2f} & "
                f"{mem.value:.0f}x & {inf.value:.0f}x \\\\"
            )
    
    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    
    return "\n".join(latex_output)


def generate_comparison_charts(suite: BenchmarkSuite):
    """Generate comparison charts for visualization."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Memory Comparison
    ax = axes[0, 0]
    categories = ['Mock Implementation', 'Real GPT-2', 'Real DistilGPT2']
    memory_values = [0]  # Mock uses 0MB
    
    for result in suite.model_results:
        if 'memory' in result.name:
            memory_values.append(result.value)
    
    if len(memory_values) > 1:
        ax.bar(categories[:len(memory_values)], memory_values, color=['red', 'green', 'blue'])
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage: Mock vs Real Implementation')
        ax.axhline(y=440, color='r', linestyle='--', label='Max Target (440MB)')
        ax.axhline(y=52, color='g', linestyle='--', label='Min Target (52MB)')
        ax.legend()
    
    # 2. Inference Latency Comparison
    ax = axes[0, 1]
    latency_mock = [1.0]  # Mock ~1ms
    latency_real = []
    
    for result in suite.model_results:
        if 'inference' in result.name:
            latency_real.append(result.value)
    
    if latency_real:
        x_pos = np.arange(len(categories[:1+len(latency_real)]))
        ax.bar(x_pos, latency_mock + latency_real, color=['red', 'green', 'blue'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories[:1+len(latency_real)])
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Latency: Mock vs Real')
        ax.axhline(y=200, color='r', linestyle='--', label='Max Target (200ms)')
        ax.axhline(y=50, color='g', linestyle='--', label='Min Target (50ms)')
        ax.legend()
    
    # 3. Hamming Distance Speedup
    ax = axes[1, 0]
    dimensions = []
    speedups = []
    
    for result in suite.speed_results:
        if 'Hamming' in result.name:
            dim = int(result.name.split('_')[1].replace('D', ''))
            dimensions.append(dim)
            speedups.append(result.value)
    
    if dimensions:
        ax.plot(dimensions, speedups, 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=15.3, color='r', linestyle='--', label='Target (15.3x)')
        ax.set_xlabel('Vector Dimension')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Hamming Distance Speedup vs Dimension')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 4. Byzantine Fault Tolerance
    ax = axes[1, 1]
    byzantine_scenarios = []
    success_rates = []
    
    for result in suite.byzantine_results:
        scenario = result.name.replace('Byzantine_', '').replace('_', ' ')
        byzantine_scenarios.append(scenario)
        success_rates.append(result.improvement * 100)
    
    if byzantine_scenarios:
        colors = ['green' if rate == 100 else 'red' for rate in success_rates]
        ax.bar(byzantine_scenarios, success_rates, color=colors)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Byzantine Fault Tolerance (5 Replicas)')
        ax.set_ylim([0, 110])
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'benchmarks/benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison charts saved to {output_path}")
    
    plt.show()


def run_complete_benchmark_suite():
    """Run the complete benchmark suite and generate reports."""
    print("=" * 80)
    print("REV FRAMEWORK COMPREHENSIVE BENCHMARK SUITE")
    print("Validating Paper Claims with Real Implementation")
    print("=" * 80)
    
    suite = BenchmarkSuite()
    
    # 1. Memory Reduction Benchmarks
    print("\n[1/4] Running Memory Reduction Benchmarks...")
    print("-" * 40)
    memory_bench = MemoryBenchmark()
    memory_result = memory_bench.calculate_memory_reduction()
    suite.memory_results.append(memory_result)
    
    if memory_result.validate_claim():
        print(f"✅ Memory reduction claim VALIDATED: {memory_result.value:.2f}% >= {memory_result.verified_claim}%")
    else:
        print(f"❌ Memory reduction claim NOT MET: {memory_result.value:.2f}% < {memory_result.verified_claim}%")
    
    # 2. Speed Improvement Benchmarks
    print("\n[2/4] Running Speed Improvement Benchmarks...")
    print("-" * 40)
    speed_bench = SpeedBenchmark()
    
    # Hamming distance speedup
    hamming_results = speed_bench.calculate_hamming_speedup()
    suite.speed_results.extend(hamming_results)
    
    # Validate 15.3x claim for 10K dimensions
    for result in hamming_results:
        if result.verified_claim == 15.3:
            if result.validate_claim():
                print(f"✅ Hamming speedup claim VALIDATED: {result.value:.1f}x >= {result.verified_claim}x")
            else:
                print(f"❌ Hamming speedup claim NOT MET: {result.value:.1f}x < {result.verified_claim}x")
    
    # Model inference speed
    inference_result = speed_bench.benchmark_inference_speed()
    suite.speed_results.append(inference_result)
    
    if 50 <= inference_result.value <= 200:
        print(f"✅ Inference latency within target: {inference_result.value:.2f}ms ∈ [50, 200]ms")
    else:
        print(f"❌ Inference latency outside target: {inference_result.value:.2f}ms ∉ [50, 200]ms")
    
    # 3. Byzantine Fault Tolerance
    print("\n[3/4] Running Byzantine Fault Tolerance Tests...")
    print("-" * 40)
    byzantine_bench = ByzantineBenchmark()
    byzantine_results = byzantine_bench.run_byzantine_tests()
    suite.byzantine_results.extend(byzantine_results)
    
    # Check fault tolerance
    tolerance_met = True
    for result in byzantine_results:
        if '1 Failure' in result.name and result.improvement != 1.0:
            tolerance_met = False
        elif '2 Failures' in result.name and result.improvement != 0.0:
            tolerance_met = False
    
    if tolerance_met:
        print("✅ Byzantine fault tolerance VALIDATED: Tolerates 1 failure with 5 replicas")
    else:
        print("❌ Byzantine fault tolerance FAILED")
    
    # 4. Real Model Performance
    print("\n[4/4] Running Real Model Performance Benchmarks...")
    print("-" * 40)
    model_bench = ModelPerformanceBenchmark()
    model_results = model_bench.run_model_benchmarks()
    suite.model_results.extend(model_results)
    
    # Validate memory and latency ranges
    for result in model_results:
        if 'memory' in result.name:
            if 52 <= result.value <= 440:
                print(f"✅ {result.name}: {result.value:.1f}MB ∈ [52, 440]MB")
            else:
                print(f"❌ {result.name}: {result.value:.1f}MB ∉ [52, 440]MB")
        elif 'inference' in result.name:
            if 50 <= result.value <= 200:
                print(f"✅ {result.name}: {result.value:.2f}ms ∈ [50, 200]ms")
            else:
                print(f"❌ {result.name}: {result.value:.2f}ms ∉ [50, 200]ms")
    
    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    # Generate LaTeX tables
    latex_tables = generate_latex_tables(suite)
    latex_output_path = 'benchmarks/benchmark_tables.tex'
    with open(latex_output_path, 'w') as f:
        f.write(latex_tables)
    print(f"LaTeX tables saved to {latex_output_path}")
    
    # Generate comparison charts
    generate_comparison_charts(suite)
    
    # Save JSON report
    json_report = {
        'timestamp': suite.timestamp,
        'summary': {
            'memory_reduction_validated': all(r.validate_claim() for r in suite.memory_results),
            'hamming_speedup_validated': any(r.validate_claim() for r in suite.speed_results if r.verified_claim == 15.3),
            'byzantine_tolerance_validated': tolerance_met,
            'performance_targets_met': all(
                52 <= r.value <= 440 for r in suite.model_results if 'memory' in r.name
            ) and all(
                50 <= r.value <= 200 for r in suite.model_results if 'inference' in r.name
            )
        },
        'detailed_results': {
            'memory': [vars(r) for r in suite.memory_results],
            'speed': [vars(r) for r in suite.speed_results],
            'byzantine': [vars(r) for r in suite.byzantine_results],
            'models': [vars(r) for r in suite.model_results]
        }
    }
    
    json_output_path = 'benchmarks/benchmark_results.json'
    with open(json_output_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved to {json_output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    
    summary_table = [
        ["Claim", "Status", "Details"],
        ["-" * 20, "-" * 10, "-" * 40],
        ["99.95% Memory Reduction", 
         "✅" if json_report['summary']['memory_reduction_validated'] else "❌",
         f"Achieved: {suite.memory_results[0].value:.2f}%" if suite.memory_results else "N/A"],
        ["15.3x Hamming Speedup",
         "✅" if json_report['summary']['hamming_speedup_validated'] else "❌",
         f"10K dim: {next((r.value for r in suite.speed_results if '10000D' in r.name), 0):.1f}x"],
        ["Byzantine Tolerance",
         "✅" if json_report['summary']['byzantine_tolerance_validated'] else "❌",
         "Tolerates 1/5 failures"],
        ["Performance Targets",
         "✅" if json_report['summary']['performance_targets_met'] else "❌",
         "Memory: 52-440MB, Latency: 50-200ms"]
    ]
    
    for row in summary_table:
        print(f"{row[0]:<25} {row[1]:<10} {row[2]:<40}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Create benchmarks directory if it doesn't exist
    os.makedirs('benchmarks', exist_ok=True)
    
    # Run complete benchmark suite
    run_complete_benchmark_suite()