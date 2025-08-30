#!/usr/bin/env python3
"""
Comprehensive performance optimization script for the REV system.

This script profiles and tunes the REV system for optimal performance,
including memory analysis, CPU profiling, I/O optimization, and auto-tuning.
"""

import os
import sys
import time
import json
import psutil
import tracemalloc
import cProfile
import pstats
import io
import threading
import multiprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile as memory_profile
import torch
import hashlib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rev_pipeline import REVPipeline, REVConfig
from src.hdc.encoder import UnifiedHDCEncoder, HDCConfig
from src.executor.segment_runner import SegmentRunner, SegmentConfig
from src.core.sequential import EnhancedSequentialTester, SequentialConfig
from src.hypervector.hamming import OptimizedHammingCalculator


@dataclass
class ProfilingResult:
    """Results from performance profiling."""
    memory_usage: Dict[str, float] = field(default_factory=dict)
    cpu_stats: Dict[str, float] = field(default_factory=dict)
    io_stats: Dict[str, float] = field(default_factory=dict)
    cache_stats: Dict[str, float] = field(default_factory=dict)
    timing_stats: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Results from auto-tuning optimization."""
    optimal_batch_size: int = 32
    optimal_thread_count: int = 4
    optimal_buffer_size: int = 1024
    optimal_lut_size: int = 65536
    optimal_cache_size: int = 1000
    optimal_segment_size: int = 512
    performance_gain: float = 0.0
    config_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_metrics: Dict[str, float] = field(default_factory=dict)
    regression_detected: bool = False
    regression_details: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    bottleneck_analysis: Dict[str, str] = field(default_factory=dict)
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiler for REV system."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cpu_profiler = cProfile.Profile()
        self.memory_snapshots = []
        self.io_counters = {}
        self.cache_metrics = defaultdict(lambda: {"hits": 0, "misses": 0})
        
    @contextmanager
    def profile_section(self, name: str):
        """Context manager for profiling a code section."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_io = psutil.Process().io_counters() if hasattr(psutil.Process(), 'io_counters') else None
        
        yield
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_io = psutil.Process().io_counters() if hasattr(psutil.Process(), 'io_counters') else None
        
        self.io_counters[name] = {
            "duration": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "read_bytes": (end_io.read_bytes - start_io.read_bytes) if start_io else 0,
            "write_bytes": (end_io.write_bytes - start_io.write_bytes) if start_io else 0,
        }
    
    def profile_memory(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict]:
        """Profile memory usage of a function."""
        tracemalloc.start()
        
        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate statistics
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        memory_stats = {
            "peak_memory_mb": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
            "current_memory_mb": tracemalloc.get_traced_memory()[0] / 1024 / 1024,
            "top_allocations": []
        }
        
        for stat in top_stats[:10]:
            memory_stats["top_allocations"].append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_mb": stat.size_diff / 1024 / 1024,
                "count": stat.count_diff
            })
        
        tracemalloc.stop()
        return result, memory_stats
    
    def profile_cpu(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict]:
        """Profile CPU usage of a function."""
        self.cpu_profiler.enable()
        result = func(*args, **kwargs)
        self.cpu_profiler.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(self.cpu_profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        # Parse top functions
        cpu_stats = {
            "total_calls": ps.total_calls,
            "total_time": ps.total_tt,
            "top_functions": self._parse_cpu_stats(s.getvalue())
        }
        
        return result, cpu_stats
    
    def _parse_cpu_stats(self, stats_str: str) -> List[Dict]:
        """Parse CPU profiling statistics."""
        functions = []
        lines = stats_str.split('\n')
        
        for line in lines:
            if 'function calls' in line or 'ncalls' in line or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    functions.append({
                        "ncalls": parts[0],
                        "tottime": float(parts[1]),
                        "percall": float(parts[2]),
                        "cumtime": float(parts[3]),
                        "function": ' '.join(parts[5:])
                    })
                except (ValueError, IndexError):
                    continue
        
        return functions[:10]  # Top 10 functions
    
    def detect_bottlenecks(self, profiling_result: ProfilingResult) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottlenecks
        if profiling_result.memory_usage.get("peak_memory_mb", 0) > 1000:
            bottlenecks.append("High memory usage detected (>1GB)")
        
        # CPU bottlenecks
        if profiling_result.cpu_stats.get("total_time", 0) > 10:
            bottlenecks.append("High CPU time detected (>10s)")
        
        # I/O bottlenecks
        for name, stats in profiling_result.io_stats.items():
            if stats.get("read_bytes", 0) > 100 * 1024 * 1024:  # 100MB
                bottlenecks.append(f"High I/O read in {name} (>100MB)")
            if stats.get("write_bytes", 0) > 100 * 1024 * 1024:
                bottlenecks.append(f"High I/O write in {name} (>100MB)")
        
        # Cache performance
        for cache_name, stats in profiling_result.cache_stats.items():
            if stats["hits"] + stats["misses"] > 0:
                hit_rate = stats["hits"] / (stats["hits"] + stats["misses"])
                if hit_rate < 0.8:
                    bottlenecks.append(f"Low cache hit rate in {cache_name} ({hit_rate:.1%})")
        
        return bottlenecks


class AutoTuner:
    """Auto-tuning system for REV performance optimization."""
    
    def __init__(self, config: REVConfig):
        self.base_config = config
        self.profiler = PerformanceProfiler()
        self.test_samples = self._generate_test_samples()
        
    def _generate_test_samples(self, n_samples: int = 100) -> List[np.ndarray]:
        """Generate test samples for benchmarking."""
        samples = []
        for _ in range(n_samples):
            # Generate random hypervector
            dim = self.base_config.hdc_config.dimension
            if self.base_config.hdc_config.use_sparse:
                hv = np.zeros(dim)
                active_indices = np.random.choice(dim, int(dim * 0.01), replace=False)
                hv[active_indices] = np.random.choice([-1, 1], len(active_indices))
            else:
                hv = np.random.randn(dim)
            samples.append(hv)
        return samples
    
    def benchmark_configuration(self, config: REVConfig, samples: List[np.ndarray]) -> Dict[str, float]:
        """Benchmark a specific configuration."""
        metrics = {}
        
        # Initialize components
        pipeline = REVPipeline(config)
        
        # Measure throughput
        start_time = time.perf_counter()
        for sample in samples:
            _ = pipeline.hdc_encoder.encode(sample)
        encoding_time = time.perf_counter() - start_time
        metrics["encoding_throughput"] = len(samples) / encoding_time
        
        # Measure latency
        latencies = []
        for sample in samples[:10]:  # Sample subset for latency
            start = time.perf_counter()
            _ = pipeline.hdc_encoder.encode(sample)
            latencies.append(time.perf_counter() - start)
        metrics["p50_latency"] = np.percentile(latencies, 50)
        metrics["p99_latency"] = np.percentile(latencies, 99)
        
        # Measure memory
        process = psutil.Process()
        metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        return metrics
    
    def optimize_batch_size(self) -> int:
        """Find optimal batch size for processing."""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        best_throughput = 0
        best_batch_size = 32
        
        for batch_size in batch_sizes:
            config = self.base_config
            config.batch_size = batch_size
            
            metrics = self.benchmark_configuration(config, self.test_samples)
            throughput = metrics["encoding_throughput"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
        
        return best_batch_size
    
    def optimize_thread_count(self) -> int:
        """Find optimal thread count for parallel processing."""
        cpu_count = multiprocessing.cpu_count()
        thread_counts = [1, 2, 4, 8, min(16, cpu_count), cpu_count]
        best_throughput = 0
        best_threads = 4
        
        for threads in thread_counts:
            # Test with thread pool
            with ThreadPoolExecutor(max_workers=threads) as executor:
                start_time = time.perf_counter()
                futures = [executor.submit(self._process_sample, s) for s in self.test_samples]
                [f.result() for f in futures]
                duration = time.perf_counter() - start_time
            
            throughput = len(self.test_samples) / duration
            if throughput > best_throughput:
                best_throughput = throughput
                best_threads = threads
        
        return best_threads
    
    def _process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single sample (helper for thread testing)."""
        encoder = UnifiedHDCEncoder(self.base_config.hdc_config)
        return encoder.encode(sample)
    
    def optimize_buffer_size(self) -> int:
        """Find optimal buffer size for streaming operations."""
        buffer_sizes = [256, 512, 1024, 2048, 4096, 8192]
        best_latency = float('inf')
        best_buffer_size = 1024
        
        for buffer_size in buffer_sizes:
            # Simulate streaming with buffer
            buffer = deque(maxlen=buffer_size)
            
            start_time = time.perf_counter()
            for sample in self.test_samples:
                buffer.append(sample)
                if len(buffer) == buffer_size:
                    # Process buffer
                    _ = np.mean(list(buffer), axis=0)
                    buffer.clear()
            latency = time.perf_counter() - start_time
            
            if latency < best_latency:
                best_latency = latency
                best_buffer_size = buffer_size
        
        return best_buffer_size
    
    def optimize_lut_size(self) -> int:
        """Find optimal lookup table size for Hamming distance."""
        lut_sizes = [256, 1024, 4096, 16384, 65536, 262144]
        best_performance = 0
        best_lut_size = 65536
        
        for lut_size in lut_sizes:
            calculator = OptimizedHammingCalculator(
                dimension=self.base_config.hdc_config.dimension,
                use_lut=True,
                lut_size=lut_size
            )
            
            # Benchmark Hamming distance calculations
            start_time = time.perf_counter()
            for i in range(len(self.test_samples) - 1):
                _ = calculator.hamming_distance(
                    self.test_samples[i],
                    self.test_samples[i + 1]
                )
            duration = time.perf_counter() - start_time
            
            performance = (len(self.test_samples) - 1) / duration
            if performance > best_performance:
                best_performance = performance
                best_lut_size = lut_size
        
        return best_lut_size
    
    def optimize_cache_size(self) -> int:
        """Find optimal cache size for verification results."""
        cache_sizes = [100, 500, 1000, 2000, 5000, 10000]
        best_hit_rate = 0
        best_cache_size = 1000
        
        # Generate access pattern (with some repetition)
        access_pattern = np.random.choice(len(self.test_samples), 
                                        size=len(self.test_samples) * 5,
                                        p=self._zipf_distribution(len(self.test_samples)))
        
        for cache_size in cache_sizes:
            cache = {}
            hits = 0
            misses = 0
            
            for idx in access_pattern:
                if idx in cache:
                    hits += 1
                else:
                    misses += 1
                    cache[idx] = self.test_samples[idx]
                    # Evict if cache is full (LRU)
                    if len(cache) > cache_size:
                        # Simple eviction (remove first item)
                        cache.pop(next(iter(cache)))
            
            hit_rate = hits / (hits + misses)
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_cache_size = cache_size
        
        return best_cache_size
    
    def _zipf_distribution(self, n: int, alpha: float = 1.5) -> np.ndarray:
        """Generate Zipf distribution for cache access pattern."""
        weights = np.arange(1, n + 1) ** (-alpha)
        return weights / weights.sum()
    
    def run_full_optimization(self) -> OptimizationResult:
        """Run complete auto-tuning optimization."""
        print("Starting auto-tuning optimization...")
        
        result = OptimizationResult()
        
        # Optimize each parameter
        print("  Optimizing batch size...")
        result.optimal_batch_size = self.optimize_batch_size()
        
        print("  Optimizing thread count...")
        result.optimal_thread_count = self.optimize_thread_count()
        
        print("  Optimizing buffer size...")
        result.optimal_buffer_size = self.optimize_buffer_size()
        
        print("  Optimizing LUT size...")
        result.optimal_lut_size = self.optimize_lut_size()
        
        print("  Optimizing cache size...")
        result.optimal_cache_size = self.optimize_cache_size()
        
        # Calculate performance gain
        baseline_metrics = self.benchmark_configuration(self.base_config, self.test_samples)
        
        # Apply optimizations
        optimized_config = self.base_config
        optimized_config.batch_size = result.optimal_batch_size
        optimized_config.segment_config.buffer_size = result.optimal_buffer_size
        
        optimized_metrics = self.benchmark_configuration(optimized_config, self.test_samples)
        
        result.performance_gain = (
            (optimized_metrics["encoding_throughput"] - baseline_metrics["encoding_throughput"]) 
            / baseline_metrics["encoding_throughput"] * 100
        )
        
        result.config_changes = {
            "batch_size": result.optimal_batch_size,
            "thread_count": result.optimal_thread_count,
            "buffer_size": result.optimal_buffer_size,
            "lut_size": result.optimal_lut_size,
            "cache_size": result.optimal_cache_size
        }
        
        return result


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_metrics = self._load_baseline() if baseline_file else {}
        
    def _load_baseline(self) -> Dict[str, float]:
        """Load baseline metrics from file."""
        if Path(self.baseline_file).exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def detect_regression(self, current_metrics: Dict[str, float], 
                         threshold: float = 0.1) -> Tuple[bool, List[str]]:
        """Detect performance regression compared to baseline."""
        if not self.baseline_metrics:
            return False, []
        
        regressions = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                # Check for regression based on metric type
                if "throughput" in metric.lower():
                    # Higher is better
                    if current_value < baseline_value * (1 - threshold):
                        regressions.append(
                            f"{metric}: {current_value:.2f} vs baseline {baseline_value:.2f} "
                            f"({(baseline_value - current_value) / baseline_value * 100:.1f}% decrease)"
                        )
                elif "latency" in metric.lower() or "time" in metric.lower():
                    # Lower is better
                    if current_value > baseline_value * (1 + threshold):
                        regressions.append(
                            f"{metric}: {current_value:.2f} vs baseline {baseline_value:.2f} "
                            f"({(current_value - baseline_value) / baseline_value * 100:.1f}% increase)"
                        )
                elif "memory" in metric.lower():
                    # Lower is better
                    if current_value > baseline_value * (1 + threshold):
                        regressions.append(
                            f"{metric}: {current_value:.2f} MB vs baseline {baseline_value:.2f} MB "
                            f"({(current_value - baseline_value) / baseline_value * 100:.1f}% increase)"
                        )
        
        return len(regressions) > 0, regressions
    
    def generate_recommendations(self, profiling_result: ProfilingResult,
                                optimization_result: OptimizationResult) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        if profiling_result.memory_usage.get("peak_memory_mb", 0) > 1000:
            recommendations.append(
                "Consider reducing hypervector dimension or using sparse encoding to reduce memory usage"
            )
        
        # CPU recommendations
        cpu_time = profiling_result.cpu_stats.get("total_time", 0)
        if cpu_time > 10:
            recommendations.append(
                f"Enable parallel processing with {optimization_result.optimal_thread_count} threads"
            )
        
        # Batch size recommendations
        if optimization_result.optimal_batch_size > 32:
            recommendations.append(
                f"Increase batch size to {optimization_result.optimal_batch_size} for better throughput"
            )
        
        # Cache recommendations
        for cache_name, stats in profiling_result.cache_stats.items():
            if stats["hits"] + stats["misses"] > 0:
                hit_rate = stats["hits"] / (stats["hits"] + stats["misses"])
                if hit_rate < 0.8:
                    recommendations.append(
                        f"Increase {cache_name} cache size to {optimization_result.optimal_cache_size}"
                    )
        
        # LUT recommendations
        if optimization_result.optimal_lut_size != 65536:
            recommendations.append(
                f"Adjust Hamming LUT size to {optimization_result.optimal_lut_size} for optimal performance"
            )
        
        # I/O recommendations
        high_io = any(
            stats.get("read_bytes", 0) > 100 * 1024 * 1024 or 
            stats.get("write_bytes", 0) > 100 * 1024 * 1024
            for stats in profiling_result.io_stats.values()
        )
        if high_io:
            recommendations.append(
                "Consider using memory-mapped files or async I/O for large data operations"
            )
        
        return recommendations
    
    def generate_report(self, profiling_result: ProfilingResult,
                       optimization_result: OptimizationResult,
                       current_metrics: Dict[str, float]) -> PerformanceReport:
        """Generate comprehensive performance report."""
        report = PerformanceReport()
        
        # Set metrics
        report.baseline_metrics = self.baseline_metrics
        report.optimized_metrics = current_metrics
        
        # Detect regression
        report.regression_detected, report.regression_details = self.detect_regression(current_metrics)
        
        # Generate recommendations
        report.recommendations = self.generate_recommendations(profiling_result, optimization_result)
        
        # Bottleneck analysis
        for bottleneck in profiling_result.bottlenecks:
            if "memory" in bottleneck.lower():
                report.bottleneck_analysis["memory"] = bottleneck
            elif "cpu" in bottleneck.lower():
                report.bottleneck_analysis["cpu"] = bottleneck
            elif "i/o" in bottleneck.lower():
                report.bottleneck_analysis["io"] = bottleneck
            elif "cache" in bottleneck.lower():
                report.bottleneck_analysis["cache"] = bottleneck
        
        # Benchmark comparison
        if self.baseline_metrics:
            for metric in current_metrics:
                if metric in self.baseline_metrics:
                    baseline = self.baseline_metrics[metric]
                    current = current_metrics[metric]
                    if "throughput" in metric.lower():
                        change = (current - baseline) / baseline * 100
                    else:  # latency, memory, etc.
                        change = -(current - baseline) / baseline * 100
                    report.benchmark_comparison[metric] = change
        
        return report
    
    def save_report(self, report: PerformanceReport, output_file: str):
        """Save performance report to file."""
        report_dict = asdict(report)
        report_dict["timestamp"] = time.time()
        report_dict["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    def generate_visualization(self, report: PerformanceReport, output_dir: str):
        """Generate performance visualization charts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Performance comparison chart
        if report.benchmark_comparison:
            plt.figure(figsize=(12, 6))
            metrics = list(report.benchmark_comparison.keys())
            changes = list(report.benchmark_comparison.values())
            
            colors = ['green' if c >= 0 else 'red' for c in changes]
            plt.bar(metrics, changes, color=colors)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xlabel('Metrics')
            plt.ylabel('Performance Change (%)')
            plt.title('Performance Comparison vs Baseline')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'performance_comparison.png')
            plt.close()
        
        # Optimization impact chart
        if report.optimized_metrics and report.baseline_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Throughput comparison
            if 'encoding_throughput' in report.optimized_metrics:
                ax = axes[0, 0]
                throughputs = [
                    report.baseline_metrics.get('encoding_throughput', 0),
                    report.optimized_metrics.get('encoding_throughput', 0)
                ]
                ax.bar(['Baseline', 'Optimized'], throughputs, color=['blue', 'green'])
                ax.set_ylabel('Throughput (ops/sec)')
                ax.set_title('Encoding Throughput')
            
            # Latency comparison
            if 'p99_latency' in report.optimized_metrics:
                ax = axes[0, 1]
                latencies = [
                    report.baseline_metrics.get('p99_latency', 0) * 1000,
                    report.optimized_metrics.get('p99_latency', 0) * 1000
                ]
                ax.bar(['Baseline', 'Optimized'], latencies, color=['blue', 'green'])
                ax.set_ylabel('P99 Latency (ms)')
                ax.set_title('P99 Latency')
            
            # Memory comparison
            if 'memory_mb' in report.optimized_metrics:
                ax = axes[1, 0]
                memories = [
                    report.baseline_metrics.get('memory_mb', 0),
                    report.optimized_metrics.get('memory_mb', 0)
                ]
                ax.bar(['Baseline', 'Optimized'], memories, color=['blue', 'green'])
                ax.set_ylabel('Memory (MB)')
                ax.set_title('Memory Usage')
            
            # Bottleneck distribution
            ax = axes[1, 1]
            if report.bottleneck_analysis:
                bottlenecks = list(report.bottleneck_analysis.keys())
                counts = [1] * len(bottlenecks)
                ax.pie(counts, labels=bottlenecks, autopct='%1.1f%%')
                ax.set_title('Bottleneck Distribution')
            
            plt.suptitle('REV System Performance Analysis')
            plt.tight_layout()
            plt.savefig(output_path / 'optimization_impact.png')
            plt.close()


def main():
    """Main function to run performance optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='REV Performance Optimization Tool')
    parser.add_argument('--profile', action='store_true', help='Run profiling')
    parser.add_argument('--optimize', action='store_true', help='Run auto-tuning')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--baseline', type=str, help='Baseline metrics file')
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, help='REV configuration file')
    parser.add_argument('--save-config', type=str, help='Save optimized configuration')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Create config from dict (simplified for example)
        config = REVConfig()
    else:
        # Default configuration
        config = REVConfig(
            hdc_config=HDCConfig(
                dimension=10000,
                use_sparse=True,
                sparse_density=0.01
            ),
            segment_config=SegmentConfig(
                segment_size=512,
                buffer_size=1024
            ),
            sequential_config=SequentialConfig(
                alpha=0.05,
                beta=0.10
            ),
            batch_size=32
        )
    
    profiling_result = ProfilingResult()
    optimization_result = OptimizationResult()
    
    if args.profile:
        print("Running performance profiling...")
        profiler = PerformanceProfiler(str(output_dir / "profiling"))
        
        # Profile REV pipeline
        pipeline = REVPipeline(config)
        
        # Generate test data
        test_samples = [np.random.randn(config.hdc_config.dimension) for _ in range(100)]
        
        # Profile encoding
        with profiler.profile_section("encoding"):
            for sample in test_samples:
                _ = pipeline.hdc_encoder.encode(sample)
        
        # Profile verification
        with profiler.profile_section("verification"):
            for i in range(len(test_samples) - 1):
                _ = pipeline.hdc_encoder.compute_similarity(
                    test_samples[i], test_samples[i + 1]
                )
        
        # Collect profiling results
        profiling_result.io_stats = profiler.io_counters
        profiling_result.bottlenecks = profiler.detect_bottlenecks(profiling_result)
        
        print(f"Profiling complete. Found {len(profiling_result.bottlenecks)} bottlenecks.")
    
    if args.optimize:
        print("Running auto-tuning optimization...")
        tuner = AutoTuner(config)
        optimization_result = tuner.run_full_optimization()
        
        print(f"Optimization complete. Performance gain: {optimization_result.performance_gain:.1f}%")
        print(f"Optimal configuration:")
        for param, value in optimization_result.config_changes.items():
            print(f"  {param}: {value}")
        
        # Save optimized configuration if requested
        if args.save_config:
            optimized_config = {
                "batch_size": optimization_result.optimal_batch_size,
                "thread_count": optimization_result.optimal_thread_count,
                "buffer_size": optimization_result.optimal_buffer_size,
                "lut_size": optimization_result.optimal_lut_size,
                "cache_size": optimization_result.optimal_cache_size,
                "segment_size": optimization_result.optimal_segment_size
            }
            with open(args.save_config, 'w') as f:
                json.dump(optimized_config, f, indent=2)
            print(f"Optimized configuration saved to {args.save_config}")
    
    if args.report:
        print("Generating performance report...")
        reporter = PerformanceReporter(args.baseline)
        
        # Collect current metrics
        tuner = AutoTuner(config)
        current_metrics = tuner.benchmark_configuration(
            config, tuner.test_samples
        )
        
        # Generate report
        report = reporter.generate_report(
            profiling_result,
            optimization_result,
            current_metrics
        )
        
        # Save report
        report_file = output_dir / "performance_report.json"
        reporter.save_report(report, str(report_file))
        print(f"Report saved to {report_file}")
        
        # Generate visualizations
        reporter.generate_visualization(report, str(output_dir / "charts"))
        print(f"Visualizations saved to {output_dir / 'charts'}")
        
        # Print summary
        print("\n=== Performance Report Summary ===")
        if report.regression_detected:
            print("⚠️  Performance regression detected:")
            for regression in report.regression_details:
                print(f"  - {regression}")
        else:
            print("✅ No performance regression detected")
        
        if report.recommendations:
            print("\n📊 Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        if report.bottleneck_analysis:
            print("\n🔍 Bottleneck Analysis:")
            for component, issue in report.bottleneck_analysis.items():
                print(f"  - {component}: {issue}")
        
        # Save current metrics as new baseline if requested
        if not args.baseline:
            baseline_file = output_dir / "baseline_metrics.json"
            with open(baseline_file, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            print(f"\nCurrent metrics saved as baseline: {baseline_file}")


if __name__ == "__main__":
    main()