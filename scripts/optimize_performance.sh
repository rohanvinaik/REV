#!/bin/bash
# Mac Pro 2013 Performance Optimization

echo "⚡ Optimizing Mac Pro for Computational Experiments..."

# 1. Configure for maximum performance
sudo pmset -a sleep 0
sudo pmset -a disksleep 0
sudo pmset -a hibernatemode 0
sudo pmset -a autopoweroff 0

# 2. Optimize memory pressure
sudo sysctl -w vm.swappiness=10
sudo sysctl -w kern.maxvnodes=300000

# 3. Create GPU utilization script for AMD FirePro
cat > ~/experiments/gpu_benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
GPU Benchmark for Mac Pro AMD FirePro
Tests OpenCL performance for HD computing
"""

import numpy as np
import time
import pyopencl as cl
import pyopencl.array as cl_array

class AMDGPUBenchmark:
    def __init__(self):
        # Get AMD GPU
        platforms = cl.get_platforms()
        self.device = None
        for platform in platforms:
            if 'AMD' in platform.name:
                self.device = platform.get_devices()[0]
                break
        
        if not self.device:
            print("No AMD GPU found!")
            return
            
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
    def benchmark_hd_operations(self, dimension=10000, num_vectors=1000):
        """Benchmark HD vector operations on GPU"""
        print(f"\n🎮 GPU HD Computing Benchmark")
        print(f"  Device: {self.device.name}")
        print(f"  Dimension: {dimension}")
        print(f"  Vectors: {num_vectors}")
        
        # Create random binary vectors
        vectors = np.random.randint(0, 2, (num_vectors, dimension)).astype(np.int32)
        
        # OpenCL kernel for Hamming distance
        kernel_code = """
        __kernel void hamming_distance(
            __global const int* vec1,
            __global const int* vec2,
            __global float* result,
            const int dimension)
        {
            int gid = get_global_id(0);
            int differences = 0;
            
            for (int i = 0; i < dimension; i++) {
                if (vec1[gid * dimension + i] != vec2[i]) {
                    differences++;
                }
            }
            
            result[gid] = 1.0f - (float)differences / dimension;
        }
        """
        
        # Compile kernel
        program = cl.Program(self.context, kernel_code).build()
        
        # Transfer to GPU
        gpu_vectors = cl_array.to_device(self.queue, vectors)
        gpu_query = cl_array.to_device(self.queue, vectors[0])
        gpu_result = cl_array.empty(self.queue, num_vectors, dtype=np.float32)
        
        # Benchmark
        start = time.time()
        
        program.hamming_distance(
            self.queue,
            (num_vectors,),
            None,
            gpu_vectors.data,
            gpu_query.data,
            gpu_result.data,
            np.int32(dimension)
        )
        
        self.queue.finish()
        gpu_time = time.time() - start
        
        # CPU comparison
        start = time.time()
        cpu_result = np.array([
            np.sum(vectors[0] == vec) / dimension 
            for vec in vectors
        ])
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"\n  Results:")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  CPU time: {cpu_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        return speedup

if __name__ == '__main__':
    try:
        benchmark = AMDGPUBenchmark()
        benchmark.benchmark_hd_operations(10000, 1000)
        benchmark.benchmark_hd_operations(50000, 1000)
        benchmark.benchmark_hd_operations(100000, 1000)
    except Exception as e:
        print(f"GPU benchmark failed: {e}")
        print("Make sure PyOpenCL is installed: pip install pyopencl")
EOF

# 4. Create efficiency testing framework based on your reference guide
cat > ~/experiments/efficiency_methods.py << 'EOF'
#!/usr/bin/env python3
"""
Test computational efficiency methods from reference guide
Implementations for Mac Pro experiments
"""

import numpy as np
import time
from abc import ABC, abstractmethod

class EfficiencyMethod(ABC):
    """Base class for efficiency methods"""
    
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def test(self, data_size):
        pass

class HyperdimensionalComputing(EfficiencyMethod):
    """Test HD computing efficiency"""
    
    def name(self):
        return "Hyperdimensional Computing (10,000D)"
    
    def test(self, data_size):
        dimension = 10000
        num_items = data_size
        
        # Create item memory
        item_memory = np.random.randint(0, 2, (num_items, dimension))
        
        # Test encoding speed
        start = time.time()
        
        # Bind pairs
        bound_pairs = []
        for i in range(0, num_items-1, 2):
            bound = item_memory[i] ^ item_memory[i+1]
            bound_pairs.append(bound)
        
        # Bundle all
        bundled = np.zeros(dimension)
        for vec in bound_pairs:
            bundled += vec
        bundled = (bundled > len(bound_pairs) / 2).astype(int)
        
        encoding_time = time.time() - start
        
        # Test query
        start = time.time()
        query = item_memory[0]
        similarities = [np.sum(query == item) / dimension for item in item_memory[:100]]
        query_time = time.time() - start
        
        return {
            'encoding_time': encoding_time,
            'query_time': query_time,
            'items_per_second': num_items / encoding_time
        }

class SparseLotteryTicket(EfficiencyMethod):
    """Simulate lottery ticket pruning"""
    
    def name(self):
        return "Lottery Ticket (90% sparse)"
    
    def test(self, data_size):
        # Simulate a neural network layer
        input_size = data_size
        output_size = data_size // 2
        
        # Full network
        weights_full = np.random.randn(input_size, output_size)
        
        # Find lottery ticket (90% pruning)
        threshold = np.percentile(np.abs(weights_full), 90)
        mask = np.abs(weights_full) > threshold
        weights_sparse = weights_full * mask
        
        # Test inference speed
        test_input = np.random.randn(input_size)
        
        # Full network
        start = time.time()
        output_full = np.dot(test_input, weights_full)
        full_time = time.time() - start
        
        # Sparse network
        start = time.time()
        # Simulate sparse computation
        sparse_indices = np.where(mask)
        output_sparse = np.zeros(output_size)
        for i, j in zip(*sparse_indices):
            output_sparse[j] += test_input[i] * weights_sparse[i, j]
        sparse_time = time.time() - start
        
        return {
            'full_time': full_time,
            'sparse_time': sparse_time,
            'speedup': full_time / sparse_time,
            'sparsity': 1 - np.sum(mask) / mask.size
        }

class TopologicalCompression(EfficiencyMethod):
    """Simulate TDA-based compression"""
    
    def name(self):
        return "Topological Data Analysis"
    
    def test(self, data_size):
        # Generate point cloud data
        points = np.random.randn(data_size, 3)
        
        # Simulate persistent homology computation
        start = time.time()
        
        # Compute pairwise distances (simplified)
        distances = np.zeros((min(100, data_size), min(100, data_size)))
        for i in range(min(100, data_size)):
            for j in range(i+1, min(100, data_size)):
                distances[i, j] = np.linalg.norm(points[i] - points[j])
                distances[j, i] = distances[i, j]
        
        # Extract topological features (simplified)
        persistence_pairs = []
        threshold = np.median(distances[distances > 0])
        for i in range(min(100, data_size)):
            birth = np.min(distances[i, distances[i] > 0])
            death = np.max(distances[i])
            if death - birth > threshold:
                persistence_pairs.append((birth, death))
        
        tda_time = time.time() - start
        
        compression_ratio = data_size * 3 * 8 / (len(persistence_pairs) * 2 * 8)
        
        return {
            'computation_time': tda_time,
            'features_extracted': len(persistence_pairs),
            'compression_ratio': compression_ratio
        }

def run_efficiency_comparison(data_sizes=[1000, 5000, 10000]):
    """Compare different efficiency methods"""
    methods = [
        HyperdimensionalComputing(),
        SparseLotteryTicket(),
        TopologicalCompression()
    ]
    
    print("\n" + "="*60)
    print("Computational Efficiency Methods Comparison")
    print("="*60)
    
    for size in data_sizes:
        print(f"\nData size: {size}")
        print("-" * 40)
        
        for method in methods:
            print(f"\n{method.name()}:")
            try:
                results = method.test(size)
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == '__main__':
    run_efficiency_comparison()
EOF

echo "✅ Performance optimization complete!"
echo
echo "Next steps:"
echo "1. Run the setup script: bash ~/mac_pro_server_setup.sh"
echo "2. Set up experiments: bash ~/setup_experiment_env.sh" 
echo "3. Configure monitoring: bash ~/setup_monitoring.sh"
echo "4. Enable remote access: bash ~/setup_remote_access.sh"
echo
echo "Then test with:"
echo "  python3 ~/experiments/genomevault_hd_experiment.py"
echo "  python3 ~/experiments/efficiency_methods.py"
echo "  python3 ~/experiments/monitor_server.py"
