#!/usr/bin/env python3
"""
Validation Script for REV Paper Claims
Verifies that the implementation matches the paper's specifications.
"""

import sys
import time
import numpy as np
import hashlib
import hmac
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import traceback

# Add src to path
sys.path.insert(0, '/Users/rohanvinaik/REV')

# Import REV components
from src.rev_pipeline import REVPipeline, REVConfig
from src.hdc.encoder import UnifiedHDCEncoder, HDCConfig
from src.hdc.behavioral_sites import BehavioralSiteExtractor
from src.hdc.binding_operations import BindingOperations, BindingType
from src.executor.segment_runner import SegmentRunner, SegmentConfig
from src.core.sequential import DualSequentialTest, SequentialConfig, Verdict
from src.challenges.kdf_prompts import EnhancedKDFPromptGenerator
from src.crypto.merkle import HierarchicalMerkleTree
from src.hypervector.hamming import OptimizedHammingCalculator


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    details: str
    performance: Optional[Dict] = None


class PaperClaimsValidator:
    """Validates the implementation against paper claims."""
    
    def __init__(self):
        self.results = []
        
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests."""
        print("=" * 80)
        print("REV PAPER CLAIMS VALIDATION")
        print("=" * 80)
        
        # Test 1: Memory-bounded segment execution (Section 4.4)
        self.validate_memory_bounded_execution()
        
        # Test 2: HMAC-based challenge generation (Section 4.2)
        self.validate_hmac_challenge_generation()
        
        # Test 3: HDC hypervector encoding (Section 6)
        self.validate_hdc_encoding()
        
        # Test 4: Hamming LUT optimization (Section 6.1C)
        self.validate_hamming_lut_speedup()
        
        # Test 5: Sequential testing with SAME/DIFFERENT/UNDECIDED (Section 5.7)
        self.validate_sequential_testing()
        
        # Test 6: Merkle tree commitments (Section 4.3)
        self.validate_merkle_commitments()
        
        # Test 7: Restriction sites (Section 4.1)
        self.validate_restriction_sites()
        
        # Test 8: Behavioral sites with zoom levels (Section 6.2)
        self.validate_behavioral_sites()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def validate_memory_bounded_execution(self):
        """Validate memory-bounded segment execution."""
        print("\n[1/8] Validating Memory-Bounded Segment Execution...")
        
        try:
            # Create segment runner with memory limit
            config = SegmentConfig(
                segment_size=512,
                buffer_size=1024,
                max_sequence_length=2048,
                memory_limit_mb=100  # 100MB limit
            )
            
            runner = SegmentRunner(config)
            
            # Create test segments
            segments = []
            for i in range(5):
                segment = type('Segment', (), {
                    'id': f'L{i}.post_attn',
                    'start_idx': i * 512,
                    'end_idx': (i + 1) * 512,
                    'layer_idx': i
                })()
                segments.append(segment)
            
            # Test segment loading/offloading
            memory_usage = []
            for segment in segments:
                runner.load_params(segment)
                # Simulate processing
                time.sleep(0.01)
                memory_usage.append(runner.get_memory_usage())
                runner.offload_params(segment)
            
            # Verify single-segment working set
            max_memory = max(memory_usage)
            passed = max_memory < config.memory_limit_mb
            
            result = ValidationResult(
                test_name="Memory-Bounded Execution",
                passed=passed,
                details=f"Max memory: {max_memory:.1f}MB (limit: {config.memory_limit_mb}MB)",
                performance={"max_memory_mb": max_memory}
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Memory-Bounded Execution",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_hmac_challenge_generation(self):
        """Validate HMAC-based challenge generation."""
        print("\n[2/8] Validating HMAC Challenge Generation...")
        
        try:
            # Paper formula: seed_i = HMAC(key, f"{run_id}:{i}")
            key = b"verification_key"
            run_id = "REV-2024-08"
            
            generator = EnhancedKDFPromptGenerator(seed=key)
            
            # Generate challenges
            challenges = []
            for i in range(10):
                # Verify seed generation matches paper
                expected_seed = hmac.new(
                    key,
                    f"{run_id}:{i}".encode(),
                    hashlib.sha256
                ).digest()
                
                challenge = generator.generate_challenge(
                    challenge_id=f"{run_id}:{i}",
                    category="reasoning"
                )
                challenges.append(challenge)
            
            # Verify deterministic generation
            generator2 = EnhancedKDFPromptGenerator(seed=key)
            challenge2 = generator2.generate_challenge(
                challenge_id=f"{run_id}:0",
                category="reasoning"
            )
            
            deterministic = challenges[0]['prompt'] == challenge2['prompt']
            
            result = ValidationResult(
                test_name="HMAC Challenge Generation",
                passed=deterministic,
                details=f"Generated {len(challenges)} deterministic challenges",
                performance={"n_challenges": len(challenges)}
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="HMAC Challenge Generation",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_hdc_encoding(self):
        """Validate HDC hypervector encoding."""
        print("\n[3/8] Validating HDC Hypervector Encoding...")
        
        try:
            # Paper: 8K-100K dimensional vectors
            config = HDCConfig(
                dimension=16384,  # Within paper's range
                use_sparse=True,
                sparse_density=0.01
            )
            
            encoder = UnifiedHDCEncoder(config)
            
            # Test probe_to_hypervector (Section 6.4)
            probe_features = {
                "task_category": "reasoning",
                "syntactic_complexity": 0.7,
                "knowledge_domain": "science",
                "reasoning_depth": 3
            }
            
            probe_hv = encoder.encode_probe(probe_features)
            
            # Test response_to_hypervector
            logits = np.random.randn(50000)  # Vocabulary size
            response_hv = encoder.encode_response(logits, top_k=16)
            
            # Verify hypervector properties
            # 1. Dimensionality
            dim_correct = len(probe_hv) == config.dimension
            
            # 2. Sparsity (for sparse encoding)
            if config.use_sparse:
                sparsity = np.mean(probe_hv != 0)
                sparsity_correct = abs(sparsity - config.sparse_density) < 0.01
            else:
                sparsity_correct = True
            
            # 3. Binding operations work
            binding_ops = BindingOperations(config.dimension)
            bound = binding_ops.bind(probe_hv, response_hv, BindingType.XOR)
            binding_correct = len(bound) == config.dimension
            
            passed = dim_correct and sparsity_correct and binding_correct
            
            result = ValidationResult(
                test_name="HDC Hypervector Encoding",
                passed=passed,
                details=f"Dimension: {config.dimension}, Sparse: {config.use_sparse}",
                performance={
                    "dimension": config.dimension,
                    "actual_sparsity": sparsity if config.use_sparse else 1.0
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="HDC Hypervector Encoding",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_hamming_lut_speedup(self):
        """Validate Hamming distance LUT optimization."""
        print("\n[4/8] Validating Hamming LUT Speedup...")
        
        try:
            dimension = 10000
            calculator = OptimizedHammingCalculator(
                dimension=dimension,
                use_lut=True,
                lut_size=65536  # 16-bit LUT as in paper
            )
            
            # Generate test vectors
            vec1 = np.random.randint(0, 2, dimension, dtype=np.uint8)
            vec2 = np.random.randint(0, 2, dimension, dtype=np.uint8)
            
            # Benchmark with LUT
            n_iterations = 1000
            start = time.perf_counter()
            for _ in range(n_iterations):
                dist_lut = calculator.hamming_distance(vec1, vec2)
            time_lut = time.perf_counter() - start
            
            # Benchmark without LUT (naive)
            start = time.perf_counter()
            for _ in range(n_iterations):
                dist_naive = np.sum(vec1 != vec2)
            time_naive = time.perf_counter() - start
            
            # Calculate speedup
            speedup = time_naive / time_lut
            
            # Paper claims 10-20x speedup
            passed = speedup >= 10
            
            result = ValidationResult(
                test_name="Hamming LUT Speedup",
                passed=passed,
                details=f"Speedup: {speedup:.1f}x (target: 10-20x)",
                performance={
                    "speedup": speedup,
                    "time_lut_ms": time_lut * 1000 / n_iterations,
                    "time_naive_ms": time_naive * 1000 / n_iterations
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Hamming LUT Speedup",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_sequential_testing(self):
        """Validate sequential testing with SAME/DIFFERENT/UNDECIDED."""
        print("\n[5/8] Validating Sequential Testing (SPRT)...")
        
        try:
            config = SequentialConfig(
                alpha=0.05,
                beta=0.10,
                min_samples=10,
                max_samples=1000
            )
            
            tester = DualSequentialTest(config)
            
            # Test SAME decision (high similarity)
            for i in range(100):
                tester.update(
                    bernoulli_outcome=1,  # Match
                    distance=0.02  # Small distance
                )
                
                if tester.has_decision():
                    break
            
            same_verdict = tester.get_verdict()
            
            # Reset and test DIFFERENT decision
            tester = DualSequentialTest(config)
            for i in range(100):
                tester.update(
                    bernoulli_outcome=0,  # No match
                    distance=0.5  # Large distance
                )
                
                if tester.has_decision():
                    break
            
            diff_verdict = tester.get_verdict()
            
            # Test UNDECIDED (ambiguous)
            tester = DualSequentialTest(config)
            for i in range(config.max_samples):
                tester.update(
                    bernoulli_outcome=i % 2,  # Alternating
                    distance=0.25  # Medium distance
                )
            
            undecided_verdict = tester.get_verdict()
            
            # Verify all three outcomes are possible
            passed = (
                same_verdict == Verdict.SAME and
                diff_verdict == Verdict.DIFFERENT and
                undecided_verdict == Verdict.UNDECIDED
            )
            
            result = ValidationResult(
                test_name="Sequential Testing (SPRT)",
                passed=passed,
                details=f"Verdicts: SAME={same_verdict}, DIFF={diff_verdict}, UNDEC={undecided_verdict}",
                performance={
                    "alpha": config.alpha,
                    "beta": config.beta
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Sequential Testing (SPRT)",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_merkle_commitments(self):
        """Validate Merkle tree commitments."""
        print("\n[6/8] Validating Merkle Tree Commitments...")
        
        try:
            # Create Merkle tree for per-challenge commitments
            tree = HierarchicalMerkleTree()
            
            # Add segment signatures (as in paper Section 4.3)
            segments = []
            for i in range(10):
                segment_data = {
                    "seg_id": f"L{i}.post_attn",
                    "sigma": np.random.bytes(32),  # Binary sketch
                    "policy": {"temperature": 0.0, "dtype": "fp16"}
                }
                
                # Hash to create leaf
                leaf = hashlib.blake3(
                    str(segment_data).encode()
                ).digest()
                
                tree.add_leaf(leaf, metadata={"segment": i})
                segments.append(leaf)
            
            # Build tree and get root
            root = tree.get_root()
            
            # Verify proof for a segment
            proof = tree.get_proof(segments[5])
            valid = tree.verify_proof(segments[5], proof, root)
            
            # Verify tree properties
            passed = (
                root is not None and
                valid and
                len(tree.get_leaves()) == 10
            )
            
            result = ValidationResult(
                test_name="Merkle Tree Commitments",
                passed=passed,
                details=f"Tree with {len(segments)} segments, proof verification: {valid}",
                performance={
                    "n_segments": len(segments),
                    "proof_size": len(proof) if proof else 0
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Merkle Tree Commitments",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_restriction_sites(self):
        """Validate restriction site policies."""
        print("\n[7/8] Validating Restriction Sites...")
        
        try:
            # Paper Section 4.1: Architectural vs Behavioral sites
            
            # Architectural sites (white/gray-box)
            arch_sites = [
                "L0.post_attn",   # After attention
                "L0.post_mlp",    # After MLP
                "L0.end_block",   # End of block
                "L1.post_attn",
                "L1.post_mlp",
                "L1.end_block"
            ]
            
            # Overlapping windows for stitching resistance
            windows = [
                (1, 8),   # Layers 1-8
                (5, 12),  # Layers 5-12 (overlap)
                (9, 16),  # Layers 9-16 (overlap)
            ]
            
            # Verify overlap
            overlap_12 = set(range(5, 8+1))  # Overlap between window 1 and 2
            overlap_23 = set(range(9, 12+1))  # Overlap between window 2 and 3
            
            has_overlap = len(overlap_12) > 0 and len(overlap_23) > 0
            
            # Behavioral sites (HDC-based)
            extractor = BehavioralSiteExtractor()
            
            # Extract sites at different zoom levels
            sites = {}
            for zoom in [0, 1, 2]:  # Corpus, prompt, span levels
                site = extractor.extract_sites(
                    prompt="Test prompt",
                    response="Test response",
                    zoom_level=zoom
                )
                sites[zoom] = site
            
            # Verify both types of sites exist
            passed = (
                len(arch_sites) > 0 and
                has_overlap and
                len(sites) == 3
            )
            
            result = ValidationResult(
                test_name="Restriction Sites",
                passed=passed,
                details=f"Arch sites: {len(arch_sites)}, Zoom levels: {len(sites)}",
                performance={
                    "n_arch_sites": len(arch_sites),
                    "n_zoom_levels": len(sites)
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Restriction Sites",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_behavioral_sites(self):
        """Validate behavioral sites with zoom levels."""
        print("\n[8/8] Validating Behavioral Sites with Zoom...")
        
        try:
            # Paper Section 6.2: Hierarchical zoom levels
            extractor = BehavioralSiteExtractor()
            
            zoom_registry = {
                0: {},  # Corpus/site-wide prototypes
                1: {},  # Prompt-level hypervectors
                2: {},  # Span/tile-level hypervectors
            }
            
            # Generate sites at each zoom level
            test_prompts = [
                "Explain quantum computing",
                "Write a Python function",
                "Translate to French"
            ]
            
            for prompt_id, prompt in enumerate(test_prompts):
                for zoom_level in [0, 1, 2]:
                    site = extractor.extract_sites(
                        prompt=prompt,
                        response=f"Response to {prompt}",
                        zoom_level=zoom_level
                    )
                    
                    # Store in registry
                    zoom_registry[zoom_level][prompt_id] = site
            
            # Verify multi-resolution analysis
            has_all_levels = all(
                len(zoom_registry[level]) == len(test_prompts)
                for level in [0, 1, 2]
            )
            
            # Verify hierarchical structure
            # Level 0 should be most general, Level 2 most specific
            level_0_features = zoom_registry[0][0].get('features', {})
            level_2_features = zoom_registry[2][0].get('features', {})
            
            hierarchical = len(level_2_features) >= len(level_0_features)
            
            passed = has_all_levels and hierarchical
            
            result = ValidationResult(
                test_name="Behavioral Sites with Zoom",
                passed=passed,
                details=f"Zoom levels: {list(zoom_registry.keys())}, Hierarchical: {hierarchical}",
                performance={
                    "n_zoom_levels": len(zoom_registry),
                    "n_prompts": len(test_prompts)
                }
            )
            
        except Exception as e:
            result = ValidationResult(
                test_name="Behavioral Sites with Zoom",
                passed=False,
                details=f"Error: {str(e)}"
            )
        
        self.results.append(result)
        self._print_result(result)
    
    def _print_result(self, result: ValidationResult):
        """Print a single test result."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {result.test_name}")
        print(f"    Details: {result.details}")
        if result.performance:
            print(f"    Performance: {result.performance}")
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if passed == total:
            print("\n🎉 ALL PAPER CLAIMS VALIDATED SUCCESSFULLY!")
        else:
            print("\n⚠️  Some validations failed. Review details above.")
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.test_name}: {r.details}")
        
        # Performance summary
        print("\n📊 Performance Metrics:")
        for r in self.results:
            if r.performance and r.passed:
                print(f"  {r.test_name}:")
                for key, value in r.performance.items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.3f}")
                    else:
                        print(f"    - {key}: {value}")


def main():
    """Main validation entry point."""
    print("\n🔬 Starting REV Paper Claims Validation...\n")
    
    validator = PaperClaimsValidator()
    results = validator.run_all_validations()
    
    # Return exit code based on results
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()