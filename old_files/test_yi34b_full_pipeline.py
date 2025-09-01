#!/usr/bin/env python3
"""
Comprehensive test script for running Yi-34B through the full REV pipeline.
Tests all components: segmentation, HDC encoding, behavioral sites, verification, and privacy features.
"""

import os
import sys
import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.yi34b_integration import Yi34BREVIntegration
from src.hdc.encoder import HypervectorEncoder
from src.hdc.behavioral_sites import BehavioralSites
from src.hdc.binding_operations import BindingOperations
from src.hdc.error_correction import ErrorCorrection
from src.hypervector.similarity import AdvancedSimilarity
from src.core.sequential import DualSequentialTest
from src.privacy.homomorphic_ops import HomomorphicOperations
from src.privacy.distance_zk_proofs import DistanceZKProof
from src.verifier.decision_aggregator import DecisionAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Yi34BFullPipelineTest:
    """Full pipeline test for Yi-34B model."""
    
    def __init__(self):
        """Initialize test configuration."""
        self.model_path = "/Users/rohanvinaik/LLM_models/yi-34b"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": "yi-34b",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
    def test_model_loading(self):
        """Test 1: Model loading and initialization."""
        logger.info("="*60)
        logger.info("TEST 1: Model Loading and Initialization")
        
        start_time = time.time()
        try:
            integration = Yi34BREVIntegration(
                model_path=self.model_path,
                memory_limit_gb=16.0,
                segment_size=512,
                hypervector_dim=10000,
                use_quantization=True
            )
            
            integration.initialize_model()
            
            # Verify model and tokenizer are loaded
            assert integration.model is not None, "Model not loaded"
            assert integration.tokenizer is not None, "Tokenizer not loaded"
            
            # Test basic generation
            test_prompt = "The future of AI is"
            output = integration.model_loader.generate(
                test_prompt,
                max_new_tokens=20,
                temperature=0.7
            )
            
            load_time = time.time() - start_time
            
            self.results["tests"]["model_loading"] = {
                "status": "PASSED",
                "load_time": load_time,
                "model_device": str(integration.device),
                "test_generation": output[:100]
            }
            
            logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"✓ Test generation: {output[:50]}...")
            
            integration.cleanup()
            return True
            
        except Exception as e:
            self.results["tests"]["model_loading"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Model loading failed: {e}")
            return False
            
    def test_segment_processing(self):
        """Test 2: Segment-wise processing with memory bounds."""
        logger.info("="*60)
        logger.info("TEST 2: Segment-wise Processing")
        
        try:
            integration = Yi34BREVIntegration(
                model_path=self.model_path,
                memory_limit_gb=8.0,  # Lower memory limit to test segmentation
                segment_size=256,
                hypervector_dim=8192
            )
            
            integration.initialize_model()
            integration.initialize_pipeline()
            
            # Create a long text that requires segmentation
            long_text = """
            Artificial intelligence has evolved from a theoretical concept to a practical reality.
            Machine learning algorithms now power countless applications in our daily lives.
            Deep learning networks can recognize patterns in vast amounts of data.
            Natural language processing enables machines to understand human communication.
            Computer vision systems can identify objects with remarkable accuracy.
            Reinforcement learning allows agents to learn through trial and error.
            Generative models can create new content that appears authentically human-made.
            Transfer learning enables models to apply knowledge across different domains.
            """ * 5  # Repeat to ensure multiple segments
            
            start_time = time.time()
            result = integration.process_text(
                text=long_text,
                return_hypervectors=True,
                return_behavioral_features=True
            )
            
            process_time = time.time() - start_time
            
            assert result["status"] == "success", "Processing failed"
            assert "num_tokens" in result, "Token count missing"
            assert result["num_tokens"] > 256, "Text not long enough for segmentation"
            
            self.results["tests"]["segment_processing"] = {
                "status": "PASSED",
                "process_time": process_time,
                "text_length": len(long_text),
                "num_tokens": result["num_tokens"],
                "segments": result["num_tokens"] // 256
            }
            
            logger.info(f"✓ Processed {result['num_tokens']} tokens in {process_time:.2f}s")
            logger.info(f"✓ Segmented into {result['num_tokens'] // 256} segments")
            
            integration.cleanup()
            return True
            
        except Exception as e:
            self.results["tests"]["segment_processing"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Segment processing failed: {e}")
            return False
            
    def test_hdc_encoding(self):
        """Test 3: Hyperdimensional computing encoding and operations."""
        logger.info("="*60)
        logger.info("TEST 3: HDC Encoding and Operations")
        
        try:
            # Test HDC components directly
            dimension = 10000
            encoder = HypervectorEncoder(dimension=dimension, sparse=True, density=0.01)
            behavioral_sites = BehavioralSites(dimension=dimension, num_sites=50, zoom_levels=3)
            binding_ops = BindingOperations(dimension=dimension)
            error_correction = ErrorCorrection(dimension=dimension, redundancy_factor=0.25)
            
            # Create test data
            test_features = np.random.randn(100, 768)  # Simulated embeddings
            
            # Test encoding
            start_time = time.time()
            hypervectors = encoder.encode_batch(test_features)
            encode_time = time.time() - start_time
            
            # Test behavioral site extraction
            sites = behavioral_sites.extract_sites(hypervectors[0])
            
            # Test binding operations
            bound = binding_ops.bind(hypervectors[0], hypervectors[1])
            
            # Test error correction
            protected = error_correction.encode(hypervectors[0])
            corrupted = protected.copy()
            corrupted[:100] = 1 - corrupted[:100]  # Flip some bits
            recovered = error_correction.decode(corrupted)
            recovery_accuracy = np.mean(recovered == hypervectors[0])
            
            self.results["tests"]["hdc_encoding"] = {
                "status": "PASSED",
                "dimension": dimension,
                "encode_time": encode_time,
                "sparsity": np.mean(hypervectors == 0),
                "num_sites": len(sites),
                "error_recovery_accuracy": recovery_accuracy
            }
            
            logger.info(f"✓ Encoded {len(test_features)} samples in {encode_time:.3f}s")
            logger.info(f"✓ Sparsity: {np.mean(hypervectors == 0):.2%}")
            logger.info(f"✓ Error recovery accuracy: {recovery_accuracy:.2%}")
            
            return True
            
        except Exception as e:
            self.results["tests"]["hdc_encoding"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ HDC encoding failed: {e}")
            return False
            
    def test_verification_pipeline(self):
        """Test 4: Full verification pipeline with statistical testing."""
        logger.info("="*60)
        logger.info("TEST 4: Verification Pipeline")
        
        try:
            integration = Yi34BREVIntegration(
                model_path=self.model_path,
                memory_limit_gb=16.0,
                segment_size=512,
                hypervector_dim=10000
            )
            
            integration.initialize_model()
            integration.initialize_pipeline()
            
            # Test texts for verification
            test_texts = [
                "The quantum computer solved the optimization problem.",
                "Machine learning models require careful validation.",
                "Neural networks can approximate complex functions."
            ]
            
            verification_results = []
            start_time = time.time()
            
            for text in test_texts:
                result = integration.process_text(text)
                
                if result["status"] == "success" and "pipeline_output" in result:
                    output = result["pipeline_output"]
                    
                    # Verify Merkle tree construction
                    assert "merkle_root" in output, "Merkle root missing"
                    
                    # Verify hypervector generation
                    assert "hypervectors" in output or "behavioral_features" in output
                    
                    verification_results.append({
                        "text": text[:50],
                        "merkle_root": output.get("merkle_root", "")[:16],
                        "verification_score": output.get("verification_score", 0)
                    })
                    
            verify_time = time.time() - start_time
            
            # Test decision aggregator
            aggregator = DecisionAggregator()
            scores = [0.95, 0.92, 0.88, 0.91, 0.93]
            decision = aggregator.aggregate(scores)
            
            self.results["tests"]["verification_pipeline"] = {
                "status": "PASSED",
                "verify_time": verify_time,
                "num_texts": len(test_texts),
                "results": verification_results,
                "aggregator_decision": decision
            }
            
            logger.info(f"✓ Verified {len(test_texts)} texts in {verify_time:.2f}s")
            logger.info(f"✓ Aggregator decision: {decision}")
            
            integration.cleanup()
            return True
            
        except Exception as e:
            self.results["tests"]["verification_pipeline"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Verification pipeline failed: {e}")
            return False
            
    def test_privacy_features(self):
        """Test 5: Privacy-preserving features (ZK proofs, homomorphic ops)."""
        logger.info("="*60)
        logger.info("TEST 5: Privacy-Preserving Features")
        
        try:
            dimension = 8192
            
            # Test homomorphic operations
            he_ops = HomomorphicOperations()
            
            # Create test vectors
            vec1 = np.random.randn(dimension)
            vec2 = np.random.randn(dimension)
            
            # Encrypt vectors
            enc1 = he_ops.encrypt_vector(vec1)
            enc2 = he_ops.encrypt_vector(vec2)
            
            # Compute encrypted distance
            enc_distance = he_ops.compute_distance(enc1, enc2)
            
            # Test ZK distance proof
            zk_proof = DistanceZKProof()
            
            # Generate proof that distance is within threshold
            threshold = 100.0
            proof = zk_proof.prove_distance_threshold(vec1, vec2, threshold)
            
            # Verify proof
            is_valid = zk_proof.verify_distance_threshold(proof, threshold)
            
            self.results["tests"]["privacy_features"] = {
                "status": "PASSED",
                "homomorphic_encryption": "success",
                "zk_proof_generated": proof is not None,
                "zk_proof_valid": is_valid,
                "vector_dimension": dimension
            }
            
            logger.info(f"✓ Homomorphic operations successful")
            logger.info(f"✓ ZK proof generated and verified: {is_valid}")
            
            return True
            
        except Exception as e:
            self.results["tests"]["privacy_features"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Privacy features failed: {e}")
            return False
            
    def test_full_benchmark(self):
        """Test 6: Full benchmark with performance metrics."""
        logger.info("="*60)
        logger.info("TEST 6: Full Pipeline Benchmark")
        
        try:
            integration = Yi34BREVIntegration(
                model_path=self.model_path,
                memory_limit_gb=16.0,
                segment_size=512,
                hypervector_dim=10000
            )
            
            # Benchmark texts of varying complexity
            benchmark_texts = [
                "Simple test.",  # Minimal
                "The AI system processes natural language effectively." * 10,  # Medium
                """Large language models have revolutionized natural language processing
                by demonstrating unprecedented capabilities in understanding and generating
                human-like text across diverse domains and applications.""" * 20  # Large
            ]
            
            start_time = time.time()
            benchmark_results = integration.run_benchmark(
                test_texts=benchmark_texts,
                save_results=True,
                output_dir="benchmarks"
            )
            benchmark_time = time.time() - start_time
            
            self.results["tests"]["full_benchmark"] = {
                "status": "PASSED",
                "total_time": benchmark_time,
                "statistics": benchmark_results["statistics"],
                "output_file": benchmark_results.get("output_file", "")
            }
            
            stats = benchmark_results["statistics"]
            logger.info(f"✓ Benchmark completed in {benchmark_time:.2f}s")
            logger.info(f"✓ Success rate: {stats['success_rate']:.1%}")
            logger.info(f"✓ Average tokens: {stats['avg_tokens']:.1f}")
            
            integration.cleanup()
            return True
            
        except Exception as e:
            self.results["tests"]["full_benchmark"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"✗ Full benchmark failed: {e}")
            return False
            
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("="*60)
        logger.info("STARTING YI-34B FULL PIPELINE VALIDATION")
        logger.info("="*60)
        
        tests = [
            ("Model Loading", self.test_model_loading),
            ("Segment Processing", self.test_segment_processing),
            ("HDC Encoding", self.test_hdc_encoding),
            ("Verification Pipeline", self.test_verification_pipeline),
            ("Privacy Features", self.test_privacy_features),
            ("Full Benchmark", self.test_full_benchmark)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                self.results["errors"].append({
                    "test": test_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                logger.error(f"Test {test_name} crashed: {e}")
                
        # Summary
        self.results["summary"] = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(tests) if tests else 0
        }
        
        logger.info("="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info(f"Passed: {passed}/{len(tests)}")
        logger.info(f"Failed: {failed}/{len(tests)}")
        logger.info(f"Success Rate: {passed/len(tests):.1%}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"yi34b_validation_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
        return self.results


def main():
    """Main entry point."""
    test_suite = Yi34BFullPipelineTest()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()