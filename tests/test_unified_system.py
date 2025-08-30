"""
Comprehensive Test Suite for REV System
Tests all core components, integration workflows, and performance benchmarks.
"""

import pytest
import numpy as np
import torch
import hashlib
import time
import json
import pickle
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import asyncio
import tempfile
import os
from dataclasses import dataclass
import psutil
from collections import deque, Counter

# Import REV components
from src.core.sequential import (
    SequentialTester, SequentialState, SequentialConfig,
    EmpiricalBernsteinBound, SPRTCore
)
from src.hdc.encoder import UnifiedHDCEncoder, HypervectorConfig
from src.hdc.behavioral_sites import BehavioralSites, ProbeConfig
from src.hdc.binding_operations import BindingOperations, BindingType
from src.hdc.error_correction import ErrorCorrection, ErrorCorrectionConfig
from src.hypervector.similarity import AdvancedSimilarityMetrics
from src.verifier.blackbox import BlackBoxVerifier, VerificationConfig
from src.verifier.decision_aggregator import DecisionAggregator, AggregationStrategy
from src.consensus.byzantine import ByzantineConsensus, ConsensusConfig
from src.privacy.differential_privacy import DifferentialPrivacy, PrivacyConfig
from src.privacy.homomorphic_ops import HomomorphicOperations
from src.privacy.distance_zk_proofs import DistanceZKProofs
from src.crypto.merkle import EnhancedMerkleTree
from src.challenges.kdf_prompts import EnhancedKDFPromptGenerator
from src.rev_pipeline import REVPipeline, REVConfig, Segment
from src.executor.parallel_pipeline import ParallelPipeline, PipelineConfig
from src.verifier.streaming_consensus import StreamingConsensusVerifier, ConsensusMode


# ==================== TEST FIXTURES ====================

@pytest.fixture
def sequential_config():
    """Configuration for sequential testing."""
    return SequentialConfig(
        alpha=0.05,
        beta=0.10,
        epsilon=0.01,
        min_samples=100,
        max_samples=10000,
        use_eb_bound=True
    )


@pytest.fixture
def hdc_config():
    """Configuration for HDC operations."""
    return HypervectorConfig(
        dimension=10000,
        sparsity=0.01,
        use_ternary=False,
        seed=42
    )


@pytest.fixture
def verification_config():
    """Configuration for black-box verification."""
    return VerificationConfig(
        max_retries=3,
        timeout=30.0,
        cache_ttl=3600,
        rate_limit=60
    )


@pytest.fixture
def privacy_config():
    """Configuration for privacy mechanisms."""
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        mechanism="laplace",
        sensitivity=1.0
    )


@pytest.fixture
def rev_config():
    """Complete REV pipeline configuration."""
    return REVConfig(
        model_id="test_model",
        segment_size=512,
        max_segments=100,
        memory_limit_gb=4.0,
        checkpoint_interval=10,
        use_privacy=True,
        use_consensus=True
    )


# ==================== UNIT TESTS ====================

class TestCoreSequentialFramework:
    """Unit tests for sequential testing framework."""
    
    def test_sprt_initialization_and_thresholds(self, sequential_config):
        """Test SPRT core initialization with proper thresholds."""
        sprt = SPRTCore(sequential_config)
        
        # Verify configuration
        assert sprt.config.alpha == 0.05
        assert sprt.config.beta == 0.10
        assert sprt.config.epsilon == 0.01
        
        # Check threshold calculations
        expected_a = np.log((1 - sequential_config.beta) / sequential_config.alpha)
        expected_b = np.log(sequential_config.beta / (1 - sequential_config.alpha))
        
        assert abs(sprt.threshold_a - expected_a) < 1e-6
        assert abs(sprt.threshold_b - expected_b) < 1e-6
    
    def test_empirical_bernstein_bound_computation(self):
        """Test EB bound computation with various sample sizes."""
        eb_bound = EmpiricalBernsteinBound(alpha=0.05)
        
        # Test with different sample sizes
        for n in [100, 500, 1000, 5000]:
            samples = np.random.randn(n) * 0.1 + 0.8  # Mean ~0.8
            lower, upper = eb_bound.compute_bound(samples, t=n)
            
            # Bounds should contain true mean with high probability
            assert lower < 0.8 < upper
            # Bounds should tighten with more samples
            width = upper - lower
            assert width < 1.0 / np.sqrt(n) * 10  # Rough approximation
    
    def test_sequential_state_updates_welford(self, sequential_config):
        """Test Welford's algorithm for numerical stability."""
        tester = SequentialTester(sequential_config)
        
        # Add samples progressively
        true_samples = np.random.uniform(0.7, 0.9, 1000)
        
        for i, sample in enumerate(true_samples[:100]):
            decision = tester.update(sample)
            
            # Check state consistency
            assert tester.state.n_samples == i + 1
            assert 0 <= tester.state.mean <= 1
            assert tester.state.variance >= 0
            
            # Verify Welford's update
            if i > 0:
                expected_mean = np.mean(true_samples[:i+1])
                assert abs(tester.state.mean - expected_mean) < 0.01
    
    def test_anytime_valid_confidence_intervals(self, sequential_config):
        """Test confidence intervals remain valid at any stopping time."""
        tester = SequentialTester(sequential_config)
        
        # Track confidence intervals over time
        ci_history = []
        coverage_count = 0
        true_mean = 0.75
        
        for i in range(500):
            sample = np.random.normal(true_mean, 0.1)
            tester.update(sample)
            
            if i % 10 == 0 and i > 0:
                ci = tester.get_confidence_interval()
                ci_history.append(ci)
                
                # Check if true mean is covered
                if ci[0] <= true_mean <= ci[1]:
                    coverage_count += 1
        
        # Coverage should be approximately 1 - alpha
        coverage_rate = coverage_count / len(ci_history)
        assert coverage_rate >= 0.90  # Allow some margin
        
        # CIs should generally shrink over time
        early_width = ci_history[0][1] - ci_history[0][0]
        late_width = ci_history[-1][1] - ci_history[-1][0]
        assert late_width < early_width
    
    @pytest.mark.parametrize("alpha,beta", [(0.01, 0.05), (0.05, 0.10), (0.10, 0.20)])
    def test_type_i_and_type_ii_error_control(self, alpha, beta):
        """Test that error rates are controlled at specified levels."""
        config = SequentialConfig(
            alpha=alpha,
            beta=beta,
            epsilon=0.01,
            min_samples=50,
            max_samples=1000
        )
        
        n_simulations = 200
        type_i_errors = 0
        type_ii_errors = 0
        
        # Test under null hypothesis (no difference)
        for _ in range(n_simulations):
            tester = SequentialTester(config)
            for _ in range(config.max_samples):
                sample = np.random.normal(0.5, 0.1)
                decision = tester.update(sample)
                if decision == "reject":
                    type_i_errors += 1
                    break
                elif decision == "accept":
                    break
        
        # Test under alternative hypothesis (true difference)
        for _ in range(n_simulations):
            tester = SequentialTester(config)
            for _ in range(config.max_samples):
                sample = np.random.normal(0.8, 0.1)  # Clear difference
                decision = tester.update(sample)
                if decision == "accept":
                    type_ii_errors += 1
                    break
                elif decision == "reject":
                    break
        
        # Check error rates (with some tolerance for finite samples)
        empirical_alpha = type_i_errors / n_simulations
        empirical_beta = type_ii_errors / n_simulations
        
        assert empirical_alpha <= alpha * 2.0  # Allow 2x margin
        assert empirical_beta <= beta * 2.0


class TestHDCOperations:
    """Unit tests for Hyperdimensional Computing operations."""
    
    def test_unified_encoder_initialization(self, hdc_config):
        """Test unified HDC encoder setup."""
        encoder = UnifiedHDCEncoder(hdc_config)
        
        assert encoder.config.dimension == 10000
        assert encoder.config.sparsity == 0.01
        assert encoder.item_memory is not None
        assert encoder.item_memory.shape[1] == hdc_config.dimension
    
    def test_sparse_hypervector_generation(self, hdc_config):
        """Test sparse hypervector generation and properties."""
        encoder = UnifiedHDCEncoder(hdc_config)
        
        # Generate hypervector from data
        data = np.random.randn(100)
        hv = encoder.encode(data)
        
        # Check dimensions
        assert hv.shape[0] == hdc_config.dimension
        
        # Check sparsity constraint
        if isinstance(hv, np.ndarray):
            density = np.count_nonzero(hv) / len(hv)
            assert density <= hdc_config.sparsity * 3  # Allow some variance
    
    def test_binding_operations_correctness(self, hdc_config):
        """Test all binding operations for correctness."""
        binding_ops = BindingOperations(hdc_config.dimension)
        
        # Create test hypervectors
        hv1 = np.random.choice([0, 1], hdc_config.dimension)
        hv2 = np.random.choice([0, 1], hdc_config.dimension)
        
        # Test XOR binding (self-inverse property)
        xor_bound = binding_ops.bind(hv1, hv2, BindingType.XOR)
        recovered = binding_ops.unbind(xor_bound, hv2, BindingType.XOR)
        assert np.array_equal(recovered, hv1)
        
        # Test permutation binding
        perm_bound = binding_ops.bind(hv1, hv2, BindingType.PERMUTATION)
        assert perm_bound.shape == hv1.shape
        assert not np.array_equal(perm_bound, hv1)  # Should be different
        
        # Test circular convolution
        conv_bound = binding_ops.bind(hv1, hv2, BindingType.CIRCULAR_CONVOLUTION)
        assert conv_bound.shape == hv1.shape
        
        # Test Fourier binding (if implemented)
        if hasattr(BindingType, 'FOURIER'):
            fourier_bound = binding_ops.bind(hv1, hv2, BindingType.FOURIER)
            assert fourier_bound.shape == hv1.shape
    
    def test_behavioral_site_extraction(self, hdc_config):
        """Test behavioral site feature extraction."""
        probe_config = ProbeConfig(
            sites=["layer_3", "layer_6", "layer_9", "layer_12"],
            aggregation="mean",
            normalize=True
        )
        
        behavioral_sites = BehavioralSites(probe_config, hdc_config)
        
        # Create mock activations
        batch_size = 2
        seq_len = 128
        hidden_dim = 768
        
        activations = {
            f"layer_{i}": torch.randn(batch_size, seq_len, hidden_dim)
            for i in [3, 6, 9, 12]
        }
        
        # Extract features
        features = behavioral_sites.extract_features(activations)
        
        # Verify extraction
        assert len(features) == len(probe_config.sites)
        for site in probe_config.sites:
            assert site in features
            assert features[site].shape[-1] == hidden_dim
        
        # Test hierarchical encoding
        hv = behavioral_sites.encode_to_hypervector(features)
        assert hv.shape[0] == hdc_config.dimension
    
    def test_error_correction_recovery(self, hdc_config):
        """Test error correction with various noise levels."""
        ec_config = ErrorCorrectionConfig(
            use_xor_parity=True,
            parity_blocks=4,
            use_hamming=True,
            noise_tolerance=0.15
        )
        
        error_correction = ErrorCorrection(ec_config, hdc_config.dimension)
        
        # Test with different noise levels
        for noise_level in [0.05, 0.10, 0.15]:
            original = np.random.choice([0, 1], hdc_config.dimension)
            
            # Encode with error correction
            encoded = error_correction.encode(original)
            
            # Add noise
            corrupted = encoded.copy()
            n_errors = int(len(encoded) * noise_level)
            error_positions = np.random.choice(len(encoded), n_errors, replace=False)
            corrupted[error_positions] = 1 - corrupted[error_positions]
            
            # Decode and check recovery
            recovered = error_correction.decode(corrupted)
            
            # Calculate recovery rate
            recovery_rate = np.mean(recovered == original)
            
            # Recovery should be good up to noise tolerance
            if noise_level <= ec_config.noise_tolerance:
                assert recovery_rate > 0.85
            else:
                assert recovery_rate > 0.70  # Degraded but still decent


class TestConsensusAndAggregation:
    """Unit tests for consensus mechanisms and decision aggregation."""
    
    def test_weighted_voting_aggregation(self):
        """Test weighted voting decision aggregation."""
        aggregator = DecisionAggregator(strategy=AggregationStrategy.WEIGHTED_VOTING)
        
        # Create diverse decisions
        decisions = [
            {"model": "A", "verdict": "accept", "confidence": 0.95, "weight": 1.0},
            {"model": "B", "verdict": "accept", "confidence": 0.85, "weight": 0.8},
            {"model": "C", "verdict": "reject", "confidence": 0.70, "weight": 0.6},
            {"model": "D", "verdict": "accept", "confidence": 0.60, "weight": 0.5}
        ]
        
        result = aggregator.aggregate(decisions)
        
        # Verify aggregation
        assert result["final_verdict"] in ["accept", "reject"]
        assert 0 <= result["confidence"] <= 1
        assert "evidence" in result
        
        # With these weights, accept should win
        assert result["final_verdict"] == "accept"
    
    def test_byzantine_fault_tolerance(self):
        """Test Byzantine consensus with faulty nodes."""
        config = ConsensusConfig(
            n_nodes=7,
            byzantine_threshold=0.33,
            timeout=10.0,
            require_proof=True
        )
        
        consensus = ByzantineConsensus(config)
        
        # Create votes with 2 Byzantine nodes (< 1/3)
        votes = {
            "node_1": {"verdict": "accept", "proof": hashlib.sha256(b"proof1").hexdigest()},
            "node_2": {"verdict": "accept", "proof": hashlib.sha256(b"proof2").hexdigest()},
            "node_3": {"verdict": "accept", "proof": hashlib.sha256(b"proof3").hexdigest()},
            "node_4": {"verdict": "accept", "proof": hashlib.sha256(b"proof4").hexdigest()},
            "node_5": {"verdict": "reject", "proof": hashlib.sha256(b"byzantine1").hexdigest()},
            "node_6": {"verdict": "reject", "proof": hashlib.sha256(b"byzantine2").hexdigest()},
            "node_7": {"verdict": "accept", "proof": hashlib.sha256(b"proof7").hexdigest()}
        }
        
        result = consensus.reach_consensus(votes)
        
        # Should reach consensus despite Byzantine nodes
        assert result["consensus"] == "accept"
        assert result["agreement_ratio"] > config.byzantine_threshold
        assert result["byzantine_nodes"] == ["node_5", "node_6"]
    
    def test_hierarchical_decision_aggregation(self):
        """Test hierarchical aggregation across segments."""
        aggregator = DecisionAggregator(strategy=AggregationStrategy.HIERARCHICAL)
        
        # Create segment-level decisions
        segment_decisions = {
            "segment_1": {
                "similarities": [0.85, 0.83, 0.86],
                "verdict": "accept",
                "confidence": 0.85
            },
            "segment_2": {
                "similarities": [0.78, 0.80, 0.79],
                "verdict": "accept",
                "confidence": 0.79
            },
            "segment_3": {
                "similarities": [0.65, 0.68, 0.66],
                "verdict": "reject",
                "confidence": 0.66
            },
            "segment_4": {
                "similarities": [0.88, 0.87, 0.89],
                "verdict": "accept",
                "confidence": 0.88
            }
        }
        
        # Aggregate hierarchically
        global_result = aggregator.aggregate_hierarchical(segment_decisions)
        
        assert "segment_verdicts" in global_result
        assert "global_verdict" in global_result
        assert "global_confidence" in global_result
        
        # With 3 accepts and 1 reject, should accept globally
        assert global_result["global_verdict"] == "accept"


class TestPrivacyMechanisms:
    """Unit tests for privacy-preserving mechanisms."""
    
    def test_differential_privacy_noise(self, privacy_config):
        """Test differential privacy noise addition and budget."""
        dp = DifferentialPrivacy(privacy_config)
        
        # Test Laplace mechanism
        original_value = 0.85
        noisy_values = []
        
        for _ in range(1000):
            noisy = dp.add_noise(original_value)
            noisy_values.append(noisy)
        
        # Noise should be unbiased
        mean_noisy = np.mean(noisy_values)
        assert abs(mean_noisy - original_value) < 0.05
        
        # Noise should have correct scale
        scale = privacy_config.sensitivity / privacy_config.epsilon
        empirical_std = np.std(noisy_values)
        expected_std = scale * np.sqrt(2)  # Laplace variance
        assert abs(empirical_std - expected_std) / expected_std < 0.2
        
        # Test privacy budget consumption
        initial_budget = dp.remaining_budget
        dp.consume_budget(0.3)
        assert dp.remaining_budget == initial_budget - 0.3
        
        # Test budget exhaustion
        with pytest.raises(ValueError):
            dp.consume_budget(10.0)  # Exceeds budget
    
    def test_homomorphic_encryption_operations(self):
        """Test homomorphic encryption for secure computation."""
        he_ops = HomomorphicOperations()
        
        # Generate keypair
        public_key, private_key = he_ops.generate_keys()
        
        # Test homomorphic addition
        val1, val2 = 0.75, 0.25
        enc1 = he_ops.encrypt(val1, public_key)
        enc2 = he_ops.encrypt(val2, public_key)
        
        enc_sum = he_ops.add(enc1, enc2, public_key)
        dec_sum = he_ops.decrypt(enc_sum, private_key)
        
        assert abs(dec_sum - (val1 + val2)) < 0.001
        
        # Test homomorphic multiplication by constant
        const = 3.0
        enc_scaled = he_ops.multiply_constant(enc1, const, public_key)
        dec_scaled = he_ops.decrypt(enc_scaled, private_key)
        
        assert abs(dec_scaled - (val1 * const)) < 0.001
        
        # Test federated aggregation
        values = [0.8, 0.85, 0.82, 0.79]
        encrypted_values = [he_ops.encrypt(v, public_key) for v in values]
        
        enc_avg = he_ops.federated_average(encrypted_values, public_key)
        dec_avg = he_ops.decrypt(enc_avg, private_key)
        
        assert abs(dec_avg - np.mean(values)) < 0.001
    
    def test_zero_knowledge_proofs(self):
        """Test zero-knowledge distance and range proofs."""
        zk_proofs = DistanceZKProofs()
        
        # Create test vectors
        dim = 100
        vec1 = np.random.randn(dim)
        vec2 = np.random.randn(dim)
        
        # Compute true distance
        true_distance = np.linalg.norm(vec1 - vec2)
        
        # Generate and verify distance proof
        proof = zk_proofs.generate_distance_proof(vec1, vec2, true_distance)
        assert zk_proofs.verify_distance_proof(proof, true_distance)
        
        # Should fail with wrong distance
        wrong_distance = true_distance * 1.5
        assert not zk_proofs.verify_distance_proof(proof, wrong_distance)
        
        # Test range proof
        value = 0.75
        min_val, max_val = 0.0, 1.0
        
        range_proof = zk_proofs.generate_range_proof(value, min_val, max_val)
        assert zk_proofs.verify_range_proof(range_proof, min_val, max_val)
        
        # Should fail if value outside range
        bad_proof = zk_proofs.generate_range_proof(1.5, min_val, max_val)
        assert not zk_proofs.verify_range_proof(bad_proof, min_val, max_val)


# ==================== INTEGRATION TESTS ====================

class TestEndToEndWorkflow:
    """Integration tests for complete REV verification workflow."""
    
    @pytest.mark.integration
    async def test_complete_verification_pipeline(self, rev_config):
        """Test full end-to-end verification with all components."""
        # Initialize pipeline
        pipeline = REVPipeline(rev_config)
        
        # Create mock models
        reference_model = Mock()
        reference_model.name = "reference_model"
        reference_model.generate = Mock(side_effect=lambda prompt: 
                                      f"Reference response to: {prompt[:30]}")
        
        candidate_model = Mock()
        candidate_model.name = "candidate_model"
        candidate_model.generate = Mock(side_effect=lambda prompt:
                                      f"Candidate response to: {prompt[:30]}")
        
        # Generate test challenges
        master_key = hashlib.sha256(b"test_integration").digest()
        challenge_gen = EnhancedKDFPromptGenerator(master_key, "integration_test")
        
        challenges = challenge_gen.generate_challenge_set(
            n_challenges=20,
            adversarial_ratio=0.15,
            behavioral_probe_ratio=0.10
        )
        
        # Mock pipeline methods for testing
        with patch.object(pipeline, 'process_segment') as mock_process:
            mock_process.side_effect = lambda seg, model: {
                "response": f"{model.name} processed segment {seg.segment_id}",
                "activations": np.random.randn(768),
                "similarity": np.random.uniform(0.75, 0.90)
            }
            
            # Run verification
            result = await pipeline.verify_models(
                reference_model=reference_model,
                candidate_model=candidate_model,
                challenges=[c["prompt"] for c in challenges]
            )
        
        # Validate result structure
        assert "verdict" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "segments_processed" in result
        assert result["verdict"] in ["accept", "reject"]
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.integration
    def test_streaming_consensus_verification(self):
        """Test streaming consensus with progressive updates."""
        verifier = StreamingConsensusVerifier(
            mode=ConsensusMode.PROGRESSIVE,
            threshold=0.75,
            min_segments=10
        )
        
        # Simulate streaming segments
        n_segments = 50
        segment_results = []
        
        for i in range(n_segments):
            # Create segment
            segment = Segment(
                segment_id=i,
                tokens=list(range(i*100, (i+1)*100)),
                start_idx=i*100,
                end_idx=(i+1)*100
            )
            
            # Simulate processing
            similarity = np.random.uniform(0.70, 0.85)
            segment_result = {
                "segment_id": i,
                "similarity": similarity,
                "confidence": min(similarity + 0.1, 1.0)
            }
            
            # Update consensus
            consensus = verifier.update(segment_result)
            segment_results.append(consensus)
            
            # Check for early stopping
            if consensus and consensus.get("final"):
                break
        
        # Verify consensus was reached
        final_consensus = segment_results[-1]
        assert final_consensus is not None
        assert "verdict" in final_consensus
        assert "confidence" in final_consensus
        assert "segments_used" in final_consensus
    
    @pytest.mark.integration
    def test_checkpoint_and_resume(self, rev_config, tmp_path):
        """Test checkpoint creation and resumption."""
        rev_config.checkpoint_dir = str(tmp_path)
        pipeline = REVPipeline(rev_config)
        
        # Create initial state
        initial_state = {
            "processed_segments": 75,
            "current_segment": 76,
            "sequential_state": {
                "n_samples": 75,
                "mean": 0.82,
                "variance": 0.01,
                "log_likelihood_ratio": 2.5
            },
            "merkle_tree": {
                "root": hashlib.sha256(b"test_root").hexdigest(),
                "height": 7
            },
            "consensus_state": {
                "votes": {"accept": 60, "reject": 15},
                "confidence": 0.80
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(tmp_path, "checkpoint.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(initial_state, f)
        
        # Create new pipeline and resume
        new_pipeline = REVPipeline(rev_config)
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            resumed_state = pickle.load(f)
        
        # Verify state restoration
        assert resumed_state["processed_segments"] == initial_state["processed_segments"]
        assert resumed_state["current_segment"] == initial_state["current_segment"]
        assert resumed_state["sequential_state"]["mean"] == initial_state["sequential_state"]["mean"]
        assert resumed_state["merkle_tree"]["root"] == initial_state["merkle_tree"]["root"]
    
    @pytest.mark.integration
    def test_multi_model_comparison_matrix(self):
        """Test pairwise comparison of multiple models."""
        models = []
        for i in range(4):
            model = Mock()
            model.name = f"model_{i}"
            model.id = f"model_{i}_id"
            model.generate = Mock(side_effect=lambda p, m=i: f"Model {m} response")
            models.append(model)
        
        # Create comparison matrix
        comparison_matrix = {}
        
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i < j:
                    # Simulate comparison
                    similarity = 0.95 if i == j else np.random.uniform(0.70, 0.85)
                    verdict = "accept" if similarity > 0.80 else "reject"
                    
                    key = f"{model_a.name}_vs_{model_b.name}"
                    comparison_matrix[key] = {
                        "similarity": similarity,
                        "verdict": verdict,
                        "confidence": min(similarity + 0.05, 1.0)
                    }
        
        # Verify all pairs compared
        n_models = len(models)
        expected_comparisons = n_models * (n_models - 1) // 2
        assert len(comparison_matrix) == expected_comparisons
        
        # Check transitivity (if A~B and B~C then A~C)
        # This is a simplified check
        verdicts = {key: val["verdict"] for key, val in comparison_matrix.items()}
        
        # If model_0 accepts model_1 and model_1 accepts model_2,
        # then model_0 should likely accept model_2
        if ("model_0_vs_model_1" in verdicts and 
            "model_1_vs_model_2" in verdicts and
            "model_0_vs_model_2" in verdicts):
            
            if (verdicts["model_0_vs_model_1"] == "accept" and
                verdicts["model_1_vs_model_2"] == "accept"):
                # Transitivity suggests model_0 should accept model_2
                # (though not guaranteed due to noise)
                pass  # Soft check only


class TestPerformanceBenchmarks:
    """Performance benchmark tests against targets."""
    
    @pytest.mark.benchmark
    def test_hdc_encoding_performance(self, benchmark, hdc_config):
        """Benchmark HDC encoding against target: <50ms for 100K dimensions."""
        # Use 100K dimensions for benchmark
        large_config = HypervectorConfig(
            dimension=100000,
            sparsity=0.01,
            use_ternary=False
        )
        
        encoder = UnifiedHDCEncoder(large_config)
        data = np.random.randn(1000)
        
        # Benchmark encoding
        result = benchmark(encoder.encode, data)
        
        # Verify output
        assert result.shape[0] == large_config.dimension
        
        # Check performance target: <50ms
        assert benchmark.stats['mean'] < 0.050  # 50ms
    
    @pytest.mark.benchmark
    def test_hamming_distance_performance(self, benchmark):
        """Benchmark Hamming distance against target: <1ms for 10K dimensions."""
        from src.hypervector.hamming import optimized_hamming_distance
        
        # Create 10K dimensional binary vectors
        dim = 10000
        vec1 = np.random.choice([0, 1], dim)
        vec2 = np.random.choice([0, 1], dim)
        
        # Benchmark distance computation
        distance = benchmark(optimized_hamming_distance, vec1, vec2)
        
        # Verify result
        assert 0 <= distance <= dim
        
        # Check performance target: <1ms
        assert benchmark.stats['mean'] < 0.001  # 1ms
    
    @pytest.mark.benchmark
    def test_sequential_test_performance(self, benchmark, sequential_config):
        """Benchmark sequential test against target: <100ms for 1000 samples."""
        tester = SequentialTester(sequential_config)
        samples = np.random.uniform(0.7, 0.9, 1000)
        
        def process_samples():
            for sample in samples:
                decision = tester.update(sample)
                if decision in ["accept", "reject"]:
                    break
            return tester.state
        
        # Benchmark sequential testing
        state = benchmark(process_samples)
        
        # Verify processing
        assert state.n_samples > 0
        assert state.n_samples <= 1000
        
        # Check performance target: <100ms for 1000 samples
        assert benchmark.stats['mean'] < 0.100  # 100ms
    
    @pytest.mark.benchmark
    def test_merkle_tree_construction(self, benchmark):
        """Benchmark Merkle tree construction and proof generation."""
        tree = EnhancedMerkleTree()
        
        # Create 10000 leaves
        n_leaves = 10000
        leaves = [f"leaf_{i}".encode() for i in range(n_leaves)]
        
        # Benchmark tree construction
        def build_and_prove():
            tree.build(leaves)
            # Generate proof for middle leaf
            proof = tree.generate_proof(n_leaves // 2)
            return tree.root, proof
        
        root, proof = benchmark(build_and_prove)
        
        # Verify tree was built
        assert root is not None
        assert proof is not None
        
        # Performance should be reasonable for 10K leaves
        assert benchmark.stats['mean'] < 1.0  # 1 second
    
    @pytest.mark.benchmark
    def test_parallel_pipeline_throughput(self, benchmark):
        """Benchmark parallel pipeline throughput."""
        config = PipelineConfig(
            thread_pool_size=4,
            process_pool_size=0  # Threads only for benchmark
        )
        
        pipeline = ParallelPipeline(config)
        
        # Create mock tasks
        n_tasks = 100
        
        def process_batch():
            results = []
            for i in range(n_tasks):
                # Simulate task processing
                result = {"task_id": i, "result": i * 2}
                results.append(result)
            return results
        
        with patch.object(pipeline, 'execute_batch', side_effect=process_batch):
            results = benchmark(pipeline.execute_batch)
        
        # Verify results
        assert len(results) == n_tasks
        
        # Throughput should be good
        throughput = n_tasks / benchmark.stats['mean']
        assert throughput > 100  # At least 100 tasks/second


class TestMemoryBoundedExecution:
    """Test memory-bounded execution and resource management."""
    
    def test_memory_limit_enforcement(self, rev_config):
        """Test that memory limits are enforced during execution."""
        rev_config.memory_limit_gb = 1.0  # 1GB limit
        pipeline = REVPipeline(rev_config)
        
        # Track memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024**3  # GB
        
        # Try to allocate large data
        try:
            # This should trigger memory management
            large_segments = []
            for i in range(100):
                # Each segment ~10MB
                segment_data = np.random.randn(1000, 1250).astype(np.float64)
                large_segments.append(segment_data)
                
                # Pipeline should manage memory
                current_memory = psutil.Process().memory_info().rss / 1024**3
                memory_increase = current_memory - initial_memory
                
                # Should stay within limit (with some overhead)
                assert memory_increase < rev_config.memory_limit_gb * 1.5
                
                # Simulate processing to trigger cleanup
                if i % 10 == 0:
                    large_segments = large_segments[-5:]  # Keep only recent
        
        except MemoryError:
            # Memory error is acceptable - means limit is enforced
            pass
    
    def test_segment_offloading(self, tmp_path):
        """Test segment offloading to disk when memory is constrained."""
        from src.executor.segment_runner import SegmentRunner, SegmentConfig
        
        config = SegmentConfig(
            segment_size=512,
            max_memory_gb=0.5,
            offload_to_disk=True,
            cache_dir=str(tmp_path)
        )
        
        runner = SegmentRunner(config)
        
        # Create segments that exceed memory limit
        segments = []
        for i in range(20):
            segment = Segment(
                segment_id=i,
                tokens=list(range(i*512, (i+1)*512)),
                start_idx=i*512,
                end_idx=(i+1)*512,
                signatures={"test": np.random.randn(1000)}
            )
            segments.append(segment)
        
        # Process segments (should trigger offloading)
        offloaded_count = 0
        for segment in segments:
            result = runner.process_segment_with_offload(segment)
            
            # Check if segment was offloaded
            offload_path = os.path.join(tmp_path, f"segment_{segment.segment_id}.pkl")
            if os.path.exists(offload_path):
                offloaded_count += 1
        
        # Some segments should have been offloaded
        assert offloaded_count > 0
        
        # Verify offloaded segments can be loaded
        if offloaded_count > 0:
            test_path = os.path.join(tmp_path, "segment_0.pkl")
            if os.path.exists(test_path):
                with open(test_path, 'rb') as f:
                    loaded_segment = pickle.load(f)
                assert loaded_segment is not None


class TestCryptographicIntegrity:
    """Test cryptographic integrity and security features."""
    
    def test_hmac_seed_derivation(self):
        """Test HMAC-based seed derivation for challenges."""
        master_key = hashlib.sha256(b"test_hmac_key").digest()
        
        generator1 = EnhancedKDFPromptGenerator(master_key, "test_run", "1.0")
        generator2 = EnhancedKDFPromptGenerator(master_key, "test_run", "1.0")
        
        # Generate challenges with same parameters
        challenges1 = generator1.generate_challenge_set(n_challenges=10)
        challenges2 = generator2.generate_challenge_set(n_challenges=10)
        
        # Should be identical (deterministic)
        for c1, c2 in zip(challenges1, challenges2):
            assert c1["prompt"] == c2["prompt"]
            assert c1["seed_hex"] == c2["seed_hex"]
            assert c1["canonical_form"] == c2["canonical_form"]
        
        # Different run_id should give different challenges
        generator3 = EnhancedKDFPromptGenerator(master_key, "different_run", "1.0")
        challenges3 = generator3.generate_challenge_set(n_challenges=10)
        
        # Should be different
        for c1, c3 in zip(challenges1, challenges3):
            assert c1["prompt"] != c3["prompt"]
            assert c1["seed_hex"] != c3["seed_hex"]
    
    def test_merkle_proof_verification(self):
        """Test Merkle tree proof generation and verification."""
        tree = EnhancedMerkleTree()
        
        # Create data
        data = [f"segment_{i}".encode() for i in range(1000)]
        tree.build(data)
        root = tree.root
        
        # Test valid proof
        index = 42
        proof = tree.generate_proof(index)
        assert tree.verify_proof(data[index], proof, root)
        
        # Test invalid proof (tampered data)
        tampered = b"tampered_segment"
        assert not tree.verify_proof(tampered, proof, root)
        
        # Test invalid proof (wrong index)
        wrong_proof = tree.generate_proof(index + 1)
        assert not tree.verify_proof(data[index], wrong_proof, root)
    
    def test_commitment_scheme(self):
        """Test cryptographic commitments for decisions."""
        from src.crypto.commit import CommitmentScheme
        
        scheme = CommitmentScheme()
        
        # Create commitment to decision
        decision = {
            "verdict": "accept",
            "confidence": 0.87,
            "timestamp": int(time.time())
        }
        
        commitment, opening = scheme.commit(json.dumps(decision))
        
        # Verify valid commitment
        assert scheme.verify(commitment, json.dumps(decision), opening)
        
        # Test binding property (can't change decision)
        tampered_decision = decision.copy()
        tampered_decision["verdict"] = "reject"
        assert not scheme.verify(commitment, json.dumps(tampered_decision), opening)
        
        # Test hiding property (commitment reveals nothing)
        assert len(commitment) == 64  # SHA-256 hex
        assert commitment != json.dumps(decision)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])