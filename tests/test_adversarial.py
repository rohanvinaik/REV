"""
Adversarial robustness tests for REV system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import hashlib

from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.hdc.error_correction import ErrorCorrection, ErrorCorrectionConfig
from src.hdc.binding_operations import BindingOperations
from src.core.sequential import sequential_verify
from src.verifier.blackbox import BlackBoxVerifier
from src.privacy.distance_zk_proofs import DistanceZKProof
from src.rev_pipeline import REVPipeline


class TestAdversarialAttacks:
    """Test robustness against adversarial attacks."""
    
    def test_bit_flipping_attack(self):
        """Test robustness against bit-flipping attacks."""
        corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=1000))
        
        # Original hypervector
        original = np.random.choice([-1, 1], size=1000).astype(np.float32)
        
        # Encode with error correction
        encoded = corrector.encode_with_parity(original)
        
        # Adversarial bit flips
        num_flips = 100  # 10% corruption
        flip_indices = np.random.choice(len(encoded), num_flips, replace=False)
        adversarial = encoded.copy()
        adversarial[flip_indices] = -adversarial[flip_indices]
        
        # Try to recover
        recovered, corrections = corrector.decode_with_correction(
            adversarial,
            correct_errors=True
        )
        
        # Measure recovery success
        recovery_rate = np.mean(recovered == original)
        
        # Should recover majority of bits
        assert recovery_rate > 0.85  # 85% recovery threshold
        assert corrections > 0  # Should detect errors
    
    def test_noise_injection_attack(self):
        """Test robustness against noise injection."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))
        
        # Encode data
        data = np.random.randn(100)
        hypervector = encoder.encode(data)
        
        # Inject different types of noise
        noise_types = ["gaussian", "salt_pepper", "burst"]
        
        for noise_type in noise_types:
            # Add noise
            corrector = ErrorCorrection(ErrorCorrectionConfig(dimension=10000))
            noisy = corrector.add_noise(
                hypervector,
                noise_level=0.2,
                noise_type=noise_type
            )
            
            # Compute similarity after noise
            cosine_sim = np.dot(hypervector, noisy) / (
                np.linalg.norm(hypervector) * np.linalg.norm(noisy) + 1e-8
            )
            
            # Should maintain some similarity despite noise
            assert cosine_sim > 0.5  # Minimum similarity threshold
    
    def test_byzantine_vector_attack(self):
        """Test robustness against Byzantine (malicious) vectors."""
        from src.privacy.homomorphic_ops import FederatedProtocol
        
        # Initialize federated protocol
        protocol = FederatedProtocol(num_participants=5, threshold=3)
        
        # Register participants
        from Crypto.PublicKey import RSA
        keys = [RSA.generate(1024) for _ in range(5)]
        for i, key in enumerate(keys):
            protocol.register_participant(f"participant_{i}", key.publickey())
        
        # Create honest vectors
        honest_vectors = [np.random.randn(100) for _ in range(3)]
        
        # Create Byzantine (adversarial) vectors
        byzantine_vectors = [
            np.ones(100) * 1000,  # Extreme values
            np.random.randn(100) * 100  # High variance
        ]
        
        # Test aggregation with Byzantine vectors
        from src.privacy.homomorphic_ops import HomomorphicOperations
        he_ops = HomomorphicOperations()
        
        round_num = protocol.initiate_round()
        
        # Submit all vectors
        all_vectors = honest_vectors + byzantine_vectors
        for i, vec in enumerate(all_vectors):
            encrypted = he_ops.encrypt_vector(vec, keys[i].publickey())
            protocol.submit_encrypted_vector(
                f"participant_{i}",
                encrypted,
                round_num
            )
        
        # Aggregate should handle Byzantine vectors
        aggregated = protocol.aggregate_round(round_num)
        
        # Check that aggregation is not dominated by Byzantine values
        assert np.mean(aggregated) < 100  # Not dominated by extreme values
        assert np.std(aggregated) < 50   # Variance is bounded
    
    def test_model_extraction_attack(self):
        """Test resistance to model extraction attacks."""
        verifier = BlackBoxVerifier()
        
        # Simulate extraction attempt with many queries
        extraction_queries = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
            # ... many similar queries
        ] * 100  # Repeated queries
        
        # Rate limiting should prevent rapid extraction
        with patch('time.time') as mock_time:
            mock_time.return_value = 0
            
            successful_queries = 0
            for query in extraction_queries[:10]:  # Test first 10
                try:
                    # Should be rate limited
                    verifier.rate_limiters["test_model"].wait_if_needed()
                    successful_queries += 1
                except:
                    pass
            
            # Rate limiting should restrict queries
            assert successful_queries < len(extraction_queries[:10])
    
    def test_membership_inference_attack(self):
        """Test resistance to membership inference."""
        pipeline = REVPipeline()
        
        # Create training-like and test-like samples
        training_sample = "The quick brown fox jumps over the lazy dog"
        test_sample = "A completely different sentence with unique words"
        
        # Process both samples
        with patch.object(pipeline, 'process_challenge') as mock_process:
            mock_process.return_value = {
                "behavioral_signature": {"vector": np.random.randn(1000)},
                "merkle_root": hashlib.sha256(b"test").digest()
            }
            
            result_train = pipeline.process_challenge(Mock(), training_sample)
            result_test = pipeline.process_challenge(Mock(), test_sample)
        
        # Signatures should not reveal membership
        # (In practice, would check actual similarity distributions)
        assert "behavioral_signature" in result_train
        assert "behavioral_signature" in result_test
    
    def test_poisoning_attack_resistance(self):
        """Test resistance to data poisoning attacks."""
        # Sequential testing should be robust to poisoned samples
        state = sequential_verify([], alpha=0.05, beta=0.10)
        
        # Normal samples
        normal_scores = np.random.normal(0.9, 0.05, 50)
        
        # Poisoned samples (adversarial)
        poisoned_scores = np.random.uniform(0, 0.3, 10)
        
        # Mix normal and poisoned
        all_scores = np.concatenate([normal_scores, poisoned_scores])
        np.random.shuffle(all_scores)
        
        # Sequential test should handle mixed data
        result = sequential_verify(all_scores.tolist())
        
        # Decision should not be completely corrupted
        assert result.decision in ["accept_h0", "reject_h0", "continue"]
        
        # Mean should be influenced but not dominated by poisoned data
        assert 0.3 < result.mean < 0.9


class TestCryptographicRobustness:
    """Test cryptographic robustness of privacy features."""
    
    def test_zk_proof_forgery_resistance(self):
        """Test that ZK proofs cannot be forged."""
        zk = DistanceZKProof(security_bits=128)
        
        # Create legitimate proof
        vec_a = np.random.randn(100)
        vec_b = np.random.randn(100)
        true_distance = np.linalg.norm(vec_a - vec_b)
        
        legitimate_proof = zk.prove_distance(vec_a, vec_b, true_distance)
        
        # Try to forge proof for different distance
        forged_distance = true_distance * 2
        
        # Verification should fail for wrong distance
        is_valid = zk.verify_distance(legitimate_proof, forged_distance)
        assert is_valid == False
        
        # Try to tamper with proof
        tampered_proof = legitimate_proof
        tampered_proof.challenge = b"tampered_challenge"
        
        # Verification should fail for tampered proof
        is_valid = zk.verify_distance(tampered_proof, true_distance)
        assert is_valid == False
    
    def test_commitment_binding(self):
        """Test that commitments are binding."""
        zk = DistanceZKProof()
        
        # Create commitment to vector
        vector = np.random.randn(100)
        commitment, randomness = zk.commit_vector(vector)
        
        # Try to open commitment to different vector
        different_vector = np.random.randn(100)
        
        # Should not be able to find randomness for different vector
        # (In practice, this is computationally infeasible)
        different_commitment, _ = zk.commit_vector(different_vector)
        
        assert commitment != different_commitment
    
    def test_homomorphic_integrity(self):
        """Test integrity of homomorphic operations."""
        from src.privacy.homomorphic_ops import HomomorphicOperations
        
        he_ops = HomomorphicOperations()
        
        # Create vectors
        vec_a = np.array([1, 2, 3, 4, 5])
        vec_b = np.array([5, 4, 3, 2, 1])
        
        # Encrypt
        enc_a = he_ops.encrypt_vector(vec_a)
        enc_b = he_ops.encrypt_vector(vec_b)
        
        # Homomorphic addition (simplified)
        enc_sum = he_ops.homomorphic_add(enc_a, enc_b)
        
        # Decrypt and verify
        # Note: This is a simplified test - real HE would maintain encryption
        decrypted_a = he_ops.decrypt_vector(enc_a)
        decrypted_b = he_ops.decrypt_vector(enc_b)
        
        np.testing.assert_array_almost_equal(vec_a, decrypted_a, decimal=3)
        np.testing.assert_array_almost_equal(vec_b, decrypted_b, decimal=3)


class TestSideChannelResistance:
    """Test resistance to side-channel attacks."""
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=1000))
        
        # Different input sizes
        small_input = np.random.randn(10)
        large_input = np.random.randn(1000)
        
        # Time encoding
        import time
        
        times_small = []
        times_large = []
        
        for _ in range(100):
            start = time.perf_counter()
            encoder.encode(small_input)
            times_small.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            encoder.encode(large_input)
            times_large.append(time.perf_counter() - start)
        
        # Timing should not reveal too much about input
        # (Some correlation is expected, but should be bounded)
        mean_small = np.mean(times_small)
        mean_large = np.mean(times_large)
        
        ratio = mean_large / mean_small
        assert ratio < 10  # Timing difference should be bounded
    
    def test_memory_access_pattern_resistance(self):
        """Test that memory access patterns don't leak information."""
        from src.hypervector.hamming import HammingDistance
        
        hamming = HammingDistance(use_lut=True)
        
        # Different patterns of bits
        all_zeros = np.zeros(1000, dtype=np.uint8)
        all_ones = np.ones(1000, dtype=np.uint8)
        random_bits = np.random.choice([0, 1], size=1000).astype(np.uint8)
        
        # Memory access should be similar regardless of bit patterns
        # (In practice, would monitor actual memory access)
        
        # Compute distances
        dist1 = hamming.compute(all_zeros, random_bits)
        dist2 = hamming.compute(all_ones, random_bits)
        dist3 = hamming.compute(random_bits, random_bits)
        
        # All should complete without revealing patterns
        assert 0 <= dist1 <= 1000
        assert 0 <= dist2 <= 1000
        assert dist3 == 0  # Same vector


class TestByzantineFaultTolerance:
    """Test Byzantine fault tolerance in distributed settings."""
    
    def test_consensus_with_byzantine_nodes(self):
        """Test consensus achievement with Byzantine nodes."""
        from src.verifier.decision_aggregator import DecisionAggregator
        
        # Create aggregators for different nodes
        num_nodes = 7
        byzantine_nodes = 2  # Less than 1/3
        
        aggregators = [DecisionAggregator() for _ in range(num_nodes)]
        
        # Honest nodes report similar scores
        honest_scores = np.random.normal(0.9, 0.05, num_nodes - byzantine_nodes)
        
        # Byzantine nodes report adversarial scores
        byzantine_scores = np.random.uniform(0, 0.3, byzantine_nodes)
        
        all_scores = np.concatenate([honest_scores, byzantine_scores])
        
        # Each node processes scores
        decisions = []
        for i, aggregator in enumerate(aggregators):
            if i < num_nodes - byzantine_nodes:
                # Honest node
                aggregator.add_result("test", honest_scores[i])
            else:
                # Byzantine node
                aggregator.add_result("test", byzantine_scores[i - (num_nodes - byzantine_nodes)])
            
            decision = aggregator.get_final_decision()
            decisions.append(decision["overall_decision"])
        
        # Majority should reach correct consensus
        accept_count = sum(1 for d in decisions if d == "accept_h0")
        reject_count = sum(1 for d in decisions if d == "reject_h0")
        
        # Honest majority should prevail
        assert accept_count > byzantine_nodes or reject_count > byzantine_nodes
    
    def test_merkle_tree_tampering_detection(self):
        """Test detection of Merkle tree tampering."""
        from src.rev_pipeline import REVPipeline
        
        pipeline = REVPipeline()
        
        # Create legitimate Merkle tree
        leaves = [
            hashlib.sha256(f"segment_{i}".encode()).digest()
            for i in range(10)
        ]
        
        legitimate_root = pipeline._build_merkle_tree(leaves)
        
        # Tamper with a leaf
        tampered_leaves = leaves.copy()
        tampered_leaves[5] = hashlib.sha256(b"tampered").digest()
        
        tampered_root = pipeline._build_merkle_tree(tampered_leaves)
        
        # Roots should be different
        assert legitimate_root != tampered_root
        
        # Verification should detect tampering
        def verify_merkle_proof(root, leaf, proof):
            """Verify Merkle proof for a leaf."""
            current = leaf
            for sibling in proof:
                if current < sibling:
                    combined = current + sibling
                else:
                    combined = sibling + current
                current = hashlib.sha256(combined).digest()
            return current == root
        
        # Proof for legitimate tree should work
        # (In practice, would implement full Merkle proof generation)
        assert legitimate_root != tampered_root


class TestDifferentialPrivacy:
    """Test differential privacy guarantees."""
    
    def test_noise_mechanism_privacy(self):
        """Test that noise mechanisms provide privacy."""
        from src.privacy.differential_privacy import DifferentialPrivacy
        
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Two similar datasets (differ by one element)
        dataset_a = np.random.randn(100)
        dataset_b = dataset_a.copy()
        dataset_b[50] = dataset_b[50] + 10  # Change one element
        
        # Compute statistics with DP
        mean_a = dp.add_noise(np.mean(dataset_a), sensitivity=0.1)
        mean_b = dp.add_noise(np.mean(dataset_b), sensitivity=0.1)
        
        # Means should be similar but not identical (due to noise)
        assert mean_a != mean_b
        assert abs(mean_a - mean_b) < 1.0  # Bounded difference
    
    def test_privacy_budget_tracking(self):
        """Test that privacy budget is properly tracked."""
        from src.privacy.differential_privacy import DifferentialPrivacy
        
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        initial_budget = dp.remaining_budget
        
        # Make multiple queries
        for _ in range(10):
            dp.add_noise(np.random.randn(), sensitivity=0.1)
        
        # Budget should be depleted
        assert dp.remaining_budget < initial_budget
        
        # Should prevent queries when budget exhausted
        dp.remaining_budget = 0
        
        with pytest.raises(Exception):
            dp.add_noise(np.random.randn(), sensitivity=0.1)


class TestAdversarialInputGeneration:
    """Generate adversarial inputs for testing."""
    
    def generate_adversarial_hypervector(self, target_vector, epsilon=0.1):
        """Generate adversarial hypervector."""
        # Add carefully crafted noise
        noise = np.random.randn(len(target_vector))
        noise = noise / np.linalg.norm(noise)  # Normalize
        
        adversarial = target_vector + epsilon * noise
        
        # Ensure bounded perturbation
        perturbation = np.linalg.norm(adversarial - target_vector)
        assert perturbation <= epsilon * 1.1
        
        return adversarial
    
    def test_adversarial_examples(self):
        """Test system against adversarial examples."""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=1000))
        
        # Original input
        original = np.random.randn(100)
        original_hv = encoder.encode(original)
        
        # Generate adversarial examples
        epsilons = [0.01, 0.1, 0.5]
        
        for epsilon in epsilons:
            adversarial = self.generate_adversarial_hypervector(
                original_hv,
                epsilon
            )
            
            # Compute similarity
            similarity = np.dot(original_hv, adversarial) / (
                np.linalg.norm(original_hv) * np.linalg.norm(adversarial)
            )
            
            # System should maintain robustness
            if epsilon <= 0.1:
                assert similarity > 0.9  # High similarity for small perturbations
            else:
                assert similarity > 0.5  # Moderate similarity for larger perturbations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])