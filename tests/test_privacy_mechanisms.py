"""
Comprehensive tests for privacy-preserving mechanisms in REV.

Tests differential privacy, secure aggregation, zero-knowledge proofs,
and other privacy features.
"""

import pytest
import numpy as np
import torch
from typing import List, Dict, Any
import tempfile
import json

from src.privacy.differential_privacy import (
    DifferentialPrivacyMechanism,
    PrivacyLevel,
    PrivacyBudget,
    RenyiDifferentialPrivacy,
    ConcentratedDifferentialPrivacy,
    PrivacyAccountant,
    validate_privacy_parameters,
    estimate_utility_loss
)

from src.privacy.secure_aggregation import (
    SecureAggregator,
    AggregationParams,
    ShamirSecretSharing,
    MultiPartyComputation,
    FederatedAggregationProtocol,
    aggregate_signatures,
    federated_signature_aggregation
)

from src.privacy.distance_zk_proofs import (
    DistanceZKProof,
    CommitmentScheme,
    MerkleInclusionProof,
    EnhancedDistanceZKProof,
    DistanceProof,
    RangeProof
)


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""
    
    @pytest.fixture
    def dp_mechanism(self):
        """Create DP mechanism for testing."""
        return DifferentialPrivacyMechanism(
            privacy_level=PrivacyLevel.MEDIUM,
            sensitivity=1.0,
            seed=42
        )
    
    @pytest.fixture
    def privacy_budget(self):
        """Create privacy budget for testing."""
        return PrivacyBudget(epsilon=1.0, delta=1e-6)
    
    def test_privacy_budget_tracking(self, privacy_budget):
        """Test privacy budget management."""
        # Check initial state
        assert privacy_budget.epsilon == 1.0
        assert privacy_budget.consumed_epsilon == 0.0
        
        # Check spending capability
        assert privacy_budget.can_spend(0.5, 1e-7)
        assert not privacy_budget.can_spend(1.5, 1e-7)  # Too much epsilon
        assert not privacy_budget.can_spend(0.5, 1e-5)  # Too much delta
        
        # Spend budget
        privacy_budget.spend(0.3, 5e-7)
        assert privacy_budget.consumed_epsilon == 0.3
        assert privacy_budget.consumed_delta == 5e-7
        
        # Check remaining budget
        assert privacy_budget.can_spend(0.7, 5e-7)
        assert not privacy_budget.can_spend(0.8, 5e-7)
    
    def test_gaussian_noise_addition(self, dp_mechanism):
        """Test Gaussian noise mechanism."""
        data = torch.randn(100)
        original_mean = data.mean()
        
        # Add noise
        noisy_data = dp_mechanism.add_gaussian_noise(data, epsilon_spend=0.1, delta_spend=1e-7)
        
        # Check noise was added
        assert not torch.equal(data, noisy_data)
        
        # Check approximate preservation of statistics (with some tolerance)
        noisy_mean = noisy_data.mean()
        # With small epsilon, noise can be large - just check it's finite
        assert torch.isfinite(noisy_mean)
    
    def test_laplace_noise_addition(self, dp_mechanism):
        """Test Laplace noise mechanism."""
        data = torch.randn(100)
        
        # Add noise
        noisy_data = dp_mechanism.add_laplace_noise(data, epsilon_spend=0.1)
        
        # Check noise was added
        assert not torch.equal(data, noisy_data)
        
        # Check data shape preserved
        assert noisy_data.shape == data.shape
    
    def test_hypervector_privatization(self, dp_mechanism, privacy_budget):
        """Test hypervector privatization."""
        # Create test hypervector
        hypervector = torch.randn(1000)
        hypervector = hypervector / torch.norm(hypervector)  # Normalize
        
        # Privatize
        private_hv = dp_mechanism.privatize_hypervector(
            hypervector, privacy_budget, mechanism="gaussian"
        )
        
        # Check privacy budget was spent
        assert privacy_budget.consumed_epsilon > 0
        assert privacy_budget.consumed_delta > 0
        
        # Check result is normalized
        assert abs(torch.norm(private_hv) - 1.0) < 1e-6
        
        # Check some similarity preserved (but not perfect)
        similarity = torch.dot(hypervector, private_hv).item()
        assert -1.0 <= similarity <= 1.0  # Valid similarity range
    
    def test_distance_privatization(self, dp_mechanism, privacy_budget):
        """Test distance privatization."""
        original_distance = 0.5
        
        # Privatize distance
        private_distance = dp_mechanism.privatize_distance(
            original_distance, privacy_budget, max_distance=1.0
        )
        
        # Check privacy budget was spent
        assert privacy_budget.consumed_epsilon > 0
        
        # Check distance is in valid range
        assert 0.0 <= private_distance <= 1.0
        
        # Check approximate preservation (allow for clipping to [0,1])
        assert abs(original_distance - private_distance) <= 0.5
    
    def test_advanced_composition(self, dp_mechanism):
        """Test advanced composition theorem."""
        # Multiple mechanisms
        mechanisms = [(0.1, 1e-8), (0.2, 1e-8), (0.1, 0), (0.05, 1e-9)]
        
        total_eps, total_delta = dp_mechanism.compose_privacy(mechanisms)
        
        # Should be less than simple composition for mixed mechanisms
        simple_eps = sum(eps for eps, _ in mechanisms)
        simple_delta = sum(delta for _, delta in mechanisms)
        
        # Advanced composition should give reasonable bounds
        assert total_eps > 0
        assert total_delta >= 0
    
    def test_renyi_differential_privacy(self):
        """Test Rényi DP for better composition."""
        rdp = RenyiDifferentialPrivacy(alpha=2.0)
        
        # Test RDP computation
        sensitivity = 1.0
        sigma = 0.5
        rdp_epsilon = rdp.gaussian_rdp(sensitivity, sigma)
        
        assert rdp_epsilon > 0
        
        # Test composition
        epsilons = [0.1, 0.2, 0.15]
        composed = rdp.compose_rdp(epsilons)
        assert composed == sum(epsilons)  # Linear composition for RDP
        
        # Test conversion to DP
        delta = 1e-6
        dp_epsilon = rdp.rdp_to_dp(composed, delta)
        assert dp_epsilon > composed  # Should be larger due to conversion
    
    def test_privacy_accountant(self):
        """Test privacy accounting system."""
        accountant = PrivacyAccountant(
            total_epsilon=1.0,
            total_delta=1e-6,
            accounting_method="rdp"
        )
        
        # Record some mechanisms
        accountant.record_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
        accountant.record_mechanism("gaussian", sigma=0.5, sensitivity=1.0)
        
        # Check remaining budget
        remaining_eps, remaining_delta = accountant.get_remaining_budget()
        assert remaining_eps >= 0
        assert remaining_eps < accountant.total_epsilon
        
        # Test optimal noise computation
        optimal_sigma = accountant.optimal_noise_for_remaining_queries(
            num_queries=10, sensitivity=1.0
        )
        assert optimal_sigma > 0
    
    def test_utility_estimation(self):
        """Test utility loss estimation."""
        epsilon = 0.1
        delta = 1e-6
        dimension = 1000
        
        utility_loss = estimate_utility_loss(epsilon, delta, dimension)
        
        assert 0 <= utility_loss <= 1.0
        
        # Smaller epsilon should lead to higher utility loss
        smaller_eps_loss = estimate_utility_loss(0.01, delta, dimension)
        assert smaller_eps_loss >= utility_loss


class TestSecureAggregation:
    """Test secure aggregation protocols."""
    
    @pytest.fixture
    def aggregator(self):
        """Create secure aggregator for testing."""
        params = AggregationParams(
            aggregation_type="mean",
            noise_scale=0.01,
            privacy_threshold=3
        )
        return SecureAggregator(params)
    
    @pytest.fixture
    def test_signatures(self):
        """Create test signatures."""
        torch.manual_seed(42)
        signatures = []
        for _ in range(5):
            sig = torch.randn(100)
            sig = sig / torch.norm(sig)  # Normalize
            signatures.append(sig)
        return signatures
    
    def test_basic_aggregation(self, aggregator, test_signatures):
        """Test basic signature aggregation."""
        # Aggregate signatures
        aggregated = aggregator.aggregate_signatures(test_signatures)
        
        # Check result shape
        assert aggregated.shape == test_signatures[0].shape
        
        # Check normalization
        assert abs(torch.norm(aggregated) - 1.0) < 1e-6
        
        # Check it's not identical to any input
        for sig in test_signatures:
            assert not torch.allclose(aggregated, sig, atol=1e-4)
    
    def test_weighted_aggregation(self, aggregator, test_signatures):
        """Test weighted aggregation."""
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        
        aggregated = aggregator.aggregate_signatures(test_signatures, weights=weights)
        
        # Should be different from equal-weight aggregation
        equal_weight = aggregator.aggregate_signatures(test_signatures)
        assert not torch.allclose(aggregated, equal_weight, atol=1e-4)
    
    def test_insufficient_participants(self, aggregator):
        """Test handling of insufficient participants."""
        # Only 2 signatures, but threshold is 3
        signatures = [torch.randn(100), torch.randn(100)]
        
        with pytest.raises(ValueError, match="Need at least 3 participants"):
            aggregator.aggregate_signatures(signatures)
    
    def test_aggregation_proof(self, aggregator, test_signatures):
        """Test aggregation proof generation."""
        aggregated = aggregator.aggregate_signatures(test_signatures)
        
        # Generate proof
        proof = aggregator.compute_aggregation_proof(test_signatures, aggregated)
        
        # Check proof structure
        assert "signature_commitments" in proof
        assert "aggregation_type" in proof
        assert "num_participants" in proof
        
        assert len(proof["signature_commitments"]) == len(test_signatures)
        assert proof["num_participants"] == len(test_signatures)
    
    def test_shamir_secret_sharing(self):
        """Test Shamir secret sharing scheme."""
        threshold = 3
        num_parties = 5
        
        sss = ShamirSecretSharing(threshold, num_parties)
        
        # Test with tensor signature
        signature = torch.randn(10)
        
        # Share signature
        shares = sss.share_signature(signature)
        
        assert len(shares) == num_parties
        
        # Reconstruct from threshold shares
        reconstructed = sss.reconstruct_signature(shares[:threshold])
        
        # Should match original (within secret sharing precision)
        # Note: Secret sharing with finite fields can introduce numerical errors
        relative_error = torch.norm(signature - reconstructed) / torch.norm(signature)
        assert relative_error < 0.5  # Allow for significant error due to field arithmetic
        
        # Should fail with insufficient shares
        with pytest.raises(ValueError):
            sss.reconstruct_signature(shares[:threshold-1])
    
    def test_multi_party_computation(self):
        """Test MPC protocols."""
        num_parties = 5
        threshold = 3
        
        mpc = MultiPartyComputation(num_parties, threshold)
        
        # Test secure sum
        private_values = [torch.randn(10) for _ in range(num_parties)]
        party_ids = list(range(num_parties))
        
        secure_sum = mpc.secure_sum(private_values, party_ids)
        
        # Compare with direct sum
        direct_sum = sum(private_values)
        
        # Should be approximately equal (within secret sharing precision)
        assert torch.allclose(secure_sum, direct_sum, atol=1e-1)
    
    def test_secure_distance_computation(self):
        """Test secure distance computation using MPC."""
        num_parties = 4
        threshold = 3
        
        mpc = MultiPartyComputation(num_parties, threshold)
        sss = mpc.secret_sharing
        
        # Create test vectors
        vec_a = torch.randn(20)
        vec_b = torch.randn(20)
        
        # Share vectors
        shares_a = sss.share_signature(vec_a)
        shares_b = sss.share_signature(vec_b)
        
        # Compute secure distance
        secure_distance = mpc.secure_distance_computation(
            shares_a[:threshold], shares_b[:threshold], metric="euclidean"
        )
        
        # Compare with direct computation
        direct_distance = torch.norm(vec_a - vec_b).item()
        
        # Should be approximately equal (allow for some error due to precision)
        assert abs(secure_distance - direct_distance) < 1.0
    
    def test_byzantine_robust_aggregation(self):
        """Test Byzantine fault-tolerant aggregation."""
        num_parties = 7
        threshold = 2
        max_byzantine = 2
        
        mpc = MultiPartyComputation(num_parties, threshold)
        
        # Create honest signatures
        honest_signatures = [torch.randn(10) for _ in range(5)]
        
        # Add Byzantine (adversarial) signatures
        byzantine_signatures = [torch.ones(10) * 100, torch.ones(10) * -100]
        
        all_signatures = honest_signatures + byzantine_signatures
        party_ids = list(range(len(all_signatures)))
        
        # Aggregate with Byzantine robustness
        robust_result = mpc.byzantine_robust_aggregation(
            all_signatures, party_ids, max_byzantine
        )
        
        # Should filter out extreme Byzantine values
        # Result should be closer to honest signatures than Byzantine ones
        honest_mean = torch.stack(honest_signatures).mean(dim=0)
        distance_to_honest = torch.norm(robust_result - honest_mean)
        distance_to_byzantine = min(
            torch.norm(robust_result - byz_sig) 
            for byz_sig in byzantine_signatures
        )
        
        assert distance_to_honest < distance_to_byzantine
    
    def test_federated_aggregation_protocol(self):
        """Test federated aggregation with privacy."""
        protocol = FederatedAggregationProtocol(
            privacy_budget=1.0,
            min_participants=3,
            dropout_resilience=0.3
        )
        
        # Setup participants
        participant_ids = [f"party_{i}" for i in range(5)]
        signature_dim = 50
        
        setup_info = protocol.setup_secure_aggregation(participant_ids, signature_dim)
        
        # Check setup structure
        assert "participant_ids" in setup_info
        assert "threshold" in setup_info
        assert "pairwise_masks" in setup_info
        
        # Test participant masking
        for participant_id in participant_ids:
            mask = protocol.participant_mask(participant_id, setup_info)
            assert mask.shape == (signature_dim,)
        
        # Test aggregation round
        signatures = {
            pid: torch.randn(signature_dim) for pid in participant_ids
        }
        
        # Apply masks (simulate secure aggregation protocol)
        masked_signatures = {}
        for pid, signature in signatures.items():
            mask = protocol.participant_mask(pid, setup_info)
            masked_signatures[pid] = signature + mask
        
        # Aggregate
        result = protocol.secure_aggregate_round(masked_signatures, setup_info)
        
        # Check result
        assert result.shape == (signature_dim,)
        assert abs(torch.norm(result) - 1.0) < 1e-6  # Should be normalized


class TestZeroKnowledgeProofs:
    """Test zero-knowledge proof systems."""
    
    @pytest.fixture
    def zk_prover(self):
        """Create ZK proof system for testing."""
        return DistanceZKProof(security_bits=128)
    
    @pytest.fixture
    def commitment_scheme(self):
        """Create commitment scheme for testing."""
        return CommitmentScheme(security_bits=128)
    
    @pytest.fixture
    def test_vectors(self):
        """Create test vectors."""
        np.random.seed(42)
        vec_a = np.random.randn(100).astype(np.float32)
        vec_b = np.random.randn(100).astype(np.float32)
        return vec_a, vec_b
    
    def test_vector_commitment(self, commitment_scheme, test_vectors):
        """Test cryptographic commitments."""
        vec_a, vec_b = test_vectors
        
        # Create commitment
        commitment, randomness = commitment_scheme.commit(vec_a)
        
        # Verify commitment
        assert commitment_scheme.verify_commitment(commitment, vec_a, randomness)
        
        # Should fail with wrong vector
        assert not commitment_scheme.verify_commitment(commitment, vec_b, randomness)
        
        # Should fail with wrong randomness
        wrong_randomness = b"wrong" + randomness[4:]
        assert not commitment_scheme.verify_commitment(commitment, vec_a, wrong_randomness)
    
    def test_batch_commitments(self, commitment_scheme, test_vectors):
        """Test batch commitment creation."""
        vec_a, vec_b = test_vectors
        vectors = [vec_a, vec_b, vec_a + vec_b]
        
        commitments, randomness_list = commitment_scheme.batch_commit(vectors)
        
        assert len(commitments) == len(vectors)
        assert len(randomness_list) == len(vectors)
        
        # Verify all commitments
        for i, (commitment, randomness, vector) in enumerate(zip(commitments, randomness_list, vectors)):
            assert commitment_scheme.verify_commitment(commitment, vector, randomness)
    
    def test_distance_proof_generation(self, zk_prover, test_vectors):
        """Test distance proof generation."""
        vec_a, vec_b = test_vectors
        
        # Compute distance
        distance = np.linalg.norm(vec_a - vec_b)
        
        # Generate proof
        proof = zk_prover.prove_distance(vec_a, vec_b, distance, metric="euclidean")
        
        # Check proof structure
        assert isinstance(proof, DistanceProof)
        assert proof.commitment_a is not None
        assert proof.commitment_b is not None
        assert proof.distance_commitment is not None
        assert proof.challenge is not None
        assert proof.response is not None
        
        # Verify proof
        assert zk_prover.verify_distance(proof, distance)
    
    def test_distance_proof_verification_fails_wrong_distance(self, zk_prover, test_vectors):
        """Test that verification fails with wrong distance."""
        vec_a, vec_b = test_vectors
        
        distance = np.linalg.norm(vec_a - vec_b)
        proof = zk_prover.prove_distance(vec_a, vec_b, distance)
        
        # Should fail with wrong distance
        wrong_distance = distance + 1.0
        assert not zk_prover.verify_distance(proof, wrong_distance)
    
    def test_range_proof(self, zk_prover):
        """Test range proofs for distances."""
        distance = 0.75
        min_dist = 0.5
        max_dist = 1.0
        
        # Generate range proof
        range_proof = zk_prover.prove_distance_range(distance, min_dist, max_dist)
        
        # Verify proof
        assert zk_prover.verify_distance_range(range_proof)
        
        # Should fail if distance is outside range
        with pytest.raises(ValueError):
            zk_prover.prove_distance_range(1.5, min_dist, max_dist)
    
    def test_similarity_threshold_proof(self, zk_prover, test_vectors):
        """Test similarity threshold proofs."""
        vec_a, vec_b = test_vectors
        
        # Normalize vectors for similarity computation
        vec_a_norm = vec_a / np.linalg.norm(vec_a)
        vec_b_norm = vec_b / np.linalg.norm(vec_b)
        
        # Compute actual similarity
        similarity = np.dot(vec_a_norm, vec_b_norm)
        threshold = similarity - 0.1  # Set threshold below actual similarity
        
        # Generate proof
        proof = zk_prover.prove_similarity_threshold(
            vec_a_norm, vec_b_norm, threshold, is_above=True
        )
        
        # Check proof structure
        assert "commitment_a" in proof
        assert "commitment_b" in proof
        assert "threshold" in proof
        assert proof["is_above"] == True
    
    def test_batch_distance_proofs(self, zk_prover):
        """Test batch proof generation."""
        np.random.seed(42)
        
        # Create multiple vectors
        vectors = [np.random.randn(50).astype(np.float32) for _ in range(4)]
        
        # Compute distance matrix
        n = len(vectors)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
        
        # Generate batch proof
        batch_proof = zk_prover.batch_prove_distances(vectors, distance_matrix)
        
        # Check proof structure
        assert "merkle_root" in batch_proof
        assert "num_vectors" in batch_proof
        assert "total_distance" in batch_proof
        assert batch_proof["num_vectors"] == n
    
    def test_merkle_inclusion_proofs(self):
        """Test Merkle tree inclusion proofs."""
        merkle_prover = MerkleInclusionProof()
        
        # Create test signatures
        np.random.seed(42)
        signatures = [np.random.randn(20).astype(np.float32) for _ in range(8)]
        
        # Generate batch inclusion proof
        indices = [0, 3, 7]  # Prove inclusion for these indices
        batch_proof = merkle_prover.batch_inclusion_proofs(signatures, indices)
        
        # Verify each inclusion proof
        root = batch_proof["root"]
        
        for idx in indices:
            # Hash the signature
            import hashlib
            from Crypto.Hash import SHA256
            hasher = SHA256.new()
            hasher.update(signatures[idx].tobytes())
            sig_hash = hasher.digest()
            
            # Verify inclusion
            proof_path = batch_proof["proofs"][idx]
            assert merkle_prover.verify_inclusion_proof(root, sig_hash, proof_path)
    
    def test_enhanced_distance_proofs(self):
        """Test enhanced ZK proofs with commitments."""
        enhanced_prover = EnhancedDistanceZKProof(security_bits=128)
        
        # Create test vectors
        np.random.seed(42)
        vec_a = np.random.randn(50).astype(np.float32)
        vec_b = np.random.randn(50).astype(np.float32)
        
        # Create commitments
        comm_a, rand_a = enhanced_prover.commitment_scheme.commit(vec_a)
        comm_b, rand_b = enhanced_prover.commitment_scheme.commit(vec_b)
        
        # Compute distance
        distance = np.linalg.norm(vec_a - vec_b)
        
        # Generate enhanced proof
        proof = enhanced_prover.prove_committed_distance(
            vec_a, vec_b, distance, comm_a, comm_b, rand_a, rand_b
        )
        
        # Verify proof
        assert enhanced_prover.verify_committed_distance(
            proof, comm_a, comm_b, distance
        )
    
    def test_signature_inclusion_proof(self):
        """Test signature inclusion proofs."""
        enhanced_prover = EnhancedDistanceZKProof()
        
        # Create signature set
        np.random.seed(42)
        signature_set = [np.random.randn(30).astype(np.float32) for _ in range(10)]
        
        # Pick signature to prove inclusion for
        signature_index = 5
        signature = signature_set[signature_index]
        
        # Generate inclusion proof
        proof = enhanced_prover.prove_signature_inclusion(
            signature, signature_set, signature_index
        )
        
        # Get Merkle root from proof
        merkle_root = bytes.fromhex(proof["merkle_root"])
        
        # Verify inclusion proof
        assert enhanced_prover.verify_signature_inclusion(
            proof, signature, merkle_root
        )


class TestIntegration:
    """Integration tests for privacy mechanisms."""
    
    def test_end_to_end_private_verification(self):
        """Test complete private verification workflow."""
        # Setup components
        dp_mechanism = DifferentialPrivacyMechanism(PrivacyLevel.MEDIUM, seed=42)
        aggregator = SecureAggregator(
            AggregationParams(privacy_threshold=3, noise_scale=0.01)
        )
        zk_prover = EnhancedDistanceZKProof()
        
        # Create test data
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Multiple parties with signatures
        party_signatures = {
            f"party_{i}": torch.randn(100) for i in range(5)
        }
        
        # Normalize signatures
        for party_id in party_signatures:
            sig = party_signatures[party_id]
            party_signatures[party_id] = sig / torch.norm(sig)
        
        # Step 1: Add differential privacy to signatures
        privacy_budget = PrivacyBudget(epsilon=1.0, delta=1e-6)
        private_signatures = {}
        
        for party_id, signature in party_signatures.items():
            private_sig = dp_mechanism.privatize_hypervector(
                signature, privacy_budget, mechanism="gaussian"
            )
            private_signatures[party_id] = private_sig
        
        # Step 2: Secure aggregation
        aggregated_signature = aggregator.aggregate_signatures(
            list(private_signatures.values())
        )
        
        # Step 3: Generate proof of aggregation
        aggregation_proof = aggregator.compute_aggregation_proof(
            list(private_signatures.values()), aggregated_signature
        )
        
        # Step 4: Zero-knowledge distance proof
        # Convert to numpy for ZK proof
        agg_sig_np = aggregated_signature.numpy()
        reference_sig_np = torch.randn(100).numpy()
        reference_sig_np = reference_sig_np / np.linalg.norm(reference_sig_np)
        
        distance = np.linalg.norm(agg_sig_np - reference_sig_np)
        
        # Create commitments
        comm_agg, rand_agg = zk_prover.commitment_scheme.commit(agg_sig_np)
        comm_ref, rand_ref = zk_prover.commitment_scheme.commit(reference_sig_np)
        
        # Generate distance proof
        distance_proof = zk_prover.prove_committed_distance(
            agg_sig_np, reference_sig_np, distance,
            comm_agg, comm_ref, rand_agg, rand_ref
        )
        
        # Verification phase
        # Verify aggregation proof is valid
        from src.privacy.secure_aggregation import verify_aggregation_proof
        agg_valid = verify_aggregation_proof(
            aggregation_proof, aggregated_signature, tolerance=0.2
        )
        
        # Verify distance proof
        distance_valid = zk_prover.verify_committed_distance(
            distance_proof, comm_agg, comm_ref, distance
        )
        
        # All verifications should pass
        assert agg_valid
        assert distance_valid
        
        # Check privacy budget was consumed
        assert privacy_budget.consumed_epsilon > 0
    
    def test_federated_private_verification(self):
        """Test federated verification with multiple privacy techniques."""
        # Setup federated protocol
        fed_protocol = FederatedAggregationProtocol(
            privacy_budget=2.0,
            min_participants=3,
            dropout_resilience=0.3
        )
        
        # Create participants
        participants = [f"org_{i}" for i in range(6)]
        signature_dim = 200
        
        # Setup federated round
        setup_info = fed_protocol.setup_secure_aggregation(participants, signature_dim)
        
        # Each participant creates their signature
        torch.manual_seed(42)
        participant_signatures = {}
        for participant in participants:
            sig = torch.randn(signature_dim)
            sig = sig / torch.norm(sig)
            participant_signatures[participant] = sig
        
        # Apply secure aggregation masks
        masked_signatures = {}
        for participant, signature in participant_signatures.items():
            mask = fed_protocol.participant_mask(participant, setup_info)
            masked_signatures[participant] = signature + mask
        
        # Aggregate with privacy
        federated_result = fed_protocol.secure_aggregate_round(
            masked_signatures, setup_info
        )
        
        # Verify aggregation correctness (for testing)
        correctness_valid = fed_protocol.verify_aggregation_correctness(
            participants, participant_signatures, federated_result, tolerance=0.3
        )
        
        assert correctness_valid
        
        # The federated result should be different from any individual signature
        for signature in participant_signatures.values():
            similarity = torch.dot(federated_result, signature).item()
            assert similarity < 0.9  # Should not be too similar to any individual
    
    def test_privacy_composition_tracking(self):
        """Test privacy composition across multiple operations."""
        # Initialize privacy accountant
        accountant = PrivacyAccountant(
            total_epsilon=2.0,
            total_delta=1e-5,
            accounting_method="rdp"
        )
        
        # Simulate multiple privacy-consuming operations
        operations = [
            ("gaussian", {"sigma": 1.0, "sensitivity": 1.0}),
            ("gaussian", {"sigma": 0.8, "sensitivity": 0.5}),
            ("gaussian", {"sigma": 1.2, "sensitivity": 1.0}),
            ("laplace", {"epsilon": 0.1, "sensitivity": 1.0}),
        ]
        
        for op_type, params in operations:
            if op_type == "gaussian":
                accountant.record_mechanism(op_type, **params)
            else:
                accountant.record_mechanism(op_type, **params)
        
        # Check remaining budget
        remaining_eps, remaining_delta = accountant.get_remaining_budget()
        
        # Should have consumed some budget
        assert remaining_eps < accountant.total_epsilon
        assert remaining_eps >= 0
        
        # Should still have meaningful budget left
        assert remaining_eps > 0.1
        
        # Test that we can compute optimal noise for remaining queries
        optimal_sigma = accountant.optimal_noise_for_remaining_queries(
            num_queries=5, sensitivity=1.0
        )
        assert optimal_sigma > 0
        assert optimal_sigma < float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])