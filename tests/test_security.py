"""
Comprehensive security tests for REV security features.
Tests ZK proofs, rate limiting, Merkle trees, and attestation.
"""

import pytest
import time
import numpy as np
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.security.zk_attestation import (
    ZKAttestationSystem,
    ZKProof,
    BulletProof,
    ZKCircuit,
    PedersenCommitment
)
from src.security.rate_limiter import (
    LocalRateLimiter,
    RedisRateLimiter,
    HierarchicalRateLimiter,
    AdaptiveRateLimiter,
    RateLimitConfig,
    TokenBucket
)
from src.crypto.merkle_tree import (
    MerkleTree,
    SparseMerkleTree,
    HSMIntegratedMerkleTree,
    MerkleProof,
    SparseMerkleProof
)
from src.security.attestation_server import (
    AttestationServer,
    AttestationReport,
    TEEConfig,
    create_attestation_server
)


class TestZKAttestation:
    """Test zero-knowledge attestation system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.zk_system = ZKAttestationSystem()
        self.dimension = 1000
        
        # Generate test fingerprints
        self.fingerprint1 = np.random.randn(self.dimension)
        self.fingerprint2 = np.random.randn(self.dimension)
    
    def test_pedersen_commitment(self):
        """Test Pedersen commitment creation and properties."""
        # Create commitment
        commitment = self.zk_system.create_pedersen_commitment(self.fingerprint1)
        
        assert commitment is not None
        assert commitment.commitment is not None
        assert len(commitment.commitment) == 32  # SHA256 hash size
        assert commitment.randomness is not None
        
        # Test hiding property - same value with different randomness
        commitment2 = self.zk_system.create_pedersen_commitment(self.fingerprint1)
        assert commitment.commitment != commitment2.commitment  # Different randomness
        
        # Test binding property - can't create same commitment for different values
        commitment3 = self.zk_system.create_pedersen_commitment(self.fingerprint2)
        assert commitment.commitment != commitment3.commitment
    
    def test_distance_proof_generation(self):
        """Test zero-knowledge proof of distance computation."""
        # Compute actual distance
        distance = np.linalg.norm(self.fingerprint1 - self.fingerprint2)
        
        # Generate proof
        proof = self.zk_system.prove_distance_computation(
            self.fingerprint1,
            self.fingerprint2,
            distance
        )
        
        assert proof is not None
        assert proof.proof_type == "distance"
        assert proof.public_inputs["distance"] == distance
        assert proof.commitment is not None
        assert proof.challenge is not None
        assert proof.response is not None
    
    def test_distance_proof_verification(self):
        """Test verification of distance proofs."""
        distance = np.linalg.norm(self.fingerprint1 - self.fingerprint2)
        
        # Generate proof
        proof = self.zk_system.prove_distance_computation(
            self.fingerprint1,
            self.fingerprint2,
            distance
        )
        
        # Create commitments
        commitment1 = self.zk_system.create_pedersen_commitment(self.fingerprint1)
        commitment2 = self.zk_system.create_pedersen_commitment(self.fingerprint2)
        
        # Verify proof
        is_valid = self.zk_system.verify_distance_proof(
            proof,
            commitment1.commitment,
            commitment2.commitment
        )
        
        assert is_valid
        
        # Test invalid proof
        fake_proof = ZKProof(
            proof_type="distance",
            commitment=proof.commitment,
            challenge=b"wrong_challenge",
            response=proof.response,
            public_inputs=proof.public_inputs,
            metadata=proof.metadata
        )
        
        is_valid = self.zk_system.verify_distance_proof(
            fake_proof,
            commitment1.commitment,
            commitment2.commitment
        )
        
        assert not is_valid
    
    def test_range_proof(self):
        """Test Bulletproof range proofs."""
        # Test value in range
        value = 0.75
        range_min = 0.0
        range_max = 1.0
        
        # Generate range proof
        proof = self.zk_system.create_range_proof(
            value, range_min, range_max
        )
        
        assert proof is not None
        assert proof.range_min == range_min
        assert proof.range_max == range_max
        assert proof.proof_data is not None
        
        # Verify proof
        commitment = hashlib.sha256(np.array([value]).tobytes()).digest()
        is_valid = self.zk_system.verify_range_proof(proof, commitment)
        
        assert is_valid
    
    def test_range_proof_out_of_bounds(self):
        """Test range proof fails for out-of-bounds values."""
        value = 1.5  # Outside [0, 1]
        
        with pytest.raises(ValueError):
            self.zk_system.create_range_proof(value, 0.0, 1.0)
    
    def test_membership_proof(self):
        """Test zero-knowledge membership proofs."""
        # Create Merkle tree path
        merkle_path = [
            hashlib.sha256(f"node_{i}".encode()).digest()
            for i in range(10)
        ]
        merkle_root = hashlib.sha256(b"root").digest()
        
        # Create membership proof
        proof = self.zk_system.create_membership_proof(
            self.fingerprint1,
            merkle_path,
            merkle_root
        )
        
        assert proof is not None
        assert proof.proof_type == "membership"
        assert proof.public_inputs["merkle_root"] == merkle_root.hex()
    
    def test_batch_verification(self):
        """Test batch verification of multiple proofs."""
        proofs = []
        commitments = []
        
        # Generate multiple proofs
        for i in range(5):
            fp = np.random.randn(self.dimension)
            commitment = self.zk_system.create_pedersen_commitment(fp)
            
            # Create membership proof
            merkle_path = [hashlib.sha256(f"node_{j}".encode()).digest() for j in range(5)]
            proof = self.zk_system.create_membership_proof(
                fp,
                merkle_path,
                hashlib.sha256(b"root").digest()
            )
            
            proofs.append(proof)
            commitments.append(commitment.commitment)
        
        # Batch verify
        all_valid = self.zk_system.batch_verify_proofs(proofs, commitments)
        assert all_valid
    
    def test_zk_circuit(self):
        """Test zk-SNARK circuit for distance computation."""
        circuit = ZKCircuit(num_constraints=100)
        
        # Add distance constraints
        circuit.add_distance_constraints(
            dimension=10,
            distance_type="hamming"
        )
        
        assert len(circuit.constraints) > 0
        
        # Generate proof
        private_inputs = {"fp1": self.fingerprint1, "fp2": self.fingerprint2}
        public_inputs = {"distance": 0.5}
        
        proof = circuit.generate_proof(private_inputs, public_inputs)
        
        assert proof is not None
        assert "pi_a" in proof
        assert "pi_b" in proof
        assert "pi_c" in proof
        
        # Verify proof
        is_valid = circuit.verify_proof(proof, {})
        assert is_valid


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            window_size=60,
            max_backoff=10.0
        )
        self.limiter = LocalRateLimiter(self.config)
    
    def test_token_bucket_basic(self):
        """Test basic token bucket functionality."""
        key = "test_user"
        
        # Initial requests should succeed
        for i in range(10):
            result = self.limiter.check_rate_limit(key, cost=1.0)
            assert result.allowed
            assert result.tokens_remaining >= 0
    
    def test_token_bucket_exhaustion(self):
        """Test token bucket exhaustion."""
        key = "test_user"
        
        # Exhaust tokens
        for i in range(20):  # burst_size = 20
            result = self.limiter.check_rate_limit(key, cost=1.0)
        
        # Next request should be denied
        result = self.limiter.check_rate_limit(key, cost=1.0)
        assert not result.allowed
        assert result.retry_after is not None
        assert result.retry_after > 0
    
    def test_token_refill(self):
        """Test token refill over time."""
        key = "test_user"
        
        # Exhaust tokens
        for i in range(20):
            self.limiter.check_rate_limit(key, cost=1.0)
        
        # Wait for refill
        time.sleep(0.2)  # 10 req/s = 2 tokens in 0.2s
        
        # Should have some tokens now
        result = self.limiter.check_rate_limit(key, cost=1.0)
        assert result.allowed or result.retry_after < 1.0
    
    def test_exponential_backoff(self):
        """Test exponential backoff on repeated failures."""
        key = "test_user"
        
        # Exhaust tokens
        for i in range(20):
            self.limiter.check_rate_limit(key, cost=1.0)
        
        # Record retry times
        retry_times = []
        for i in range(3):
            result = self.limiter.check_rate_limit(key, cost=1.0)
            assert not result.allowed
            retry_times.append(result.retry_after)
            time.sleep(0.01)  # Small delay
        
        # Retry times should increase
        assert retry_times[1] >= retry_times[0]
        assert retry_times[2] >= retry_times[1]
    
    def test_model_specific_limits(self):
        """Test model-specific rate limits."""
        # Set model-specific limit
        self.config.model_limits["gpt-4"] = {
            "requests_per_second": 5.0,
            "burst_size": 10
        }
        
        limiter = LocalRateLimiter(self.config)
        
        # Test model limit
        key = "model:gpt-4"
        
        # Should allow 10 requests (burst)
        for i in range(10):
            result = limiter.check_rate_limit(key, cost=1.0)
            assert result.allowed
        
        # 11th should be denied
        result = limiter.check_rate_limit(key, cost=1.0)
        assert not result.allowed
    
    def test_hierarchical_rate_limiter(self):
        """Test hierarchical rate limiting."""
        limiter = HierarchicalRateLimiter(self.config)
        
        # Test multiple levels
        result = limiter.check_rate_limit(
            user_id="user123",
            model_id="gpt-4",
            operation="inference",
            cost=1.0
        )
        
        assert result.allowed
        assert result.tokens_remaining >= 0
        
        # Check quota status
        status = limiter.get_quota_status(
            user_id="user123",
            model_id="gpt-4"
        )
        
        assert "quotas" in status
        assert "statistics" in status
    
    def test_concurrent_rate_limiting(self):
        """Test thread-safe rate limiting."""
        limiter = LocalRateLimiter(self.config)
        key = "concurrent_test"
        
        allowed_count = 0
        denied_count = 0
        
        def make_request():
            result = limiter.check_rate_limit(key, cost=1.0)
            return result.allowed
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            
            for future in as_completed(futures):
                if future.result():
                    allowed_count += 1
                else:
                    denied_count += 1
        
        # Should respect burst limit
        assert allowed_count <= 20  # burst_size
        assert denied_count == 50 - allowed_count
    
    @pytest.mark.skip(reason="Requires Redis")
    def test_redis_rate_limiter(self):
        """Test Redis-backed rate limiter."""
        config = RateLimitConfig(
            enable_distributed=True,
            redis_url="redis://localhost:6379"
        )
        
        limiter = RedisRateLimiter(config)
        
        # Should fall back to local if Redis not available
        result = limiter.check_rate_limit("test_key", cost=1.0)
        assert result is not None
    
    def test_adaptive_rate_limiter(self):
        """Test adaptive rate limiting based on system load."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20
        )
        
        limiter = AdaptiveRateLimiter(
            base_config=config,
            load_threshold=0.8,
            adaptation_rate=0.1
        )
        
        # Test basic functionality
        result = limiter.check_rate_limit(
            user_id="test",
            operation="inference"
        )
        
        assert result.allowed
        
        # Simulate load changes (would need system metrics in real test)
        time.sleep(0.1)


class TestMerkleTrees:
    """Test Merkle tree implementations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.data = [
            f"data_{i}".encode()
            for i in range(8)
        ]
    
    def test_merkle_tree_construction(self):
        """Test standard Merkle tree construction."""
        tree = MerkleTree()
        tree.build(self.data)
        
        assert tree.root is not None
        assert len(tree.leaves) == len(self.data)
        
        # Verify tree structure
        for i, leaf in enumerate(tree.leaves):
            assert leaf.is_leaf
            assert leaf.index == i
            assert leaf.value == self.data[i]
    
    def test_merkle_proof_generation(self):
        """Test Merkle proof generation."""
        tree = MerkleTree()
        tree.build(self.data)
        
        # Get proof for each element
        for i in range(len(self.data)):
            proof = tree.get_proof(i)
            
            assert proof is not None
            assert proof.leaf_index == i
            assert proof.root == tree.root.hash
            assert len(proof.siblings) > 0
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        tree = MerkleTree()
        tree.build(self.data)
        
        # Generate and verify proofs
        for i in range(len(self.data)):
            proof = tree.get_proof(i)
            is_valid = tree.verify_proof(proof)
            assert is_valid
        
        # Test invalid proof
        proof = tree.get_proof(0)
        proof.leaf_hash = b"wrong_hash"
        is_valid = tree.verify_proof(proof)
        assert not is_valid
    
    def test_batch_verification(self):
        """Test batch Merkle proof verification."""
        tree = MerkleTree()
        tree.build(self.data)
        
        # Generate multiple proofs
        proofs = [tree.get_proof(i) for i in range(len(self.data))]
        
        # Batch verify
        start = time.time()
        all_valid = tree.batch_verify(proofs)
        batch_time = time.time() - start
        
        assert all_valid
        
        # Compare with individual verification
        start = time.time()
        for proof in proofs:
            tree.verify_proof(proof)
        individual_time = time.time() - start
        
        # Batch should be comparable or faster
        print(f"Batch: {batch_time:.4f}s, Individual: {individual_time:.4f}s")
    
    def test_sparse_merkle_tree(self):
        """Test sparse Merkle tree."""
        tree = SparseMerkleTree(height=256)
        
        # Update some keys
        key1 = hashlib.sha256(b"key1").digest()
        value1 = b"value1"
        
        tree.update(key1, value1)
        
        # Get inclusion proof
        proof = tree.get_proof(key1)
        
        assert proof is not None
        assert proof.inclusion
        assert proof.key == key1
        
        # Verify proof
        is_valid = tree.verify_proof(proof)
        assert is_valid
        
        # Get non-inclusion proof
        key2 = hashlib.sha256(b"key2").digest()
        proof2 = tree.get_proof(key2)
        
        assert not proof2.inclusion
        assert proof2.value is None
        
        # Verify non-inclusion
        is_valid = tree.verify_proof(proof2)
        assert is_valid
    
    def test_hsm_merkle_tree(self):
        """Test HSM-integrated Merkle tree."""
        tree = HSMIntegratedMerkleTree(
            enable_caching=True,
            hsm_config={"type": "softhsm"}
        )
        
        tree.build(self.data)
        
        assert tree.root is not None
        
        # Test signing
        signature = tree.sign_root("test_key_id")
        assert signature is not None
        
        # Test verification
        is_valid = tree.verify_signature(
            signature,
            tree.root.hash,
            "test_key_id"
        )
        assert is_valid


class TestAttestationServer:
    """Test attestation server functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.server = create_attestation_server({
            "port": 8081,
            "enable_tee": False,
            "enable_hsm": False,
            "secret_key": "test_secret_key"
        })
        self.app = self.server.app
        self.client = self.app.test_client()
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_attestation_creation(self):
        """Test fingerprint attestation creation."""
        fingerprint = np.random.randn(100).tolist()
        
        response = self.client.post(
            '/attest/fingerprint',
            json={
                "fingerprint": fingerprint,
                "model_id": "test_model",
                "type": "membership"
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "report_id" in data
        assert "timestamp" in data
        assert "signature" in data
    
    def test_attestation_verification(self):
        """Test attestation verification."""
        # Create attestation
        fingerprint = np.random.randn(100).tolist()
        
        response = self.client.post(
            '/attest/fingerprint',
            json={
                "fingerprint": fingerprint,
                "model_id": "test_model"
            }
        )
        
        report_id = response.get_json()["report_id"]
        
        # Verify attestation
        response = self.client.get(f'/verify/attestation/{report_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["report_id"] == report_id
        assert data["valid"] is True
    
    def test_distance_proof_endpoint(self):
        """Test distance proof generation."""
        fp1 = np.random.randn(100)
        fp2 = np.random.randn(100)
        distance = np.linalg.norm(fp1 - fp2)
        
        response = self.client.post(
            '/prove/distance',
            json={
                "fingerprint1": fp1.tobytes().hex(),
                "fingerprint2": fp2.tobytes().hex(),
                "distance": distance
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "proof" in data
        assert data["proof"]["type"] == "distance"
    
    def test_range_proof_endpoint(self):
        """Test range proof generation."""
        response = self.client.post(
            '/prove/range',
            json={
                "value": 0.75,
                "min": 0.0,
                "max": 1.0
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "proof" in data
        assert "commitment" in data["proof"]
    
    def test_fingerprint_registration(self):
        """Test fingerprint registration."""
        # Generate auth token
        token = self.server.generate_auth_token("test_user")
        
        response = self.client.post(
            '/register/fingerprint',
            headers={"Authorization": f"Bearer {token}"},
            json={
                "fingerprint": np.random.randn(100).tolist(),
                "model_id": "test_model",
                "metadata": {"version": "1.0"}
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "fingerprint_id" in data
        assert data["registered"] is True
    
    def test_rate_limiting(self):
        """Test rate limiting on endpoints."""
        # Make many requests quickly
        responses = []
        for i in range(15):  # Limit is 10 per minute
            response = self.client.post(
                '/attest/fingerprint',
                json={
                    "fingerprint": [i],
                    "model_id": "test"
                }
            )
            responses.append(response.status_code)
        
        # Some should be rate limited (429)
        assert 429 in responses
    
    def test_audit_log(self):
        """Test audit log functionality."""
        # Generate admin token
        token = self.server.generate_auth_token("admin", role="admin")
        
        # Create some attestations
        for i in range(3):
            self.client.post(
                '/attest/fingerprint',
                json={
                    "fingerprint": [i],
                    "model_id": "test"
                }
            )
        
        # Get audit log
        response = self.client.get(
            '/audit/log?limit=10',
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "entries" in data
        assert len(data["entries"]) > 0


class TestEndToEndSecurity:
    """End-to-end security workflow tests."""
    
    def test_complete_attestation_workflow(self):
        """Test complete attestation workflow."""
        # Setup components
        zk_system = ZKAttestationSystem()
        tree = MerkleTree()
        server = create_attestation_server({"port": 8082})
        
        # Generate fingerprint
        fingerprint = np.random.randn(1000)
        
        # Create Pedersen commitment
        commitment = zk_system.create_pedersen_commitment(fingerprint)
        
        # Add to Merkle tree
        tree.build([commitment.commitment])
        proof = tree.get_proof(0)
        
        # Create ZK membership proof
        zk_proof = zk_system.create_membership_proof(
            fingerprint,
            [s[0] for s in proof.siblings],
            tree.root.hash
        )
        
        # Verify everything
        assert tree.verify_proof(proof)
        assert zk_system.verify_proof(
            zk_proof,
            {"merkle_root": tree.root.hash.hex()}
        ) is not False
    
    def test_performance_benchmarks(self):
        """Benchmark security operations."""
        results = {}
        
        # Benchmark ZK proof generation
        zk_system = ZKAttestationSystem()
        fp1 = np.random.randn(10000)
        fp2 = np.random.randn(10000)
        
        start = time.time()
        for _ in range(10):
            proof = zk_system.prove_distance_computation(
                fp1, fp2,
                np.linalg.norm(fp1 - fp2)
            )
        zk_time = (time.time() - start) / 10
        results["zk_proof_generation"] = f"{zk_time*1000:.2f}ms"
        
        # Benchmark Merkle proof
        tree = MerkleTree()
        data = [f"item_{i}".encode() for i in range(1000)]
        
        start = time.time()
        tree.build(data)
        build_time = time.time() - start
        results["merkle_build_1000"] = f"{build_time*1000:.2f}ms"
        
        start = time.time()
        for i in range(100):
            proof = tree.get_proof(i)
            tree.verify_proof(proof)
        verify_time = (time.time() - start) / 100
        results["merkle_verify"] = f"{verify_time*1000:.2f}ms"
        
        # Print results
        print("\nPerformance Benchmarks:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Assert reasonable performance
        assert zk_time < 1.0  # Under 1 second
        assert verify_time < 0.01  # Under 10ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])