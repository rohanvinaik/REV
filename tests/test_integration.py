"""
Integration tests for the full REV pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.rev_pipeline import REVPipeline, REVConfig
from src.executor.segment_runner import SegmentRunner, SegmentConfig
from src.verifier.blackbox import BlackBoxVerifier
from src.hdc.behavioral_sites import BehavioralSites
from src.hdc.encoder import HypervectorEncoder, HypervectorConfig
from src.verifier.decision_aggregator import DecisionAggregator
from src.core.sequential import sequential_verify


class TestREVPipelineIntegration:
    """Integration tests for REV pipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.config = Mock(hidden_size=768, num_hidden_layers=12)
        model.forward = Mock(return_value=Mock(
            logits=np.random.randn(1, 100, 50000),
            hidden_states=(np.random.randn(1, 100, 768) for _ in range(13))
        ))
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=list(range(100)))
        tokenizer.decode = Mock(return_value="test output")
        tokenizer.pad_token_id = 0
        return tokenizer
    
    @pytest.fixture
    def pipeline(self):
        """Create REV pipeline instance."""
        config = REVConfig(
            segment_size=32,
            buffer_size=2,
            memory_limit_gb=1.0,
            hdc_dimension=1000
        )
        return REVPipeline(config)
    
    def test_end_to_end_challenge_processing(self, pipeline, mock_model, mock_tokenizer):
        """Test complete challenge processing flow."""
        challenge = "Explain quantum computing in simple terms."
        
        result = pipeline.process_challenge(
            mock_model,
            challenge,
            tokenizer=mock_tokenizer
        )
        
        assert "segment_signatures" in result
        assert "merkle_root" in result
        assert "behavioral_signature" in result
        assert "metadata" in result
        
        # Verify segment processing
        assert len(result["segment_signatures"]) > 0
        
        # Verify Merkle tree construction
        assert result["merkle_root"] is not None
        assert len(result["merkle_root"]) == 32  # SHA256 hash
        
        # Verify behavioral signature
        assert result["behavioral_signature"]["dimension"] == pipeline.config.hdc_dimension
    
    def test_memory_bounded_execution(self, pipeline, mock_model, mock_tokenizer):
        """Test memory-bounded segment execution."""
        # Long challenge to trigger segmentation
        challenge = " ".join(["test"] * 500)
        
        with patch('src.executor.segment_runner.SegmentRunner.get_memory_usage') as mock_memory:
            mock_memory.return_value = 0.5  # 500MB
            
            result = pipeline.process_challenge(
                mock_model,
                challenge,
                tokenizer=mock_tokenizer
            )
            
            # Should have processed in segments
            assert len(result["segment_signatures"]) > 1
            assert result["metadata"]["memory_efficient"] == True
    
    @patch('src.verifier.blackbox.requests.post')
    def test_api_based_verification(self, mock_post):
        """Test API-based model verification."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "API response"}}]
        }
        mock_post.return_value = mock_response
        
        verifier = BlackBoxVerifier()
        
        # Compare two models
        result = verifier.compare_models(
            model_a="gpt-3.5-turbo",
            model_b="claude-2",
            prompts=["Test prompt 1", "Test prompt 2"]
        )
        
        assert "similarity_scores" in result
        assert "decision" in result
        assert len(result["similarity_scores"]) == 2
    
    def test_decision_aggregation(self):
        """Test decision aggregation across multiple challenges."""
        aggregator = DecisionAggregator()
        
        # Add multiple challenge results
        for i in range(10):
            similarity = 0.95 if i < 7 else 0.3
            aggregator.add_result(f"challenge_{i}", similarity)
        
        # Get final decision
        decision = aggregator.get_final_decision(alpha=0.05, beta=0.10)
        
        assert decision["overall_decision"] in ["accept_h0", "reject_h0", "continue"]
        assert decision["confidence"] > 0
        assert len(decision["per_challenge_scores"]) == 10
    
    def test_behavioral_site_extraction(self):
        """Test behavioral site extraction and encoding."""
        sites = BehavioralSites()
        
        # Simulate model outputs at different layers
        model_outputs = {
            f"layer_{i}": np.random.randn(100, 768)
            for i in range(12)
        }
        
        # Extract features
        probe_features = sites.extract_probe_features("Solve this math problem: 2+2")
        
        # Generate hierarchical signatures
        signatures = sites.hierarchical_analysis(model_outputs, probe_features)
        
        assert "prompt" in signatures
        assert "span_64" in signatures
        assert "token_window_8" in signatures
        
        # Each signature should be a hypervector
        for level, sig in signatures.items():
            assert len(sig) == sites.hdc_config.dimension


class TestSegmentRunnerIntegration:
    """Integration tests for segment runner."""
    
    @pytest.fixture
    def runner(self):
        """Create segment runner instance."""
        config = SegmentConfig(
            max_sequence_length=512,
            kv_cache_size=2048,
            activation_checkpointing=True
        )
        return SegmentRunner(config)
    
    def test_segment_processing_with_cache(self, runner):
        """Test segment processing with KV cache."""
        mock_model = Mock()
        mock_model.forward = Mock(return_value=Mock(
            logits=np.random.randn(1, 32, 50000),
            hidden_states=(np.random.randn(1, 32, 768) for _ in range(13)),
            past_key_values=[(np.random.randn(1, 12, 32, 64),
                            np.random.randn(1, 12, 32, 64)) for _ in range(12)]
        ))
        
        segment_tokens = list(range(32))
        
        # Process first segment
        result1 = runner.process_segment(mock_model, segment_tokens, use_cache=True)
        
        assert "activations" in result1
        assert "kv_cache" in result1
        assert result1["kv_cache"] is not None
        
        # Process second segment with cache
        result2 = runner.process_segment(
            mock_model,
            segment_tokens,
            use_cache=True,
            past_kv=result1["kv_cache"]
        )
        
        assert result2["cache_used"] == True
    
    def test_memory_efficient_mode(self, runner):
        """Test memory-efficient execution mode."""
        mock_model = Mock()
        
        with runner.memory_efficient_mode():
            # Simulate low memory scenario
            runner.config.memory_limit_mb = 100
            
            result = runner.process_segment(
                mock_model,
                list(range(32)),
                use_cache=False
            )
            
            assert result["memory_efficient"] == True
            assert result["kv_cache"] is None  # Cache disabled in memory-efficient mode


class TestPrivacyIntegration:
    """Integration tests for privacy-preserving features."""
    
    def test_homomorphic_verification(self):
        """Test homomorphic operations for privacy-preserving verification."""
        from src.privacy.homomorphic_ops import HomomorphicOperations, FederatedProtocol
        
        # Initialize homomorphic operations
        he_ops = HomomorphicOperations()
        
        # Create test vectors
        vec_a = np.random.randn(100)
        vec_b = np.random.randn(100)
        
        # Encrypt vectors
        encrypted_a = he_ops.encrypt_vector(vec_a)
        encrypted_b = he_ops.encrypt_vector(vec_b)
        
        # Compute distance on encrypted data
        distance = he_ops.compute_encrypted_distance(encrypted_a, encrypted_b)
        
        assert distance > 0
        
        # Verify decryption
        decrypted_a = he_ops.decrypt_vector(encrypted_a)
        np.testing.assert_array_almost_equal(vec_a, decrypted_a, decimal=5)
    
    def test_zero_knowledge_proofs(self):
        """Test ZK proofs for distance computation."""
        from src.privacy.distance_zk_proofs import DistanceZKProof
        
        zk = DistanceZKProof()
        
        # Create test vectors
        vec_a = np.random.randn(100)
        vec_b = np.random.randn(100)
        
        # Compute actual distance
        distance = np.linalg.norm(vec_a - vec_b)
        
        # Generate proof
        proof = zk.prove_distance(vec_a, vec_b, distance, metric="euclidean")
        
        # Verify proof
        is_valid = zk.verify_distance(proof, distance)
        
        assert is_valid == True
        
        # Test invalid proof
        is_invalid = zk.verify_distance(proof, distance * 2)
        assert is_invalid == False
    
    def test_federated_aggregation(self):
        """Test federated aggregation protocol."""
        from src.privacy.homomorphic_ops import FederatedProtocol, HomomorphicOperations
        
        # Initialize protocol
        protocol = FederatedProtocol(num_participants=3, threshold=2)
        
        # Register participants
        he_ops = HomomorphicOperations()
        for i in range(3):
            key = he_ops.key.publickey()
            protocol.register_participant(f"participant_{i}", key)
        
        # Start round
        round_num = protocol.initiate_round()
        
        # Submit encrypted vectors
        for i in range(3):
            vec = np.random.randn(100)
            encrypted = he_ops.encrypt_vector(vec)
            success = protocol.submit_encrypted_vector(
                f"participant_{i}",
                encrypted,
                round_num
            )
            assert success == True
        
        # Aggregate results
        aggregated = protocol.aggregate_round(round_num)
        
        assert aggregated is not None
        assert len(aggregated) == 100


class TestErrorCorrectionIntegration:
    """Integration tests for error correction."""
    
    def test_noisy_channel_recovery(self):
        """Test recovery from noisy channel."""
        from src.hdc.error_correction import ErrorCorrection, ErrorCorrectionConfig
        
        config = ErrorCorrectionConfig(
            dimension=1000,
            parity_overhead=0.25,
            block_size=64
        )
        corrector = ErrorCorrection(config)
        
        # Original hypervector
        original = np.random.choice([-1, 1], size=1000).astype(np.float32)
        
        # Encode with parity
        encoded = corrector.encode_with_parity(original)
        
        # Add noise
        noisy = corrector.add_noise(encoded, noise_level=0.15, noise_type="salt_pepper")
        
        # Decode and correct
        recovered, corrections = corrector.decode_with_correction(noisy, correct_errors=True)
        
        # Measure recovery
        metrics = corrector.measure_robustness(original, noisy[:1000], recovered)
        
        assert metrics["correction_success"] == 1.0
        assert metrics["ber_corrected"] < metrics["ber_noisy"]
        assert metrics["cosine_sim_corrected"] > metrics["cosine_sim_noisy"]


class TestFullPipelineScenarios:
    """Test complete pipeline scenarios."""
    
    @patch('src.verifier.blackbox.requests.post')
    def test_model_comparison_scenario(self, mock_post):
        """Test complete model comparison scenario."""
        # Setup mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response text"}}]
        }
        mock_post.return_value = mock_response
        
        # Initialize components
        pipeline = REVPipeline()
        verifier = BlackBoxVerifier()
        aggregator = DecisionAggregator()
        
        # Test challenges
        challenges = [
            "Explain machine learning",
            "Write a Python function",
            "Translate to French: Hello",
            "Solve: 2x + 3 = 7"
        ]
        
        # Process each challenge
        for challenge in challenges:
            # Get responses from both models
            response_a = verifier.get_model_response("gpt-3.5-turbo", challenge)
            response_b = verifier.get_model_response("claude-2", challenge)
            
            # Compute similarity
            similarity = verifier.compute_similarity(response_a, response_b)
            
            # Add to aggregator
            aggregator.add_result(challenge, similarity)
        
        # Get final decision
        decision = aggregator.get_final_decision()
        
        assert decision["overall_decision"] in ["accept_h0", "reject_h0", "continue"]
        assert len(decision["per_challenge_scores"]) == len(challenges)
    
    def test_memory_constrained_scenario(self):
        """Test pipeline under memory constraints."""
        config = REVConfig(
            segment_size=16,  # Small segments
            buffer_size=1,    # Minimal buffer
            memory_limit_gb=0.5  # 500MB limit
        )
        pipeline = REVPipeline(config)
        
        # Mock model with memory tracking
        mock_model = Mock()
        mock_model.config = Mock(hidden_size=768, num_hidden_layers=12)
        
        with patch('src.executor.segment_runner.SegmentRunner.get_memory_usage') as mock_memory:
            # Simulate increasing memory usage
            mock_memory.side_effect = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
            
            # Process long input
            long_challenge = " ".join(["test"] * 200)
            
            result = pipeline.process_challenge(mock_model, long_challenge)
            
            # Should have used memory-efficient mode
            assert result["metadata"]["memory_efficient"] == True
            assert result["metadata"]["segments_processed"] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])