"""
Tests for the Unified Verification API.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.api.unified_api import (
    UnifiedVerificationAPI,
    VerificationRequest,
    VerificationResponse,
    VerificationMode,
    RequirementPriority,
    create_app
)


class TestUnifiedVerificationAPI:
    """Test suite for unified verification API."""
    
    @pytest.fixture
    def api_handler(self):
        """Create API handler instance."""
        return UnifiedVerificationAPI(cache_results=False)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample verification request."""
        return VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["What is 2+2?", "Explain gravity"],
            mode=VerificationMode.AUTO,
            max_latency_ms=5000,
            min_accuracy=0.9,
            max_memory_mb=1024,
            priority=RequirementPriority.BALANCED
        )
    
    @pytest.mark.asyncio
    async def test_mode_selection_auto(self, api_handler, sample_request):
        """Test automatic mode selection."""
        mode, reason = await api_handler.select_mode(sample_request)
        
        assert mode in [VerificationMode.FAST, VerificationMode.ROBUST, VerificationMode.HYBRID]
        assert len(reason) > 0
    
    @pytest.mark.asyncio
    async def test_mode_selection_explicit(self, api_handler):
        """Test explicit mode selection."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test"],
            mode=VerificationMode.FAST
        )
        
        mode, reason = await api_handler.select_mode(request)
        
        assert mode == VerificationMode.FAST
        assert "Explicitly requested" in reason
    
    @pytest.mark.asyncio
    async def test_mode_selection_latency_priority(self, api_handler):
        """Test mode selection with latency priority."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test"] * 10,
            mode=VerificationMode.AUTO,
            max_latency_ms=100,
            priority=RequirementPriority.LATENCY
        )
        
        mode, reason = await api_handler.select_mode(request)
        
        # Should prefer FAST mode for latency
        assert mode == VerificationMode.FAST
        assert "latency" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_mode_selection_accuracy_priority(self, api_handler):
        """Test mode selection with accuracy priority."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test"],
            mode=VerificationMode.AUTO,
            min_accuracy=0.95,
            priority=RequirementPriority.ACCURACY
        )
        
        mode, reason = await api_handler.select_mode(request)
        
        # Should prefer ROBUST mode for accuracy
        assert mode == VerificationMode.ROBUST
        assert "accuracy" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_rev_fast_verify(self, api_handler):
        """Test REV fast verification mode."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["What is 2+2?"],
            mode=VerificationMode.FAST
        )
        
        # Mock model responses
        with patch.object(api_handler, '_get_model_response', new_callable=AsyncMock) as mock_response:
            mock_response.return_value = {
                "text": "The answer is 4",
                "logits": [0.1, 0.2, 0.3, 0.4]
            }
            
            response = await api_handler.rev_fast_verify(request)
        
        assert isinstance(response, VerificationResponse)
        assert response.mode_used == VerificationMode.FAST
        assert response.verdict in ["accept", "reject", "uncertain"]
        assert 0 <= response.confidence <= 1
        assert response.merkle_root
        assert response.verification_tree_id
        assert response.metrics.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_hbt_consensus_verify(self, api_handler):
        """Test HBT consensus verification mode."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["What is 2+2?"],
            mode=VerificationMode.ROBUST,
            consensus_threshold=0.67
        )
        
        # Mock model responses
        with patch.object(api_handler, '_get_model_response', new_callable=AsyncMock) as mock_response:
            mock_response.return_value = {
                "text": "The answer is 4",
                "logits": [0.1, 0.2, 0.3, 0.4]
            }
            
            response = await api_handler.hbt_consensus_verify(request)
        
        assert isinstance(response, VerificationResponse)
        assert response.mode_used == VerificationMode.ROBUST
        assert response.verdict in ["accept", "reject", "uncertain"]
        assert response.consensus_details is not None
        assert "num_validators" in response.consensus_details
        assert response.metrics.consensus_rounds is not None
    
    @pytest.mark.asyncio
    async def test_hybrid_verify(self, api_handler):
        """Test hybrid verification mode."""
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["What is 2+2?"],
            mode=VerificationMode.HYBRID
        )
        
        # Mock model responses
        with patch.object(api_handler, '_get_model_response', new_callable=AsyncMock) as mock_response:
            mock_response.return_value = {
                "text": "The answer is 4",
                "logits": [0.1, 0.2, 0.3, 0.4]
            }
            
            response = await api_handler.hybrid_verify(request)
        
        assert isinstance(response, VerificationResponse)
        assert response.mode_used == VerificationMode.HYBRID
        assert response.consensus_details is not None
        assert "rev_verdict" in response.consensus_details
        assert "hbt_verdict" in response.consensus_details
        assert "weights" in response.consensus_details
    
    @pytest.mark.asyncio
    async def test_verify_with_caching(self):
        """Test verification with result caching."""
        api_handler = UnifiedVerificationAPI(cache_results=True)
        
        request = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test"],
            mode=VerificationMode.FAST
        )
        
        with patch.object(api_handler, '_get_model_response', new_callable=AsyncMock) as mock_response:
            mock_response.return_value = {"text": "response", "logits": [0.1]}
            
            # First call
            response1 = await api_handler.verify(request)
            call_count1 = mock_response.call_count
            
            # Second call (should use cache)
            response2 = await api_handler.verify(request)
            call_count2 = mock_response.call_count
        
        assert call_count1 > 0
        assert call_count2 == call_count1  # No additional calls
        assert response1.verdict == response2.verdict
    
    def test_calculate_similarity(self, api_handler):
        """Test similarity calculation."""
        response_a = {"text": "The quick brown fox"}
        response_b = {"text": "The quick red fox"}
        
        similarity = api_handler._calculate_similarity(response_a, response_b)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be similar
    
    def test_calculate_similarity_identical(self, api_handler):
        """Test similarity calculation for identical responses."""
        response = {"text": "The answer is 42"}
        
        similarity = api_handler._calculate_similarity(response, response)
        
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self, api_handler):
        """Test similarity calculation for different responses."""
        response_a = {"text": "Yes"}
        response_b = {"text": "No"}
        
        similarity = api_handler._calculate_similarity(response_a, response_b)
        
        assert similarity == 0.0
    
    def test_generate_request_id(self, api_handler):
        """Test request ID generation."""
        id1 = api_handler._generate_request_id()
        id2 = api_handler._generate_request_id()
        
        assert id1 != id2
        assert id1.startswith("req_")
        assert len(id1.split("_")) == 3
    
    def test_get_cache_key(self, api_handler):
        """Test cache key generation."""
        request1 = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test1", "test2"],
            mode=VerificationMode.FAST
        )
        
        request2 = VerificationRequest(
            model_a="model-a",
            model_b="model-b",
            challenges=["test2", "test1"],  # Different order
            mode=VerificationMode.FAST
        )
        
        key1 = api_handler._get_cache_key(request1)
        key2 = api_handler._get_cache_key(request2)
        
        # Should be same despite different order
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex
    
    def test_get_statistics(self, api_handler):
        """Test statistics retrieval."""
        api_handler.request_count = 10
        api_handler.total_latency_ms = 5000
        
        stats = api_handler.get_statistics()
        
        assert stats["total_requests"] == 10
        assert stats["average_latency_ms"] == 500
        assert "cache_size" in stats


class TestFastAPIApp:
    """Test FastAPI application endpoints."""
    
    def test_create_app(self):
        """Test app creation."""
        app = create_app()
        
        assert app.title == "Unified REV/HBT Verification API"
        assert app.version == "1.0.0"
        
        # Check routes are registered
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/verify" in routes
        assert "/stats" in routes
        assert "/clear_cache" in routes
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_verify_endpoint_validation(self):
        """Test verify endpoint validation."""
        from fastapi.testclient import TestClient
        
        app = create_app()
        client = TestClient(app)
        
        # Invalid request (missing required fields)
        response = client.post("/verify", json={})
        assert response.status_code == 422
        
        # Invalid request (empty challenges)
        response = client.post("/verify", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "challenges": []
        })
        assert response.status_code == 422
        
        # Invalid consensus threshold
        response = client.post("/verify", json={
            "model_a": "model-a",
            "model_b": "model-b",
            "challenges": ["test"],
            "consensus_threshold": 1.5
        })
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])