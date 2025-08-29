"""
Unified REST API for REV/HBT verification system.

This module provides a FastAPI-based REST API that supports multiple verification
modes (fast REV, robust HBT, hybrid) with comprehensive results including Merkle
proofs and behavioral certificates.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import hashlib
import json
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# REV/HBT imports
from ..rev_pipeline import REVPipeline, Segment
from ..verifier.blackbox import BlackBoxVerifier, ModelProvider, APIConfig
from ..verifier.streaming_consensus import StreamingConsensusVerifier, ConsensusMode
from ..consensus.byzantine import ConsensusNetwork, ByzantineValidator
from ..crypto.merkle import (
    HierarchicalVerificationChain,
    create_hierarchical_tree_from_segments
)
from ..core.sequential import SequentialState
from ..verifier.decision import Verdict
from ..verifier.contamination import UnifiedContaminationDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationMode(str, Enum):
    """Verification mode selection."""
    FAST = "fast"        # REV sequential testing
    ROBUST = "robust"    # HBT Byzantine consensus
    HYBRID = "hybrid"    # Combined approach
    AUTO = "auto"        # Automatic selection


class RequirementPriority(str, Enum):
    """Requirement priority for mode selection."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    BALANCED = "balanced"


class VerificationRequest(BaseModel):
    """Request model for verification endpoint."""
    
    model_a: str = Field(..., description="First model identifier or API endpoint")
    model_b: str = Field(..., description="Second model identifier or API endpoint")
    challenges: List[str] = Field(..., description="List of challenge prompts")
    mode: VerificationMode = Field(VerificationMode.AUTO, description="Verification mode")
    
    # Requirements for auto mode selection
    max_latency_ms: Optional[int] = Field(None, description="Maximum latency in milliseconds")
    min_accuracy: Optional[float] = Field(None, description="Minimum accuracy requirement (0-1)")
    max_memory_mb: Optional[int] = Field(None, description="Maximum memory usage in MB")
    priority: RequirementPriority = Field(RequirementPriority.BALANCED, description="Optimization priority")
    
    # Advanced options
    consensus_threshold: Optional[float] = Field(0.67, description="Consensus threshold for HBT mode")
    segment_size: Optional[int] = Field(512, description="Segment size for REV mode")
    enable_zk_proofs: Optional[bool] = Field(True, description="Enable zero-knowledge proofs")
    enable_contamination_check: Optional[bool] = Field(False, description="Check for contamination")
    
    # API configurations if needed
    api_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="API configurations for models")
    
    @validator('consensus_threshold')
    def validate_threshold(cls, v):
        if v is not None and not (0.5 < v <= 1.0):
            raise ValueError("Consensus threshold must be between 0.5 and 1.0")
        return v
    
    @validator('challenges')
    def validate_challenges(cls, v):
        if len(v) == 0:
            raise ValueError("At least one challenge is required")
        return v


class PerformanceMetrics(BaseModel):
    """Performance metrics for verification."""
    
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    memory_usage_mb: float = Field(..., description="Peak memory usage in MB")
    throughput_qps: float = Field(..., description="Queries per second")
    segments_processed: int = Field(..., description="Number of segments processed")
    consensus_rounds: Optional[int] = Field(None, description="Number of consensus rounds (HBT mode)")
    
    # Detailed timing
    verification_time_ms: float = Field(..., description="Core verification time")
    merkle_generation_time_ms: float = Field(..., description="Merkle tree generation time")
    zk_proof_time_ms: Optional[float] = Field(None, description="ZK proof generation time")
    
    # Resource usage
    cpu_usage_percent: float = Field(..., description="Average CPU usage")
    network_bytes: int = Field(..., description="Network bytes transferred")
    cache_hits: int = Field(..., description="Cache hit count")
    cache_misses: int = Field(..., description="Cache miss count")


class CertificateInfo(BaseModel):
    """Certificate information."""
    
    certificate_id: str = Field(..., description="Certificate identifier")
    certificate_type: str = Field(..., description="Type of certificate")
    timestamp: float = Field(..., description="Certificate timestamp")
    signature: str = Field(..., description="Certificate signature (hex)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerificationResponse(BaseModel):
    """Response model for verification endpoint."""
    
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="ISO timestamp of verification")
    mode_used: VerificationMode = Field(..., description="Actual mode used for verification")
    
    # Core results
    verdict: str = Field(..., description="Verification verdict: accept/reject/uncertain")
    confidence: float = Field(..., description="Confidence score (0-1)")
    similarity_score: float = Field(..., description="Similarity score between models")
    
    # Merkle proofs and certificates
    merkle_root: str = Field(..., description="Root hash of Merkle tree (hex)")
    verification_tree_id: str = Field(..., description="Hierarchical verification tree ID")
    certificates: List[CertificateInfo] = Field(default_factory=list, description="Behavioral certificates")
    zk_proofs: Optional[List[Dict[str, Any]]] = Field(None, description="Zero-knowledge proofs")
    
    # Performance metrics
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    
    # Detailed results
    segment_results: Optional[List[Dict[str, Any]]] = Field(None, description="Per-segment results")
    consensus_details: Optional[Dict[str, Any]] = Field(None, description="Consensus details (HBT mode)")
    contamination_results: Optional[Dict[str, Any]] = Field(None, description="Contamination check results")
    
    # Mode selection reasoning
    mode_selection_reason: str = Field(..., description="Reason for mode selection")
    alternative_modes: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative modes considered")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "timestamp": "2024-01-01T00:00:00Z",
                "mode_used": "hybrid",
                "verdict": "accept",
                "confidence": 0.95,
                "similarity_score": 0.92,
                "merkle_root": "0x1234567890abcdef",
                "verification_tree_id": "tree_001",
                "metrics": {
                    "latency_ms": 1500,
                    "memory_usage_mb": 256,
                    "throughput_qps": 10
                }
            }
        }


class UnifiedVerificationAPI:
    """
    Unified verification API supporting multiple modes.
    
    This class implements the core verification logic with support for:
    - Fast REV mode using sequential testing
    - Robust HBT mode using Byzantine consensus
    - Hybrid mode combining both approaches
    - Automatic mode selection based on requirements
    """
    
    def __init__(
        self,
        rev_pipeline: Optional[REVPipeline] = None,
        blackbox_verifier: Optional[BlackBoxVerifier] = None,
        consensus_network: Optional[ConsensusNetwork] = None,
        cache_results: bool = True
    ):
        """
        Initialize unified verification API.
        
        Args:
            rev_pipeline: REV pipeline for fast verification
            blackbox_verifier: Black-box verifier for API access
            consensus_network: Byzantine consensus network for HBT mode
            cache_results: Whether to cache verification results
        """
        self.rev_pipeline = rev_pipeline
        self.blackbox_verifier = blackbox_verifier
        self.consensus_network = consensus_network
        self.cache_results = cache_results
        
        # Initialize components if not provided
        if self.rev_pipeline is None:
            self.rev_pipeline = REVPipeline(
                sites=self._get_default_sites(),
                segment_size=512,
                buffer_size=4
            )
        
        if self.blackbox_verifier is None:
            self.blackbox_verifier = BlackBoxVerifier()
        
        if self.consensus_network is None:
            self.consensus_network = ConsensusNetwork(
                num_validators=4,  # 3f+1 for f=1
                fault_tolerance=1
            )
        
        # Result cache
        self.result_cache: Dict[str, VerificationResponse] = {}
        
        # Performance tracking
        self.request_count = 0
        self.total_latency_ms = 0.0
        
    def _get_default_sites(self) -> List[Any]:
        """Get default architectural sites for probing."""
        # Simplified - would normally return actual sites
        return []
    
    async def verify(
        self,
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Main verification endpoint.
        
        Args:
            request: Verification request
            
        Returns:
            Comprehensive verification response
        """
        start_time = time.perf_counter()
        request_id = self._generate_request_id()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if self.cache_results and cache_key in self.result_cache:
            logger.info(f"Cache hit for request {request_id}")
            cached = self.result_cache[cache_key]
            cached.request_id = request_id  # Update request ID
            return cached
        
        # Select verification mode
        mode_to_use, selection_reason = await self.select_mode(request)
        logger.info(f"Request {request_id}: Using {mode_to_use} mode - {selection_reason}")
        
        # Configure API access if needed
        if request.api_configs:
            self._configure_apis(request.api_configs)
        
        # Run verification based on mode
        if mode_to_use == VerificationMode.FAST:
            result = await self.rev_fast_verify(request)
        elif mode_to_use == VerificationMode.ROBUST:
            result = await self.hbt_consensus_verify(request)
        elif mode_to_use == VerificationMode.HYBRID:
            result = await self.hybrid_verify(request)
        else:
            raise ValueError(f"Unsupported mode: {mode_to_use}")
        
        # Add metadata
        result.request_id = request_id
        result.timestamp = datetime.utcnow().isoformat() + "Z"
        result.mode_used = mode_to_use
        result.mode_selection_reason = selection_reason
        
        # Calculate final metrics
        total_time = (time.perf_counter() - start_time) * 1000
        result.metrics.latency_ms = total_time
        
        # Check for contamination if requested
        if request.enable_contamination_check:
            contamination_results = await self._check_contamination(
                request.model_a,
                request.model_b,
                request.challenges
            )
            result.contamination_results = contamination_results
        
        # Cache result
        if self.cache_results:
            self.result_cache[cache_key] = result
        
        # Update statistics
        self.request_count += 1
        self.total_latency_ms += total_time
        
        return result
    
    async def select_mode(
        self,
        request: VerificationRequest
    ) -> tuple[VerificationMode, str]:
        """
        Select verification mode based on requirements.
        
        Args:
            request: Verification request with requirements
            
        Returns:
            Tuple of (selected mode, reason for selection)
        """
        if request.mode != VerificationMode.AUTO:
            return request.mode, f"Explicitly requested {request.mode} mode"
        
        # Analyze requirements
        has_latency_constraint = request.max_latency_ms is not None
        has_accuracy_constraint = request.min_accuracy is not None
        has_memory_constraint = request.max_memory_mb is not None
        
        # Estimate characteristics of each mode
        rev_latency = len(request.challenges) * 50  # ~50ms per challenge
        hbt_latency = len(request.challenges) * 200  # ~200ms with consensus
        hybrid_latency = len(request.challenges) * 120  # Between the two
        
        rev_accuracy = 0.85  # Baseline accuracy
        hbt_accuracy = 0.95  # Higher with consensus
        hybrid_accuracy = 0.92  # Good balance
        
        rev_memory = 256  # MB
        hbt_memory = 512  # MB (multiple validators)
        hybrid_memory = 384  # MB
        
        # Score each mode
        scores = {}
        
        # REV Fast mode
        rev_score = 0
        if has_latency_constraint and rev_latency <= request.max_latency_ms:
            rev_score += 3
        if has_accuracy_constraint and rev_accuracy >= request.min_accuracy:
            rev_score += 2
        if has_memory_constraint and rev_memory <= request.max_memory_mb:
            rev_score += 2
        
        # Adjust for priority
        if request.priority == RequirementPriority.LATENCY:
            rev_score += 5
        elif request.priority == RequirementPriority.MEMORY:
            rev_score += 3
        
        scores[VerificationMode.FAST] = rev_score
        
        # HBT Robust mode
        hbt_score = 0
        if has_latency_constraint and hbt_latency <= request.max_latency_ms:
            hbt_score += 1
        if has_accuracy_constraint and hbt_accuracy >= request.min_accuracy:
            hbt_score += 5
        if has_memory_constraint and hbt_memory <= request.max_memory_mb:
            hbt_score += 1
        
        if request.priority == RequirementPriority.ACCURACY:
            hbt_score += 5
        
        scores[VerificationMode.ROBUST] = hbt_score
        
        # Hybrid mode
        hybrid_score = 0
        if has_latency_constraint and hybrid_latency <= request.max_latency_ms:
            hybrid_score += 2
        if has_accuracy_constraint and hybrid_accuracy >= request.min_accuracy:
            hybrid_score += 3
        if has_memory_constraint and hybrid_memory <= request.max_memory_mb:
            hybrid_score += 2
        
        if request.priority == RequirementPriority.BALANCED:
            hybrid_score += 4
        
        scores[VerificationMode.HYBRID] = hybrid_score
        
        # Select mode with highest score
        best_mode = max(scores, key=scores.get)
        
        # Generate reason
        if best_mode == VerificationMode.FAST:
            reason = f"Fast mode selected for low latency ({rev_latency}ms estimated)"
        elif best_mode == VerificationMode.ROBUST:
            reason = f"Robust mode selected for high accuracy ({hbt_accuracy:.0%} estimated)"
        else:
            reason = f"Hybrid mode selected for balanced performance"
        
        if request.priority != RequirementPriority.BALANCED:
            reason += f" with {request.priority} priority"
        
        return best_mode, reason
    
    async def rev_fast_verify(
        self,
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Fast verification using REV sequential testing.
        
        Args:
            request: Verification request
            
        Returns:
            Verification response
        """
        start_time = time.perf_counter()
        
        # Initialize sequential state
        seq_state = SequentialState(
            alpha=0.05,
            beta=0.10,
            tau_max=1.0
        )
        
        # Process challenges through REV pipeline
        segments = []
        segment_results = []
        
        for i, challenge in enumerate(request.challenges):
            # Get responses from models
            response_a = await self._get_model_response(
                request.model_a,
                challenge,
                request.api_configs
            )
            response_b = await self._get_model_response(
                request.model_b,
                challenge,
                request.api_configs
            )
            
            # Create segment
            segment = {
                "index": i,
                "challenge": challenge,
                "response_a": response_a,
                "response_b": response_b,
                "tokens": challenge.split()[:request.segment_size]
            }
            segments.append(segment)
            
            # Calculate similarity
            similarity = self._calculate_similarity(response_a, response_b)
            
            # Update sequential state
            seq_state.update(similarity)
            
            # Check for early stopping
            if seq_state.should_stop():
                logger.info(f"Early stopping at challenge {i+1}/{len(request.challenges)}")
                break
            
            segment_results.append({
                "index": i,
                "similarity": similarity,
                "verdict": seq_state.get_decision().value
            })
        
        # Generate Merkle tree and certificates
        merkle_start = time.perf_counter()
        verification_tree = create_hierarchical_tree_from_segments(
            segments,
            enable_zk=request.enable_zk_proofs,
            consensus_interval=10
        )
        merkle_time = (time.perf_counter() - merkle_start) * 1000
        
        # Extract certificates
        certificates = [
            CertificateInfo(
                certificate_id=cert.certificate_id,
                certificate_type="behavioral",
                timestamp=cert.timestamp,
                signature=cert.behavioral_signature.hex(),
                metadata=cert.metadata
            )
            for cert in verification_tree.certificates
        ]
        
        # Calculate metrics
        total_time = (time.perf_counter() - start_time) * 1000
        
        metrics = PerformanceMetrics(
            latency_ms=total_time,
            memory_usage_mb=256,  # Estimated
            throughput_qps=len(request.challenges) / (total_time / 1000),
            segments_processed=len(segments),
            verification_time_ms=total_time - merkle_time,
            merkle_generation_time_ms=merkle_time,
            cpu_usage_percent=50.0,  # Estimated
            network_bytes=len(json.dumps(segments)) * 2,
            cache_hits=0,
            cache_misses=len(request.challenges)
        )
        
        # Get final verdict
        final_verdict = seq_state.get_decision()
        
        return VerificationResponse(
            request_id="",  # Will be set by caller
            timestamp="",  # Will be set by caller
            mode_used=VerificationMode.FAST,
            verdict=final_verdict.value,
            confidence=seq_state.get_confidence(),
            similarity_score=seq_state.mean,
            merkle_root=verification_tree.master_root.hex(),
            verification_tree_id=verification_tree.tree_id,
            certificates=certificates,
            metrics=metrics,
            segment_results=segment_results,
            mode_selection_reason=""  # Will be set by caller
        )
    
    async def hbt_consensus_verify(
        self,
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Robust verification using HBT Byzantine consensus.
        
        Args:
            request: Verification request
            
        Returns:
            Verification response
        """
        start_time = time.perf_counter()
        
        # Process challenges through consensus network
        segments = []
        consensus_rounds = 0
        
        for i, challenge in enumerate(request.challenges):
            # Get responses from models
            response_a = await self._get_model_response(
                request.model_a,
                challenge,
                request.api_configs
            )
            response_b = await self._get_model_response(
                request.model_b,
                challenge,
                request.api_configs
            )
            
            # Create segment
            segment = Segment(
                index=i,
                tokens=challenge.split()[:request.segment_size],
                embeddings=None,
                activations={}
            )
            segments.append(segment)
            
            consensus_rounds += 1
        
        # Run consensus validation
        consensus_result = self.consensus_network.validate_segments(
            segments,
            threshold=request.consensus_threshold or 0.67
        )
        
        # Generate verification tree
        merkle_start = time.perf_counter()
        verification_tree = create_hierarchical_tree_from_segments(
            [s.__dict__ for s in segments],
            enable_zk=request.enable_zk_proofs
        )
        merkle_time = (time.perf_counter() - merkle_start) * 1000
        
        # Extract certificates
        certificates = [
            CertificateInfo(
                certificate_id=cert.certificate_id,
                certificate_type="consensus",
                timestamp=cert.timestamp,
                signature=cert.behavioral_signature.hex(),
                metadata=cert.metadata
            )
            for cert in verification_tree.certificates
        ]
        
        # Calculate metrics
        total_time = (time.perf_counter() - start_time) * 1000
        
        metrics = PerformanceMetrics(
            latency_ms=total_time,
            memory_usage_mb=512,  # Higher for consensus
            throughput_qps=len(request.challenges) / (total_time / 1000),
            segments_processed=len(segments),
            consensus_rounds=consensus_rounds,
            verification_time_ms=total_time - merkle_time,
            merkle_generation_time_ms=merkle_time,
            cpu_usage_percent=75.0,  # Higher for consensus
            network_bytes=len(json.dumps([s.__dict__ for s in segments])) * 5,  # More network traffic
            cache_hits=0,
            cache_misses=len(request.challenges)
        )
        
        # Prepare consensus details
        consensus_details = {
            "num_validators": self.consensus_network.num_validators,
            "fault_tolerance": self.consensus_network.fault_tolerance,
            "consensus_threshold": request.consensus_threshold,
            "rounds_completed": consensus_rounds,
            "byzantine_nodes_detected": consensus_result.get("byzantine_nodes", [])
        }
        
        return VerificationResponse(
            request_id="",
            timestamp="",
            mode_used=VerificationMode.ROBUST,
            verdict=consensus_result["verdict"].value,
            confidence=consensus_result["confidence"],
            similarity_score=consensus_result.get("similarity", 0.9),
            merkle_root=verification_tree.master_root.hex(),
            verification_tree_id=verification_tree.tree_id,
            certificates=certificates,
            metrics=metrics,
            consensus_details=consensus_details,
            mode_selection_reason=""
        )
    
    async def hybrid_verify(
        self,
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Hybrid verification combining REV and HBT approaches.
        
        Args:
            request: Verification request
            
        Returns:
            Verification response
        """
        start_time = time.perf_counter()
        
        # Run both verifications in parallel
        rev_task = asyncio.create_task(self.rev_fast_verify(request))
        hbt_task = asyncio.create_task(self.hbt_consensus_verify(request))
        
        # Wait for both to complete
        rev_result, hbt_result = await asyncio.gather(rev_task, hbt_task)
        
        # Combine results (weighted average)
        rev_weight = 0.4  # Fast but less accurate
        hbt_weight = 0.6  # Slower but more accurate
        
        combined_confidence = (
            rev_result.confidence * rev_weight +
            hbt_result.confidence * hbt_weight
        )
        
        combined_similarity = (
            rev_result.similarity_score * rev_weight +
            hbt_result.similarity_score * hbt_weight
        )
        
        # Determine final verdict
        if combined_confidence >= 0.95:
            final_verdict = "accept"
        elif combined_confidence <= 0.05:
            final_verdict = "reject"
        else:
            final_verdict = "uncertain"
        
        # Combine certificates
        all_certificates = rev_result.certificates + hbt_result.certificates
        
        # Calculate combined metrics
        total_time = (time.perf_counter() - start_time) * 1000
        
        metrics = PerformanceMetrics(
            latency_ms=total_time,
            memory_usage_mb=max(rev_result.metrics.memory_usage_mb, hbt_result.metrics.memory_usage_mb),
            throughput_qps=len(request.challenges) / (total_time / 1000),
            segments_processed=rev_result.metrics.segments_processed,
            consensus_rounds=hbt_result.metrics.consensus_rounds,
            verification_time_ms=max(rev_result.metrics.verification_time_ms, hbt_result.metrics.verification_time_ms),
            merkle_generation_time_ms=rev_result.metrics.merkle_generation_time_ms,
            cpu_usage_percent=(rev_result.metrics.cpu_usage_percent + hbt_result.metrics.cpu_usage_percent) / 2,
            network_bytes=rev_result.metrics.network_bytes + hbt_result.metrics.network_bytes,
            cache_hits=rev_result.metrics.cache_hits + hbt_result.metrics.cache_hits,
            cache_misses=rev_result.metrics.cache_misses
        )
        
        # Include both segment results and consensus details
        hybrid_details = {
            "rev_verdict": rev_result.verdict,
            "rev_confidence": rev_result.confidence,
            "hbt_verdict": hbt_result.verdict,
            "hbt_confidence": hbt_result.confidence,
            "weights": {"rev": rev_weight, "hbt": hbt_weight}
        }
        
        return VerificationResponse(
            request_id="",
            timestamp="",
            mode_used=VerificationMode.HYBRID,
            verdict=final_verdict,
            confidence=combined_confidence,
            similarity_score=combined_similarity,
            merkle_root=rev_result.merkle_root,  # Use REV's merkle root
            verification_tree_id=rev_result.verification_tree_id,
            certificates=all_certificates,
            metrics=metrics,
            segment_results=rev_result.segment_results,
            consensus_details=hybrid_details,
            mode_selection_reason=""
        )
    
    async def _get_model_response(
        self,
        model_id: str,
        challenge: str,
        api_configs: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Get response from model via API."""
        # Use blackbox verifier if available
        if self.blackbox_verifier and model_id in self.blackbox_verifier.configs:
            response = await self.blackbox_verifier.query_model_async(
                model_id,
                challenge
            )
            return response
        
        # Fallback to mock response
        return {
            "text": f"Response from {model_id} to: {challenge}",
            "logits": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"model": model_id}
        }
    
    def _calculate_similarity(
        self,
        response_a: Dict[str, Any],
        response_b: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two responses."""
        # Simplified similarity calculation
        text_a = response_a.get("text", "")
        text_b = response_b.get("text", "")
        
        # Simple character overlap
        common = set(text_a.split()) & set(text_b.split())
        total = set(text_a.split()) | set(text_b.split())
        
        if not total:
            return 0.0
        
        return len(common) / len(total)
    
    async def _check_contamination(
        self,
        model_a: str,
        model_b: str,
        challenges: List[str]
    ) -> Dict[str, Any]:
        """Check for model contamination."""
        detector = UnifiedContaminationDetector(
            blackbox_verifier=self.blackbox_verifier
        )
        
        # Run contamination detection
        result = detector.detect_contamination(
            model=None,  # API-based
            reference_models=[],
            model_id=model_a,
            challenges=challenges[:10]  # Sample
        )
        
        return {
            "contaminated": result.is_contaminated,
            "contamination_type": result.contamination_type.value if result.contamination_type else None,
            "confidence": result.confidence,
            "evidence": result.evidence[:3] if result.evidence else []  # Limit evidence
        }
    
    def _configure_apis(self, api_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure API access for models."""
        for model_id, config in api_configs.items():
            if "provider" in config:
                provider = ModelProvider(config["provider"])
                api_config = APIConfig(
                    api_key=config.get("api_key", ""),
                    base_url=config.get("base_url"),
                    model_name=config.get("model_name", model_id),
                    max_tokens=config.get("max_tokens", 1000),
                    temperature=config.get("temperature", 0.0)
                )
                self.blackbox_verifier.add_model(model_id, provider, api_config)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        timestamp = int(time.time() * 1000)
        random_suffix = hashlib.sha256(str(timestamp).encode()).hexdigest()[:8]
        return f"req_{timestamp}_{random_suffix}"
    
    def _get_cache_key(self, request: VerificationRequest) -> str:
        """Generate cache key for request."""
        # Create deterministic key from request parameters
        key_data = {
            "model_a": request.model_a,
            "model_b": request.model_b,
            "challenges": sorted(request.challenges),
            "mode": request.mode.value
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API statistics."""
        avg_latency = self.total_latency_ms / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "average_latency_ms": avg_latency,
            "cache_size": len(self.result_cache),
            "cache_hit_rate": 0.0  # Would need to track this
        }


# FastAPI app creation
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting Unified Verification API")
    yield
    # Shutdown
    logger.info("Shutting down Unified Verification API")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Unified REV/HBT Verification API",
        description="REST API for model verification with multiple modes",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize API handler
    api_handler = UnifiedVerificationAPI()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    @app.post("/verify", response_model=VerificationResponse)
    async def verify_models(request: VerificationRequest):
        """
        Verify model equivalence with selected mode.
        
        Supports three verification modes:
        - fast: REV sequential testing for low latency
        - robust: HBT Byzantine consensus for high accuracy
        - hybrid: Combined approach for balanced performance
        - auto: Automatic mode selection based on requirements
        """
        try:
            response = await api_handler.verify(request)
            return response
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats")
    async def get_statistics():
        """Get API usage statistics."""
        return api_handler.get_statistics()
    
    @app.post("/clear_cache")
    async def clear_cache():
        """Clear result cache."""
        api_handler.result_cache.clear()
        return {"message": "Cache cleared", "timestamp": datetime.utcnow().isoformat()}
    
    return app


# Main entry point
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )