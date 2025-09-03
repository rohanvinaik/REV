"""
Integration Tests for REV Production System
Tests API endpoints, error recovery, and system integration
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.error_handling import (
    REVException, CircuitBreaker, 
    GracefulDegradation, ErrorRecoveryManager
)
from src.utils.logging_config import LoggingConfig, MetricsCollector
from src.utils.reproducibility import SeedManager, ExperimentTracker, CheckpointManager
from src.utils.run_rev_recovery import PipelineRecovery, EnhancedREVPipeline


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        breaker = CircuitBreaker()
        
        def failing_function():
            raise Exception("Test failure")
        
        # Should trip after threshold
        for i in range(10):
            try:
                breaker.call(failing_function)
            except:
                pass
        
        # Should be open
        assert breaker.state.value == "open"
        
        # Should reject calls
        with pytest.raises(REVException):
            breaker.call(failing_function)
            
    def test_graceful_degradation(self):
        """Test graceful degradation"""
        degradation = GracefulDegradation()
        
        # Register feature
        feature = degradation.register_feature("test_feature")
        assert feature.is_enabled()
        
        # Degrade feature
        degradation.degrade_feature("test_feature", Exception("Test"))
        assert degradation.is_degraded("test_feature")
        
        # Restore feature
        degradation.restore_feature("test_feature")
        assert not degradation.is_degraded("test_feature")
        
    def test_memory_recovery(self):
        """Test memory overflow recovery"""
        recovery = PipelineRecovery()
        
        # Test memory detection
        is_overflow = recovery.detect_memory_overflow()
        assert isinstance(is_overflow, bool)
        
        # Test segment size adjustment
        new_size = recovery.adjust_segment_size(100, 0.95)
        assert new_size < 100  # Should reduce
        
        new_size = recovery.adjust_segment_size(100, 0.3)
        assert new_size > 100  # Should increase
        
    def test_checkpoint_recovery(self):
        """Test checkpoint save/load"""
        recovery = PipelineRecovery(checkpoint_dir="/tmp/test_checkpoints")
        
        # Save checkpoint
        state = {"test": "data", "array": [1, 2, 3]}
        checkpoint_path = recovery.save_checkpoint(
            model_path="test_model",
            stage="test_stage",
            state=state,
            metrics={"accuracy": 0.95}
        )
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded = recovery.load_checkpoint(checkpoint_path)
        assert loaded is not None
        assert loaded["state"] == state
        assert loaded["stage"] == "test_stage"


# ============================================================================
# Logging and Monitoring Tests
# ============================================================================

class TestLoggingMonitoring:
    """Test logging and monitoring integration"""
    
    def test_structured_logging(self):
        """Test JSON structured logging"""
        config = LoggingConfig(json_output=True)
        config.setup()
        
        logger = config.get_logger("test")
        logger.info("Test message", fields={"key": "value"})
        
        # Logger should work without errors
        assert True
        
    def test_performance_monitor(self):
        """Test performance monitoring"""
        from src.utils.logging_config import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Measure operation
        with monitor.measure("test_op", category="test"):
            time.sleep(0.1)
        
        # Get statistics
        stats = monitor.get_statistics("test_op")
        assert stats["count"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration"] > 0.09


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Test reproducibility features"""
    
    def test_seed_management(self):
        """Test seed manager"""
        seed_mgr = SeedManager(seed=42)
        seed_mgr.set_all_seeds()
        
        # Generate random numbers
        import random
        nums1 = [random.random() for _ in range(10)]
        
        # Reset seeds
        seed_mgr2 = SeedManager(seed=42)
        seed_mgr2.set_all_seeds()
        
        # Should generate same numbers
        nums2 = [random.random() for _ in range(10)]
        assert nums1 == nums2
        
    def test_experiment_tracking(self):
        """Test experiment tracker"""
        tracker = ExperimentTracker(
            backend="local",
            experiment_name="test_exp"
        )
        
        # Log parameters
        tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
        
        # Log metrics
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=1)
        
        # Check local files created
        assert (Path("experiments") / "test_exp").exists()
        
        tracker.finish()
        
    def test_checkpoint_manager(self):
        """Test checkpoint management"""
        mgr = CheckpointManager(
            checkpoint_dir="/tmp/test_ckpt",
            max_checkpoints=3,
            save_best=True,
            metric_name="loss",
            metric_mode="min"
        )
        
        # Save checkpoints
        for i in range(5):
            mgr.save(
                step=i,
                epoch=i,
                state={"model": f"state_{i}"},
                metrics={"loss": 0.5 - i * 0.1}
            )
        
        # Should keep only 3 checkpoints
        checkpoints = list(Path("/tmp/test_ckpt").glob("checkpoint_*.pt"))
        assert len(checkpoints) <= 3
        
        # Load best should get lowest loss
        best = mgr.load_best()
        assert best is not None
        assert best.metrics["loss"] == 0.1


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])