"""
Error Recovery Enhancements for run_rev.py
This module adds comprehensive error recovery capabilities to the main pipeline
"""

import os
import gc
import json
import pickle
import psutil
import torch
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import traceback
import time
import logging

from src.utils.error_handling import (
    REVException, ModelLoadingError, InferenceError, MemoryError,
    NetworkError, ResourceError, retry_with_backoff, CircuitBreaker,
    degradation_manager, ErrorRecoveryManager, RecoveryStrategy
)
from src.utils.logging_config import get_logger
from src.utils.reproducibility import CheckpointManager, Checkpoint

logger = get_logger(__name__)


class PipelineRecovery:
    """Enhanced recovery capabilities for REV pipeline"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.circuit_breakers = {}
        self.recovery_manager = ErrorRecoveryManager()
        self._setup_recovery_strategies()
        
    def _setup_recovery_strategies(self):
        """Configure recovery strategies for different error types"""
        
        # Memory recovery
        def memory_recovery(error: Exception, context: Dict[str, Any]) -> Any:
            logger.warning("Attempting memory recovery")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Reduce batch size or segment size
            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                logger.info(f"Reduced batch size to {context['batch_size']}")
            
            if 'segment_size' in context:
                context['segment_size'] = max(1, context['segment_size'] // 2)
                logger.info(f"Reduced segment size to {context['segment_size']}")
            
            return context
        
        # Network recovery
        def network_recovery(error: Exception, context: Dict[str, Any]) -> Any:
            logger.warning("Attempting network recovery")
            
            # Switch to local cache if available
            if 'use_cache' in context:
                context['use_cache'] = True
                logger.info("Switched to local cache")
            
            # Increase timeout
            if 'timeout' in context:
                context['timeout'] = context['timeout'] * 2
                logger.info(f"Increased timeout to {context['timeout']}s")
            
            return context
        
        # GPU recovery
        def gpu_recovery(error: Exception, context: Dict[str, Any]) -> Any:
            logger.warning("GPU error detected, falling back to CPU")
            
            # Switch to CPU
            context['device'] = 'cpu'
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return context
        
        self.recovery_manager.register_handler(
            RecoveryStrategy.DEGRADE, memory_recovery
        )
        self.recovery_manager.register_handler(
            RecoveryStrategy.RETRY, network_recovery
        )
        self.recovery_manager.register_handler(
            RecoveryStrategy.FALLBACK, gpu_recovery
        )
        
    def save_checkpoint(
        self,
        model_path: str,
        stage: str,
        state: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Path:
        """Save pipeline checkpoint"""
        checkpoint_data = {
            'model_path': model_path,
            'stage': stage,
            'state': state,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{stage}_{int(time.time())}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load pipeline checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if not checkpoints:
                return None
            checkpoint_path = checkpoints[-1]
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def detect_memory_overflow(self) -> bool:
        """Detect if system is running out of memory"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_percent = gpu_memory * 100
            except:
                gpu_percent = 0
        else:
            gpu_percent = 0
        
        # Check thresholds
        if memory_percent > 90:
            logger.warning(f"High system memory usage: {memory_percent}%")
            return True
        
        if gpu_percent > 95:
            logger.warning(f"High GPU memory usage: {gpu_percent}%")
            return True
        
        return False
        
    def adjust_segment_size(self, current_size: int, memory_usage: float) -> int:
        """Dynamically adjust segment size based on memory usage"""
        if memory_usage > 0.9:
            # Reduce by 50%
            new_size = max(1, current_size // 2)
        elif memory_usage > 0.8:
            # Reduce by 25%
            new_size = max(1, int(current_size * 0.75))
        elif memory_usage < 0.5:
            # Increase by 25%
            new_size = int(current_size * 1.25)
        else:
            new_size = current_size
        
        if new_size != current_size:
            logger.info(f"Adjusted segment size: {current_size} -> {new_size}")
        
        return new_size
        
    def create_local_cache(self, model_path: str) -> Optional[Path]:
        """Create local cache for network resources"""
        cache_dir = self.checkpoint_dir / "cache" / Path(model_path).name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_info_file = cache_dir / "cache_info.json"
        
        if cache_info_file.exists():
            with open(cache_info_file, 'r') as f:
                cache_info = json.load(f)
            logger.info(f"Using existing cache: {cache_dir}")
        else:
            cache_info = {
                'created': datetime.utcnow().isoformat(),
                'model_path': model_path,
                'files': []
            }
            with open(cache_info_file, 'w') as f:
                json.dump(cache_info, f)
            logger.info(f"Created new cache: {cache_dir}")
        
        return cache_dir


class EnhancedREVPipeline:
    """Enhanced REV pipeline with error recovery"""
    
    def __init__(self, base_pipeline, recovery: PipelineRecovery):
        self.base_pipeline = base_pipeline
        self.recovery = recovery
        self.max_retries = 3
        self.retry_delay = 1.0
        
    @retry_with_backoff(max_attempts=3)
    def process_model_with_recovery(
        self,
        model_path: str,
        resume_from_checkpoint: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process model with comprehensive error recovery"""
        
        # Check for existing checkpoint
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = self.recovery.load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint['stage']}")
        
        # Initialize result
        result = checkpoint['state'] if checkpoint else {'stages': {}}
        
        stages = [
            ('initialization', self._initialize_stage),
            ('loading', self._loading_stage),
            ('processing', self._processing_stage),
            ('analysis', self._analysis_stage),
            ('finalization', self._finalization_stage)
        ]
        
        # Resume from checkpoint stage if available
        start_stage = 0
        if checkpoint:
            for i, (stage_name, _) in enumerate(stages):
                if stage_name == checkpoint['stage']:
                    start_stage = i + 1
                    break
        
        # Process stages
        for stage_name, stage_func in stages[start_stage:]:
            try:
                logger.info(f"Starting stage: {stage_name}")
                
                # Check memory before stage
                if self.recovery.detect_memory_overflow():
                    self._handle_memory_overflow()
                
                # Execute stage
                stage_result = stage_func(model_path, result, **kwargs)
                result['stages'][stage_name] = stage_result
                
                # Save checkpoint after successful stage
                self.recovery.save_checkpoint(
                    model_path=model_path,
                    stage=stage_name,
                    state=result,
                    metrics=stage_result.get('metrics', {})
                )
                
            except MemoryError as e:
                logger.error(f"Memory error in {stage_name}: {e}")
                self._handle_memory_overflow()
                # Retry with reduced resources
                kwargs['batch_size'] = kwargs.get('batch_size', 32) // 2
                raise
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU OOM in {stage_name}: {e}")
                torch.cuda.empty_cache()
                # Switch to CPU
                kwargs['device'] = 'cpu'
                raise
                
            except NetworkError as e:
                logger.error(f"Network error in {stage_name}: {e}")
                # Use local cache
                cache_dir = self.recovery.create_local_cache(model_path)
                kwargs['cache_dir'] = cache_dir
                raise
                
            except Exception as e:
                logger.error(f"Error in {stage_name}: {e}")
                # Try recovery
                context = {'stage': stage_name, **kwargs}
                self.recovery.recovery_manager.recover(e, context)
                raise
        
        return result
        
    def _initialize_stage(self, model_path: str, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Initialization stage with error recovery"""
        return {
            'status': 'initialized',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def _loading_stage(self, model_path: str, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Model loading stage with fallback"""
        try:
            # Try normal loading
            return self.base_pipeline._load_model(model_path, **kwargs)
        except Exception as e:
            # Try fallback loading with reduced precision
            logger.warning("Normal loading failed, trying reduced precision")
            kwargs['quantize'] = '8bit'
            return self.base_pipeline._load_model(model_path, **kwargs)
            
    def _processing_stage(self, model_path: str, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Processing stage with adaptive resource management"""
        # Adjust resources based on available memory
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            kwargs['challenges'] = min(kwargs.get('challenges', 10), 5)
            logger.info(f"Reduced challenges due to memory constraints")
        
        return self.base_pipeline._process_challenges(**kwargs)
        
    def _analysis_stage(self, model_path: str, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Analysis stage with graceful degradation"""
        analysis_result = {}
        
        # Try full analysis
        try:
            analysis_result['full'] = self.base_pipeline._run_full_analysis(**kwargs)
        except Exception as e:
            logger.warning(f"Full analysis failed: {e}")
            degradation_manager.degrade_feature('full_analysis', e)
            
            # Fall back to basic analysis
            try:
                analysis_result['basic'] = self.base_pipeline._run_basic_analysis(**kwargs)
            except Exception as e2:
                logger.error(f"Basic analysis also failed: {e2}")
                analysis_result['error'] = str(e2)
        
        return analysis_result
        
    def _finalization_stage(self, model_path: str, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Finalization stage with cleanup"""
        # Cleanup resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def _handle_memory_overflow(self):
        """Handle memory overflow conditions"""
        logger.warning("Handling memory overflow")
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear Python caches
        import functools
        functools.lru_cache.cache_clear()
        
        # Log memory status
        memory = psutil.virtual_memory()
        logger.info(f"Memory after cleanup: {memory.percent}% used")


def add_error_recovery_to_pipeline(pipeline):
    """Add error recovery capabilities to existing pipeline"""
    recovery = PipelineRecovery()
    enhanced = EnhancedREVPipeline(pipeline, recovery)
    
    # Replace process_model method
    original_process = pipeline.process_model
    
    def process_with_recovery(*args, **kwargs):
        try:
            return enhanced.process_model_with_recovery(*args, **kwargs)
        except Exception as e:
            logger.error(f"All recovery attempts failed: {e}")
            # Fall back to original with minimal resources
            kwargs['challenges'] = 1
            kwargs['device'] = 'cpu'
            return original_process(*args, **kwargs)
    
    pipeline.process_model = process_with_recovery
    
    # Add recovery methods to pipeline
    pipeline.save_checkpoint = recovery.save_checkpoint
    pipeline.load_checkpoint = recovery.load_checkpoint
    pipeline.detect_memory_overflow = recovery.detect_memory_overflow
    pipeline.adjust_segment_size = recovery.adjust_segment_size
    
    logger.info("Error recovery capabilities added to pipeline")
    
    return pipeline