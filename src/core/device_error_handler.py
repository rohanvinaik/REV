"""
Device Error Handler for REV Framework.

Provides comprehensive error handling for device/dtype mismatches,
OOM errors, and other device-related issues during behavioral probing
and segment execution.
"""

import logging
import traceback
import warnings
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np

from .device_manager import DeviceManager, get_global_device_manager

logger = logging.getLogger(__name__)


class DeviceErrorType(Enum):
    """Types of device-related errors."""
    DEVICE_MISMATCH = "device_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    OUT_OF_MEMORY = "out_of_memory"
    MPS_PLACEHOLDER = "mps_placeholder"
    CUDA_ERROR = "cuda_error"
    TENSOR_OPERATION = "tensor_operation"
    MODEL_LOADING = "model_loading"
    UNKNOWN = "unknown"


@dataclass
class DeviceError:
    """Device error information."""
    error_type: DeviceErrorType
    original_exception: Exception
    device_info: Dict[str, Any]
    suggested_fix: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_device: Optional[torch.device] = None
    fallback_dtype: Optional[torch.dtype] = None


class DeviceErrorHandler:
    """
    Comprehensive error handler for device-related issues.
    
    Provides automatic error classification, recovery strategies,
    and detailed logging for device/dtype problems in REV operations.
    """
    
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize device error handler.
        
        Args:
            device_manager: Device manager to use for recovery
        """
        self.device_manager = device_manager or get_global_device_manager()
        self.error_history: List[DeviceError] = []
        self.recovery_strategies: Dict[DeviceErrorType, Callable] = {}
        self.error_stats: Dict[DeviceErrorType, int] = {}
        
        # Register built-in recovery strategies
        self._register_recovery_strategies()
    
    def _register_recovery_strategies(self):
        """Register built-in error recovery strategies."""
        self.recovery_strategies = {
            DeviceErrorType.DEVICE_MISMATCH: self._recover_device_mismatch,
            DeviceErrorType.DTYPE_MISMATCH: self._recover_dtype_mismatch,
            DeviceErrorType.OUT_OF_MEMORY: self._recover_out_of_memory,
            DeviceErrorType.MPS_PLACEHOLDER: self._recover_mps_placeholder,
            DeviceErrorType.CUDA_ERROR: self._recover_cuda_error,
            DeviceErrorType.TENSOR_OPERATION: self._recover_tensor_operation,
            DeviceErrorType.MODEL_LOADING: self._recover_model_loading
        }
    
    def classify_error(self, exception: Exception) -> DeviceErrorType:
        """
        Classify device-related error type.
        
        Args:
            exception: The exception to classify
            
        Returns:
            Classified error type
        """
        error_str = str(exception).lower()
        error_type_name = type(exception).__name__.lower()
        
        # Check for specific error patterns
        if "device" in error_str and "mismatch" in error_str:
            return DeviceErrorType.DEVICE_MISMATCH
        elif "expected" in error_str and ("device" in error_str or "tensor" in error_str):
            return DeviceErrorType.DEVICE_MISMATCH
        elif "dtype" in error_str or "data type" in error_str:
            return DeviceErrorType.DTYPE_MISMATCH
        elif "out of memory" in error_str or "oom" in error_str:
            return DeviceErrorType.OUT_OF_MEMORY
        elif "mps" in error_str and "placeholder" in error_str:
            return DeviceErrorType.MPS_PLACEHOLDER
        elif "cuda" in error_str and ("error" in error_str or "runtime" in error_str):
            return DeviceErrorType.CUDA_ERROR
        elif isinstance(exception, RuntimeError) and "tensor" in error_str:
            return DeviceErrorType.TENSOR_OPERATION
        elif "load" in error_str and ("model" in error_str or "state_dict" in error_str):
            return DeviceErrorType.MODEL_LOADING
        else:
            return DeviceErrorType.UNKNOWN
    
    def create_error_info(self, exception: Exception, 
                         context: Optional[Dict[str, Any]] = None) -> DeviceError:
        """
        Create comprehensive error information.
        
        Args:
            exception: The exception that occurred
            context: Optional context information
            
        Returns:
            DeviceError object with detailed information
        """
        error_type = self.classify_error(exception)
        
        # Gather device information
        device_info = {
            "current_device": str(self.device_manager.get_device()),
            "current_dtype": str(self.device_manager.get_dtype()),
            "memory_usage": self.device_manager.get_memory_usage(),
            "error_type": error_type.value,
            "error_message": str(exception),
            "error_class": type(exception).__name__
        }
        
        # Add context if provided
        if context:
            device_info.update(context)
        
        # Generate suggested fix
        suggested_fix = self._generate_suggested_fix(error_type, exception, context)
        
        return DeviceError(
            error_type=error_type,
            original_exception=exception,
            device_info=device_info,
            suggested_fix=suggested_fix
        )
    
    def _generate_suggested_fix(self, error_type: DeviceErrorType, 
                               exception: Exception,
                               context: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable suggested fix."""
        fixes = {
            DeviceErrorType.DEVICE_MISMATCH: 
                "Ensure all tensors are on the same device. Use device_manager.ensure_device_consistency().",
            DeviceErrorType.DTYPE_MISMATCH:
                "Ensure all tensors have compatible dtypes. Use device_manager.ensure_dtype_consistency().",
            DeviceErrorType.OUT_OF_MEMORY:
                "Reduce batch size, use gradient checkpointing, or fall back to CPU.",
            DeviceErrorType.MPS_PLACEHOLDER:
                "MPS placeholder storage issue. Fall back to CUDA or CPU.",
            DeviceErrorType.CUDA_ERROR:
                "CUDA runtime error. Check CUDA installation or fall back to CPU.",
            DeviceErrorType.TENSOR_OPERATION:
                "Tensor operation failed. Check tensor shapes, devices, and dtypes.",
            DeviceErrorType.MODEL_LOADING:
                "Model loading failed. Check model path and device compatibility.",
            DeviceErrorType.UNKNOWN:
                "Unknown device error. Check logs for details."
        }
        
        base_fix = fixes.get(error_type, "Check device and dtype compatibility.")
        
        # Add context-specific suggestions
        if context:
            if "tensor_shapes" in context:
                base_fix += f" Tensor shapes: {context['tensor_shapes']}"
            if "devices" in context:
                base_fix += f" Devices: {context['devices']}"
        
        return base_fix
    
    def handle_error(self, exception: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> Tuple[bool, Any]:
        """
        Handle device error with comprehensive recovery.
        
        Args:
            exception: The exception to handle
            context: Optional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        # Create error info
        error_info = self.create_error_info(exception, context)
        
        # Update statistics
        self.error_stats[error_info.error_type] = self.error_stats.get(error_info.error_type, 0) + 1
        
        # Log error details
        logger.error(f"Device error encountered: {error_info.error_type.value}")
        logger.error(f"Original exception: {exception}")
        logger.error(f"Suggested fix: {error_info.suggested_fix}")
        logger.error(f"Device info: {error_info.device_info}")
        
        recovery_successful = False
        recovery_result = None
        
        if attempt_recovery:
            # Attempt recovery using registered strategy
            if error_info.error_type in self.recovery_strategies:
                try:
                    error_info.recovery_attempted = True
                    recovery_result = self.recovery_strategies[error_info.error_type](
                        error_info, context
                    )
                    recovery_successful = recovery_result is not None
                    error_info.recovery_successful = recovery_successful
                    
                    if recovery_successful:
                        logger.info(f"Successfully recovered from {error_info.error_type.value}")
                    else:
                        logger.warning(f"Recovery attempt failed for {error_info.error_type.value}")
                        
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
                    recovery_successful = False
        
        # Store error info
        self.error_history.append(error_info)
        
        return recovery_successful, recovery_result
    
    def _recover_device_mismatch(self, error_info: DeviceError, 
                                context: Optional[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Recover from device mismatch error."""
        if not context or "tensors" not in context:
            return None
        
        try:
            tensors = context["tensors"]
            target_device = self.device_manager.get_device()
            
            # Move all tensors to target device
            recovered_tensors = {}
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    recovered_tensor = self.device_manager.ensure_device_consistency(
                        tensor, target_device
                    )
                    recovered_tensors[name] = recovered_tensor
                else:
                    recovered_tensors[name] = tensor
            
            error_info.fallback_device = target_device
            return recovered_tensors
            
        except Exception as e:
            logger.error(f"Device mismatch recovery failed: {e}")
            return None
    
    def _recover_dtype_mismatch(self, error_info: DeviceError,
                               context: Optional[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Recover from dtype mismatch error."""
        if not context or "tensors" not in context:
            return None
        
        try:
            tensors = context["tensors"]
            target_dtype = self.device_manager.get_dtype()
            
            # Convert all tensors to target dtype
            recovered_tensors = {}
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    recovered_tensor = self.device_manager.ensure_dtype_consistency(
                        tensor, target_dtype
                    )
                    recovered_tensors[name] = recovered_tensor
                else:
                    recovered_tensors[name] = tensor
            
            error_info.fallback_dtype = target_dtype
            return recovered_tensors
            
        except Exception as e:
            logger.error(f"Dtype mismatch recovery failed: {e}")
            return None
    
    def _recover_out_of_memory(self, error_info: DeviceError,
                              context: Optional[Dict[str, Any]]) -> Optional[torch.device]:
        """Recover from out of memory error."""
        try:
            # Try device manager's OOM handling
            if self.device_manager.handle_out_of_memory(error_info.original_exception):
                # Suggest CPU fallback
                cpu_device = torch.device("cpu")
                error_info.fallback_device = cpu_device
                logger.info("Suggesting CPU fallback for OOM recovery")
                return cpu_device
            return None
        except Exception as e:
            logger.error(f"OOM recovery failed: {e}")
            return None
    
    def _recover_mps_placeholder(self, error_info: DeviceError,
                                context: Optional[Dict[str, Any]]) -> Optional[torch.device]:
        """Recover from MPS placeholder storage error."""
        try:
            # Clear MPS cache and fall back to CUDA or CPU
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Try CUDA first, then CPU
            if torch.cuda.is_available():
                fallback_device = torch.device("cuda")
            else:
                fallback_device = torch.device("cpu")
            
            error_info.fallback_device = fallback_device
            logger.info(f"MPS placeholder issue: falling back to {fallback_device}")
            return fallback_device
            
        except Exception as e:
            logger.error(f"MPS placeholder recovery failed: {e}")
            return None
    
    def _recover_cuda_error(self, error_info: DeviceError,
                           context: Optional[Dict[str, Any]]) -> Optional[torch.device]:
        """Recover from CUDA error."""
        try:
            # Clear CUDA cache and fall back to CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cpu_device = torch.device("cpu")
            error_info.fallback_device = cpu_device
            logger.info("CUDA error: falling back to CPU")
            return cpu_device
            
        except Exception as e:
            logger.error(f"CUDA error recovery failed: {e}")
            return None
    
    def _recover_tensor_operation(self, error_info: DeviceError,
                                 context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Recover from tensor operation error."""
        if not context:
            return None
        
        try:
            # Try to fix tensors in context
            fixed_context = {}
            
            if "tensors" in context:
                # Fix tensor device/dtype consistency
                tensors = context["tensors"]
                fixed_tensors = {}
                
                for name, tensor in tensors.items():
                    if isinstance(tensor, torch.Tensor):
                        fixed_tensor = self.device_manager.ensure_device_consistency(tensor)
                        fixed_tensor = self.device_manager.ensure_dtype_consistency(fixed_tensor)
                        fixed_tensors[name] = fixed_tensor
                    else:
                        fixed_tensors[name] = tensor
                
                fixed_context["tensors"] = fixed_tensors
            
            # Copy other context items
            for key, value in context.items():
                if key not in fixed_context:
                    fixed_context[key] = value
            
            return fixed_context
            
        except Exception as e:
            logger.error(f"Tensor operation recovery failed: {e}")
            return None
    
    def _recover_model_loading(self, error_info: DeviceError,
                              context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Recover from model loading error."""
        try:
            # Suggest loading on CPU first, then moving to target device
            recovery_info = {
                "suggested_device": torch.device("cpu"),
                "map_location": "cpu",
                "move_to_device_after_load": True
            }
            
            error_info.fallback_device = torch.device("cpu")
            return recovery_info
            
        except Exception as e:
            logger.error(f"Model loading recovery failed: {e}")
            return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {"total_errors": 0, "message": "No device errors encountered"}
        
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
        recovery_rate = successful_recoveries / total_errors if total_errors > 0 else 0
        
        return {
            "total_errors": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": recovery_rate,
            "error_types": dict(self.error_stats),
            "most_common_error": max(self.error_stats, key=self.error_stats.get) if self.error_stats else None
        }
    
    def clear_history(self):
        """Clear error history and statistics."""
        self.error_history.clear()
        self.error_stats.clear()


def device_safe_operation(retry_on_device_error: bool = True,
                         fallback_to_cpu: bool = True):
    """
    Decorator for device-safe operations with automatic error handling.
    
    Args:
        retry_on_device_error: Whether to retry on device errors
        fallback_to_cpu: Whether to fall back to CPU on device errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = DeviceErrorHandler()
            
            try:
                # First attempt
                return func(*args, **kwargs)
                
            except Exception as e:
                # Check if this is a device-related error
                error_type = error_handler.classify_error(e)
                
                if error_type == DeviceErrorType.UNKNOWN:
                    # Not a device error, re-raise
                    raise
                
                # Handle device error
                logger.warning(f"Device error in {func.__name__}: {e}")
                
                if retry_on_device_error:
                    # Attempt recovery
                    context = {
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs
                    }
                    
                    recovery_successful, recovery_result = error_handler.handle_error(
                        e, context, attempt_recovery=True
                    )
                    
                    if recovery_successful and fallback_to_cpu:
                        # Retry with CPU
                        logger.info(f"Retrying {func.__name__} with CPU fallback")
                        
                        # Move tensor args to CPU
                        cpu_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                cpu_args.append(arg.cpu())
                            else:
                                cpu_args.append(arg)
                        
                        cpu_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor):
                                cpu_kwargs[key] = value.cpu()
                            else:
                                cpu_kwargs[key] = value
                        
                        try:
                            return func(*cpu_args, **cpu_kwargs)
                        except Exception as retry_error:
                            logger.error(f"CPU fallback also failed: {retry_error}")
                            raise
                
                # If recovery failed or not attempted, re-raise original error
                raise
                
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler: Optional[DeviceErrorHandler] = None


def get_global_error_handler() -> DeviceErrorHandler:
    """Get global device error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = DeviceErrorHandler()
    return _global_error_handler


def set_global_error_handler(handler: DeviceErrorHandler):
    """Set global device error handler instance."""
    global _global_error_handler
    _global_error_handler = handler


def handle_device_error(exception: Exception, 
                       context: Optional[Dict[str, Any]] = None,
                       attempt_recovery: bool = True) -> Tuple[bool, Any]:
    """Convenience function to handle device errors using global handler."""
    return get_global_error_handler().handle_error(exception, context, attempt_recovery)