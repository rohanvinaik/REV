"""
Device Management System for REV Framework.

Provides centralized device and dtype management to prevent device/dtype
mismatches during true behavioral probing and segment execution.

Addresses critical issues:
1. MPS placeholder storage problems
2. Device incompatibility between tensors
3. Dtype conversion consistency
4. Memory-bounded device transitions
"""

import logging
import warnings
from typing import Optional, Union, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass
import torch
import numpy as np

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types with priority ordering."""
    MPS = "mps"
    CUDA = "cuda" 
    CPU = "cpu"


@dataclass
class DeviceCapabilities:
    """Device capability information."""
    device_type: DeviceType
    available: bool
    memory_gb: Optional[float] = None
    supports_half: bool = False
    supports_bfloat16: bool = False
    max_tensor_size: Optional[int] = None
    fallback_reasons: List[str] = None

    def __post_init__(self):
        if self.fallback_reasons is None:
            self.fallback_reasons = []


class DeviceManager:
    """
    Centralized device and dtype management for REV operations.
    
    Handles device selection, tensor movement, dtype conversions,
    and provides fallback strategies for device compatibility issues.
    """
    
    def __init__(self, 
                 preferred_device: Optional[str] = None,
                 enable_half_precision: bool = True,
                 memory_fraction: float = 0.95,
                 enable_device_logging: bool = True):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device type ('mps', 'cuda', 'cpu', None=auto)
            enable_half_precision: Whether to enable half precision when available
            memory_fraction: Fraction of available memory to use
            enable_device_logging: Whether to log device operations
        """
        self.preferred_device = preferred_device
        self.enable_half_precision = enable_half_precision
        self.memory_fraction = memory_fraction
        self.enable_logging = enable_device_logging
        
        # Device state
        self.current_device: Optional[torch.device] = None
        self.current_dtype = torch.float32
        self.capabilities: Dict[DeviceType, DeviceCapabilities] = {}
        self.tensor_registry: Dict[str, torch.device] = {}
        
        # Setup logging first
        self.device_logger = None
        if self.enable_logging:
            self._setup_device_logging()
        
        # Initialize device capabilities
        self._probe_device_capabilities()
        self._select_optimal_device()

    def _setup_device_logging(self):
        """Setup device operation logging."""
        # Create device-specific logger
        device_logger = logging.getLogger(f"{__name__}.device")
        device_logger.setLevel(logging.INFO)
        
        if not device_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[DEVICE] %(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            device_logger.addHandler(handler)
        
        self.device_logger = device_logger

    def _probe_device_capabilities(self):
        """Probe and catalog device capabilities."""
        # Probe MPS
        mps_available = torch.backends.mps.is_available()
        mps_reasons = []
        
        if not mps_available:
            mps_reasons.append("MPS backend not available")
        else:
            # Check for MPS placeholder storage issues
            try:
                test_tensor = torch.randn(100, 100, device='mps')
                _ = test_tensor.cpu()
                del test_tensor
                torch.mps.empty_cache()
            except Exception as e:
                mps_available = False
                mps_reasons.append(f"MPS placeholder storage issue: {e}")

        self.capabilities[DeviceType.MPS] = DeviceCapabilities(
            device_type=DeviceType.MPS,
            available=mps_available,
            supports_half=True,  # MPS supports half precision
            supports_bfloat16=False,  # MPS doesn't support bfloat16
            fallback_reasons=mps_reasons
        )

        # Probe CUDA
        cuda_available = torch.cuda.is_available()
        cuda_reasons = []
        cuda_memory = None
        
        if cuda_available:
            try:
                cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception as e:
                cuda_reasons.append(f"CUDA memory query failed: {e}")
        else:
            cuda_reasons.append("CUDA not available")

        self.capabilities[DeviceType.CUDA] = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            available=cuda_available,
            memory_gb=cuda_memory,
            supports_half=cuda_available,
            supports_bfloat16=cuda_available,
            fallback_reasons=cuda_reasons
        )

        # CPU is always available
        self.capabilities[DeviceType.CPU] = DeviceCapabilities(
            device_type=DeviceType.CPU,
            available=True,
            supports_half=True,
            supports_bfloat16=True,
            fallback_reasons=[]
        )

    def _select_optimal_device(self):
        """Select optimal device based on capabilities and preferences."""
        if self.preferred_device:
            # Use explicitly preferred device if available
            device_type = DeviceType(self.preferred_device.lower())
            if self.capabilities[device_type].available:
                self.current_device = torch.device(self.preferred_device)
                if self.enable_logging and self.device_logger:
                    self.device_logger.info(f"Using preferred device: {self.current_device}")
                return
            else:
                if self.enable_logging and self.device_logger:
                    reasons = self.capabilities[device_type].fallback_reasons
                    self.device_logger.warning(f"Preferred device {device_type.value} not available: {reasons}")

        # Fallback chain: MPS -> CUDA -> CPU
        fallback_chain = [DeviceType.MPS, DeviceType.CUDA, DeviceType.CPU]
        
        for device_type in fallback_chain:
            if self.capabilities[device_type].available:
                self.current_device = torch.device(device_type.value)
                if self.enable_logging and self.device_logger:
                    self.device_logger.info(f"Selected device: {self.current_device}")
                break

        # Set optimal dtype for selected device
        self._select_optimal_dtype()

    def _select_optimal_dtype(self):
        """Select optimal dtype for current device."""
        if not self.current_device:
            self.current_dtype = torch.float32
            return

        device_type = DeviceType(self.current_device.type)
        capabilities = self.capabilities[device_type]
        
        if self.enable_half_precision and capabilities.supports_half:
            # Prefer half precision for memory efficiency
            if capabilities.supports_bfloat16 and self.current_device.type == 'cuda':
                # Use bfloat16 on CUDA for better numerical stability
                self.current_dtype = torch.bfloat16
            else:
                # Use float16 on MPS/CUDA or when bfloat16 not available
                self.current_dtype = torch.float16
        else:
            self.current_dtype = torch.float32

        if self.enable_logging and self.device_logger:
            self.device_logger.info(f"Selected dtype: {self.current_dtype}")

    def get_device(self) -> torch.device:
        """Get current optimal device."""
        return self.current_device

    def get_dtype(self) -> torch.dtype:
        """Get current optimal dtype."""
        return self.current_dtype

    def ensure_device_consistency(self, 
                                tensors: Union[torch.Tensor, List[torch.Tensor]], 
                                target_device: Optional[torch.device] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Ensure all tensors are on the same device.
        
        Args:
            tensors: Single tensor or list of tensors
            target_device: Target device (uses current if None)
            
        Returns:
            Tensor(s) moved to target device
        """
        if target_device is None:
            target_device = self.current_device

        is_single = isinstance(tensors, torch.Tensor)
        tensor_list = [tensors] if is_single else tensors

        # Move tensors to target device
        moved_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor):
                if tensor.device != target_device:
                    try:
                        moved_tensor = tensor.to(target_device)
                        moved_tensors.append(moved_tensor)
                        
                        if self.enable_logging and self.device_logger:
                            self.device_logger.debug(f"Moved tensor from {tensor.device} to {target_device}")
                    except Exception as e:
                        # Fallback to CPU if move fails
                        if target_device.type != 'cpu':
                            if self.device_logger:
                                self.device_logger.warning(f"Failed to move tensor to {target_device}, falling back to CPU: {e}")
                            moved_tensor = tensor.to('cpu')
                            moved_tensors.append(moved_tensor)
                        else:
                            raise RuntimeError(f"Failed to move tensor to CPU: {e}")
                else:
                    moved_tensors.append(tensor)
            else:
                moved_tensors.append(tensor)

        return moved_tensors[0] if is_single else moved_tensors

    def ensure_dtype_consistency(self, 
                                tensors: Union[torch.Tensor, List[torch.Tensor]], 
                                target_dtype: Optional[torch.dtype] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Ensure all tensors have consistent dtype.
        
        Args:
            tensors: Single tensor or list of tensors
            target_dtype: Target dtype (uses current if None)
            
        Returns:
            Tensor(s) converted to target dtype
        """
        if target_dtype is None:
            target_dtype = self.current_dtype

        is_single = isinstance(tensors, torch.Tensor)
        tensor_list = [tensors] if is_single else tensors

        converted_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor):
                if tensor.dtype != target_dtype:
                    try:
                        # Special handling for dtype conversions
                        if target_dtype in [torch.float16, torch.bfloat16]:
                            # Check if device supports the target dtype
                            device_type = DeviceType(tensor.device.type)
                            capabilities = self.capabilities[device_type]
                            
                            if target_dtype == torch.bfloat16 and not capabilities.supports_bfloat16:
                                # Fall back to float16
                                converted_tensor = tensor.to(torch.float16)
                                if self.enable_logging and self.device_logger:
                                    self.device_logger.warning(f"Device doesn't support bfloat16, using float16 instead")
                            else:
                                converted_tensor = tensor.to(target_dtype)
                        else:
                            converted_tensor = tensor.to(target_dtype)
                        
                        converted_tensors.append(converted_tensor)
                        
                        if self.enable_logging and self.device_logger:
                            self.device_logger.debug(f"Converted tensor from {tensor.dtype} to {converted_tensor.dtype}")
                    except Exception as e:
                        if self.device_logger:
                            self.device_logger.warning(f"Dtype conversion failed, keeping original dtype: {e}")
                        converted_tensors.append(tensor)
                else:
                    converted_tensors.append(tensor)
            else:
                converted_tensors.append(tensor)

        return converted_tensors[0] if is_single else converted_tensors

    def safe_tensor_operation(self, 
                            operation: callable, 
                            *tensors: torch.Tensor, 
                            **kwargs) -> torch.Tensor:
        """
        Safely perform tensor operation with device/dtype consistency.
        
        Args:
            operation: Tensor operation to perform
            *tensors: Input tensors
            **kwargs: Additional arguments for operation
            
        Returns:
            Result of operation with proper device/dtype handling
        """
        try:
            # Ensure all tensors are on same device and dtype
            consistent_tensors = self.ensure_device_consistency(list(tensors))
            consistent_tensors = self.ensure_dtype_consistency(consistent_tensors)
            
            # Perform operation
            result = operation(*consistent_tensors, **kwargs)
            
            return result
            
        except RuntimeError as e:
            if "mps" in str(e).lower() and "placeholder" in str(e).lower():
                # Handle MPS placeholder storage issue
                if self.enable_logging and self.device_logger:
                    self.device_logger.warning(f"MPS placeholder issue detected, falling back to CPU: {e}")
                
                # Move to CPU and retry
                cpu_tensors = [t.to('cpu') if isinstance(t, torch.Tensor) else t for t in tensors]
                result = operation(*cpu_tensors, **kwargs)
                
                return result
            else:
                raise

    def numpy_to_tensor(self, 
                       array: np.ndarray, 
                       preserve_device: bool = True,
                       preserve_dtype: bool = False) -> torch.Tensor:
        """
        Convert numpy array to tensor with proper device/dtype handling.
        
        Args:
            array: Numpy array to convert
            preserve_device: Whether to place on current device
            preserve_dtype: Whether to preserve numpy dtype
            
        Returns:
            PyTorch tensor with proper device/dtype
        """
        # Convert to tensor
        tensor = torch.from_numpy(array)
        
        # Handle dtype conversion
        if not preserve_dtype:
            # Convert to current optimal dtype
            if tensor.dtype in [torch.float64, torch.float32, torch.float16]:
                tensor = tensor.to(self.current_dtype)
        
        # Handle device placement
        if preserve_device and self.current_device:
            tensor = self.ensure_device_consistency(tensor)
        
        return tensor

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array with proper device handling.
        
        Args:
            tensor: PyTorch tensor to convert
            
        Returns:
            Numpy array
        """
        # Move to CPU if needed
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # Convert to numpy
        return tensor.detach().numpy()

    def handle_out_of_memory(self, error: Exception) -> bool:
        """
        Handle out of memory errors by cleaning up and suggesting fallbacks.
        
        Args:
            error: The OOM error that occurred
            
        Returns:
            True if recovery was attempted, False if no recovery possible
        """
        if self.enable_logging and self.device_logger:
            self.device_logger.error(f"Out of memory error: {error}")
        
        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.enable_logging and self.device_logger:
                self.device_logger.info("Cleared CUDA cache")
        
        if self.current_device.type == 'mps':
            torch.mps.empty_cache()
            if self.enable_logging and self.device_logger:
                self.device_logger.info("Cleared MPS cache")
        
        # Suggest fallback to CPU
        if self.current_device.type != 'cpu':
            if self.enable_logging and self.device_logger:
                self.device_logger.warning("Consider falling back to CPU for memory-bounded operation")
            return True
        
        return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        usage = {"device": str(self.current_device)}
        
        if self.current_device.type == 'cuda':
            if torch.cuda.is_available():
                usage.update({
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "cached_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
                })
        elif self.current_device.type == 'mps':
            if torch.backends.mps.is_available():
                usage.update({
                    "current_allocated_mb": torch.mps.current_allocated_memory() / (1024**2)
                })
        
        return usage

    def create_fallback_strategy(self) -> List[Tuple[torch.device, torch.dtype]]:
        """
        Create fallback strategy for device/dtype combinations.
        
        Returns:
            List of (device, dtype) pairs in fallback order
        """
        strategies = []
        
        # Current strategy first
        strategies.append((self.current_device, self.current_dtype))
        
        # Fallback to lower precision on same device
        if self.current_dtype not in [torch.float16, torch.bfloat16]:
            device_type = DeviceType(self.current_device.type)
            capabilities = self.capabilities[device_type]
            
            if capabilities.supports_bfloat16:
                strategies.append((self.current_device, torch.bfloat16))
            if capabilities.supports_half:
                strategies.append((self.current_device, torch.float16))
        
        # Fallback to different devices
        fallback_devices = [DeviceType.CUDA, DeviceType.CPU] if self.current_device.type == 'mps' else [DeviceType.CPU]
        
        for device_type in fallback_devices:
            if self.capabilities[device_type].available:
                device = torch.device(device_type.value)
                capabilities = self.capabilities[device_type]
                
                # Try different dtypes for fallback device
                if capabilities.supports_bfloat16:
                    strategies.append((device, torch.bfloat16))
                if capabilities.supports_half:
                    strategies.append((device, torch.float16))
                strategies.append((device, torch.float32))
        
        return strategies

    def log_device_info(self):
        """Log comprehensive device information."""
        if not self.enable_logging or not self.device_logger:
            return
            
        self.device_logger.info("="*60)
        self.device_logger.info("DEVICE MANAGER STATUS")
        self.device_logger.info("="*60)
        self.device_logger.info(f"Current Device: {self.current_device}")
        self.device_logger.info(f"Current Dtype: {self.current_dtype}")
        self.device_logger.info("")
        
        for device_type, capabilities in self.capabilities.items():
            status = "AVAILABLE" if capabilities.available else "UNAVAILABLE"
            self.device_logger.info(f"{device_type.value.upper()}: {status}")
            
            if capabilities.available:
                if capabilities.memory_gb:
                    self.device_logger.info(f"  Memory: {capabilities.memory_gb:.1f} GB")
                self.device_logger.info(f"  Half Precision: {capabilities.supports_half}")
                self.device_logger.info(f"  BFloat16: {capabilities.supports_bfloat16}")
            else:
                for reason in capabilities.fallback_reasons:
                    self.device_logger.info(f"  Reason: {reason}")
            self.device_logger.info("")
        
        # Memory usage
        usage = self.get_memory_usage()
        self.device_logger.info("Memory Usage:")
        for key, value in usage.items():
            if isinstance(value, float):
                self.device_logger.info(f"  {key}: {value:.1f}")
            else:
                self.device_logger.info(f"  {key}: {value}")
        
        self.device_logger.info("="*60)


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_global_device_manager() -> DeviceManager:
    """Get global device manager instance."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_global_device_manager(manager: DeviceManager):
    """Set global device manager instance."""
    global _global_device_manager
    _global_device_manager = manager


# Convenience functions
def get_optimal_device() -> torch.device:
    """Get optimal device from global manager."""
    return get_global_device_manager().get_device()


def get_optimal_dtype() -> torch.dtype:
    """Get optimal dtype from global manager."""
    return get_global_device_manager().get_dtype()


def ensure_tensor_compatibility(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Ensure tensors are compatible for operations."""
    manager = get_global_device_manager()
    consistent_tensors = manager.ensure_device_consistency(list(tensors))
    consistent_tensors = manager.ensure_dtype_consistency(consistent_tensors)
    return consistent_tensors