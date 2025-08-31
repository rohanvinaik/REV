"""
Device Operation Logging for REV Framework.

Provides centralized, structured logging for device operations,
performance metrics, and error tracking during behavioral probing
and segment execution.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import torch
import numpy as np

from .device_manager import DeviceManager, get_global_device_manager


class LogLevel(Enum):
    """Device operation log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeviceOperation(Enum):
    """Types of device operations to log."""
    TENSOR_MOVE = "tensor_move"
    DTYPE_CONVERSION = "dtype_conversion"
    MEMORY_ALLOCATION = "memory_allocation"
    MEMORY_CLEANUP = "memory_cleanup"
    MODEL_LOADING = "model_loading"
    FORWARD_PASS = "forward_pass"
    DEVICE_SELECTION = "device_selection"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class DeviceLogEntry:
    """Structured device operation log entry."""
    timestamp: str
    operation: DeviceOperation
    level: LogLevel
    message: str
    device_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tensor_info: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class DeviceLogger:
    """
    Comprehensive device operation logger.
    
    Provides structured logging for device operations with performance
    tracking, error analysis, and export capabilities.
    """
    
    def __init__(self, 
                 name: str = "device_ops",
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_structured: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize device logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            enable_console: Enable console logging
            enable_structured: Enable structured JSON logging
            log_level: Logging level
        """
        self.name = name
        self.enable_console = enable_console
        self.enable_structured = enable_structured
        self.structured_logs: List[DeviceLogEntry] = []
        
        # Setup Python logger
        self.logger = logging.getLogger(f"rev.device.{name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '[DEVICE-{name}] %(asctime)s - %(levelname)s - %(message)s'.format(name=name.upper())
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Setup file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self.operation_start_times: Dict[str, float] = {}
        self.performance_history: Dict[DeviceOperation, List[float]] = {}
        
        # Get device manager reference
        self.device_manager: Optional[DeviceManager] = None
    
    def set_device_manager(self, device_manager: DeviceManager):
        """Set device manager reference for context."""
        self.device_manager = device_manager
    
    def _get_device_context(self) -> Dict[str, Any]:
        """Get current device context information."""
        if self.device_manager is None:
            self.device_manager = get_global_device_manager()
        
        return {
            "current_device": str(self.device_manager.get_device()),
            "current_dtype": str(self.device_manager.get_dtype()),
            "memory_usage": self.device_manager.get_memory_usage()
        }
    
    def _create_log_entry(self, 
                         operation: DeviceOperation,
                         level: LogLevel,
                         message: str,
                         **kwargs) -> DeviceLogEntry:
        """Create structured log entry."""
        return DeviceLogEntry(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            level=level,
            message=message,
            device_info=self._get_device_context(),
            **kwargs
        )
    
    def log_device_operation(self,
                           operation: DeviceOperation,
                           level: LogLevel,
                           message: str,
                           tensor_info: Optional[Dict[str, Any]] = None,
                           performance_metrics: Optional[Dict[str, float]] = None,
                           error_info: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None):
        """
        Log device operation with structured data.
        
        Args:
            operation: Type of device operation
            level: Log level
            message: Log message
            tensor_info: Information about tensors involved
            performance_metrics: Performance metrics
            error_info: Error information if applicable
            context: Additional context
        """
        # Create structured log entry
        log_entry = self._create_log_entry(
            operation=operation,
            level=level,
            message=message,
            tensor_info=tensor_info or {},
            performance_metrics=performance_metrics or {},
            error_info=error_info,
            context=context or {}
        )
        
        # Store for structured logging
        if self.enable_structured:
            self.structured_logs.append(log_entry)
        
        # Log to Python logger
        python_level = getattr(logging, level.value.upper())
        
        # Create detailed message for Python logger
        details = []
        if tensor_info:
            details.append(f"Tensors: {tensor_info}")
        if performance_metrics:
            details.append(f"Perf: {performance_metrics}")
        if error_info:
            details.append(f"Error: {error_info}")
        if context:
            details.append(f"Context: {context}")
        
        detailed_message = message
        if details:
            detailed_message += f" | {' | '.join(details)}"
        
        self.logger.log(python_level, detailed_message)
        
        # Update performance history
        if performance_metrics:
            if operation not in self.performance_history:
                self.performance_history[operation] = []
            
            if "duration_ms" in performance_metrics:
                self.performance_history[operation].append(performance_metrics["duration_ms"])
    
    def start_operation_timing(self, operation_id: str):
        """Start timing an operation."""
        self.operation_start_times[operation_id] = time.perf_counter()
    
    def end_operation_timing(self, operation_id: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        if operation_id not in self.operation_start_times:
            return 0.0
        
        start_time = self.operation_start_times.pop(operation_id)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return duration_ms
    
    def log_tensor_move(self,
                       tensor_info: Dict[str, Any],
                       source_device: str,
                       target_device: str,
                       duration_ms: Optional[float] = None,
                       success: bool = True):
        """Log tensor movement operation."""
        message = f"Moved tensor from {source_device} to {target_device}"
        if not success:
            message = f"FAILED: {message}"
        
        performance_metrics = {}
        if duration_ms is not None:
            performance_metrics["duration_ms"] = duration_ms
        
        self.log_device_operation(
            operation=DeviceOperation.TENSOR_MOVE,
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=message,
            tensor_info=tensor_info,
            performance_metrics=performance_metrics
        )
    
    def log_dtype_conversion(self,
                           tensor_info: Dict[str, Any],
                           source_dtype: str,
                           target_dtype: str,
                           duration_ms: Optional[float] = None,
                           success: bool = True):
        """Log dtype conversion operation."""
        message = f"Converted tensor from {source_dtype} to {target_dtype}"
        if not success:
            message = f"FAILED: {message}"
        
        performance_metrics = {}
        if duration_ms is not None:
            performance_metrics["duration_ms"] = duration_ms
        
        self.log_device_operation(
            operation=DeviceOperation.DTYPE_CONVERSION,
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=message,
            tensor_info=tensor_info,
            performance_metrics=performance_metrics
        )
    
    def log_memory_allocation(self,
                            size_mb: float,
                            device: str,
                            allocation_type: str = "unknown"):
        """Log memory allocation."""
        self.log_device_operation(
            operation=DeviceOperation.MEMORY_ALLOCATION,
            level=LogLevel.DEBUG,
            message=f"Allocated {size_mb:.2f}MB on {device}",
            context={"allocation_type": allocation_type, "size_mb": size_mb}
        )
    
    def log_memory_cleanup(self,
                          freed_mb: Optional[float] = None,
                          device: str = "unknown"):
        """Log memory cleanup operation."""
        message = f"Memory cleanup on {device}"
        if freed_mb is not None:
            message += f" (freed {freed_mb:.2f}MB)"
        
        context = {"device": device}
        if freed_mb is not None:
            context["freed_mb"] = freed_mb
        
        self.log_device_operation(
            operation=DeviceOperation.MEMORY_CLEANUP,
            level=LogLevel.DEBUG,
            message=message,
            context=context
        )
    
    def log_model_loading(self,
                         model_info: Dict[str, Any],
                         device: str,
                         duration_ms: float,
                         success: bool = True,
                         error_info: Optional[Dict[str, Any]] = None):
        """Log model loading operation."""
        message = f"Loaded model on {device} in {duration_ms:.2f}ms"
        if not success:
            message = f"FAILED: Model loading on {device}"
        
        self.log_device_operation(
            operation=DeviceOperation.MODEL_LOADING,
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=message,
            performance_metrics={"duration_ms": duration_ms},
            error_info=error_info,
            context=model_info
        )
    
    def log_forward_pass(self,
                        input_info: Dict[str, Any],
                        output_info: Dict[str, Any],
                        duration_ms: float,
                        device: str,
                        success: bool = True,
                        error_info: Optional[Dict[str, Any]] = None):
        """Log forward pass operation."""
        message = f"Forward pass on {device} in {duration_ms:.2f}ms"
        if not success:
            message = f"FAILED: Forward pass on {device}"
        
        context = {
            "input_info": input_info,
            "output_info": output_info,
            "device": device
        }
        
        self.log_device_operation(
            operation=DeviceOperation.FORWARD_PASS,
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=message,
            performance_metrics={"duration_ms": duration_ms},
            error_info=error_info,
            context=context
        )
    
    def log_device_selection(self,
                           available_devices: List[str],
                           selected_device: str,
                           selection_reason: str):
        """Log device selection process."""
        message = f"Selected device {selected_device}: {selection_reason}"
        
        context = {
            "available_devices": available_devices,
            "selected_device": selected_device,
            "selection_reason": selection_reason
        }
        
        self.log_device_operation(
            operation=DeviceOperation.DEVICE_SELECTION,
            level=LogLevel.INFO,
            message=message,
            context=context
        )
    
    def log_error_recovery(self,
                          error_type: str,
                          original_error: str,
                          recovery_strategy: str,
                          success: bool,
                          duration_ms: Optional[float] = None):
        """Log error recovery attempt."""
        message = f"Error recovery for {error_type}: {recovery_strategy}"
        if not success:
            message = f"FAILED: {message}"
        
        error_info = {
            "error_type": error_type,
            "original_error": original_error,
            "recovery_strategy": recovery_strategy,
            "recovery_successful": success
        }
        
        performance_metrics = {}
        if duration_ms is not None:
            performance_metrics["recovery_duration_ms"] = duration_ms
        
        self.log_device_operation(
            operation=DeviceOperation.ERROR_RECOVERY,
            level=LogLevel.WARNING if success else LogLevel.ERROR,
            message=message,
            performance_metrics=performance_metrics,
            error_info=error_info
        )
    
    def log_performance_metric(self,
                              metric_name: str,
                              value: float,
                              unit: str,
                              context: Optional[Dict[str, Any]] = None):
        """Log performance metric."""
        message = f"Performance metric {metric_name}: {value} {unit}"
        
        performance_metrics = {
            metric_name: value,
            "unit": unit
        }
        
        self.log_device_operation(
            operation=DeviceOperation.PERFORMANCE_METRIC,
            level=LogLevel.DEBUG,
            message=message,
            performance_metrics=performance_metrics,
            context=context or {}
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        summary = {
            "total_operations": len(self.structured_logs),
            "operations_by_type": {},
            "average_durations": {},
            "error_rate": {}
        }
        
        # Count operations by type
        for log_entry in self.structured_logs:
            op_type = log_entry.operation.value
            summary["operations_by_type"][op_type] = summary["operations_by_type"].get(op_type, 0) + 1
        
        # Calculate average durations
        for operation, durations in self.performance_history.items():
            if durations:
                summary["average_durations"][operation.value] = {
                    "mean_ms": np.mean(durations),
                    "median_ms": np.median(durations),
                    "min_ms": np.min(durations),
                    "max_ms": np.max(durations),
                    "std_ms": np.std(durations)
                }
        
        # Calculate error rates
        for op_type in summary["operations_by_type"]:
            total_ops = summary["operations_by_type"][op_type]
            error_ops = sum(1 for log in self.structured_logs 
                           if log.operation.value == op_type and log.level in [LogLevel.ERROR, LogLevel.CRITICAL])
            summary["error_rate"][op_type] = error_ops / total_ops if total_ops > 0 else 0.0
        
        return summary
    
    def export_logs(self, 
                   output_path: str,
                   format: str = "json",
                   filter_operations: Optional[List[DeviceOperation]] = None,
                   filter_level: Optional[LogLevel] = None):
        """
        Export structured logs to file.
        
        Args:
            output_path: Output file path
            format: Export format ("json", "csv")
            filter_operations: Filter by operation types
            filter_level: Filter by minimum log level
        """
        # Filter logs
        filtered_logs = self.structured_logs
        
        if filter_operations:
            filtered_logs = [log for log in filtered_logs if log.operation in filter_operations]
        
        if filter_level:
            level_order = {LogLevel.DEBUG: 0, LogLevel.INFO: 1, LogLevel.WARNING: 2, 
                          LogLevel.ERROR: 3, LogLevel.CRITICAL: 4}
            min_level = level_order[filter_level]
            filtered_logs = [log for log in filtered_logs if level_order[log.level] >= min_level]
        
        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump([log.to_dict() for log in filtered_logs], f, indent=2, default=str)
        elif format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame([log.to_dict() for log in filtered_logs])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(filtered_logs)} log entries to {output_path}")
    
    def clear_logs(self):
        """Clear all stored logs and performance history."""
        self.structured_logs.clear()
        self.performance_history.clear()
        self.operation_start_times.clear()


# Global device logger instances
_global_loggers: Dict[str, DeviceLogger] = {}


def get_device_logger(name: str = "default", **kwargs) -> DeviceLogger:
    """Get or create device logger instance."""
    if name not in _global_loggers:
        _global_loggers[name] = DeviceLogger(name=name, **kwargs)
    return _global_loggers[name]


def setup_rev_device_logging(log_file: Optional[str] = None,
                            log_level: str = "INFO",
                            enable_console: bool = True) -> DeviceLogger:
    """
    Setup REV device logging with sensible defaults.
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        enable_console: Enable console output
        
    Returns:
        Configured device logger
    """
    return get_device_logger(
        name="rev",
        log_file=log_file,
        log_level=log_level,
        enable_console=enable_console,
        enable_structured=True
    )