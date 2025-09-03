"""
Production-Ready Error Handling Infrastructure
Provides robust error handling, retry logic, and graceful degradation
"""

import time
import random
import logging
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import json
import sys

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class REVException(Exception):
    """Base exception for all REV errors"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc()
        }


class ModelLoadingError(REVException):
    """Error loading or initializing model"""
    pass


class InferenceError(REVException):
    """Error during model inference"""
    pass


class MemoryError(REVException):
    """Memory-related errors (OOM, allocation failures)"""
    pass


class NetworkError(REVException):
    """Network-related errors (API failures, timeouts)"""
    pass


class ValidationError(REVException):
    """Input validation errors"""
    pass


class ConfigurationError(REVException):
    """Configuration or setup errors"""
    pass


class ResourceError(REVException):
    """Resource availability errors (GPU, disk space)"""
    pass


class AuthenticationError(REVException):
    """Authentication/authorization errors"""
    pass


class RateLimitError(REVException):
    """Rate limiting errors"""
    pass


class DataCorruptionError(REVException):
    """Data integrity/corruption errors"""
    pass


# ============================================================================
# Retry Mechanisms
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: tuple = (Exception,)
    
    
def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(
        config.initial_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )
    
    if config.jitter:
        # Add randomization to prevent thundering herd
        delay = delay * (0.5 + random.random())
        
    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    on_failure: Optional[Callable[[Exception], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        config: Retry configuration
        on_retry: Callback when retrying (exception, attempt_number)
        on_failure: Callback when all retries exhausted
    """
    if config is None:
        config = RetryConfig()
        
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        if on_failure:
                            on_failure(e)
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise
                        
                    delay = calculate_backoff_delay(attempt, config)
                    
                    if on_retry:
                        on_retry(e, attempt)
                        
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    
            # Should never reach here
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker entering HALF_OPEN state for {func.__name__}")
                else:
                    raise REVException(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        error_code="CIRCUIT_OPEN",
                        details={"last_failure": self.last_failure_time}
                    )
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
            
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
        
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
            else:
                self.failure_count = 0
                
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker returning to OPEN state")
                
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
                
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time
            }


def with_circuit_breaker(
    breaker: Optional[CircuitBreaker] = None,
    **breaker_config
):
    """Decorator to wrap function with circuit breaker"""
    if breaker is None:
        config = CircuitBreakerConfig(**breaker_config)
        breaker = CircuitBreaker(config)
        
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Graceful Degradation
# ============================================================================

class FeatureFlag:
    """Feature flag for enabling/disabling features"""
    
    def __init__(self, name: str, default: bool = True):
        self.name = name
        self.enabled = default
        self._fallback_mode = False
        
    def is_enabled(self) -> bool:
        """Check if feature is enabled"""
        return self.enabled and not self._fallback_mode
        
    def disable(self):
        """Disable feature"""
        self.enabled = False
        logger.info(f"Feature '{self.name}' disabled")
        
    def enable(self):
        """Enable feature"""
        self.enabled = True
        logger.info(f"Feature '{self.name}' enabled")
        
    def enter_fallback_mode(self):
        """Enter fallback mode (temporary disable)"""
        self._fallback_mode = True
        logger.warning(f"Feature '{self.name}' entering fallback mode")
        
    def exit_fallback_mode(self):
        """Exit fallback mode"""
        self._fallback_mode = False
        logger.info(f"Feature '{self.name}' exiting fallback mode")


class GracefulDegradation:
    """
    Manages graceful degradation of features
    
    When optional features fail, the system continues with reduced functionality
    """
    
    def __init__(self):
        self.features: Dict[str, FeatureFlag] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degradation_history: List[Dict[str, Any]] = []
        
    def register_feature(
        self, 
        name: str, 
        default_enabled: bool = True,
        fallback_handler: Optional[Callable] = None
    ) -> FeatureFlag:
        """Register a feature that can be degraded"""
        feature = FeatureFlag(name, default_enabled)
        self.features[name] = feature
        
        if fallback_handler:
            self.fallback_handlers[name] = fallback_handler
            
        return feature
        
    def degrade_feature(self, name: str, error: Exception = None):
        """Degrade a feature due to error"""
        if name not in self.features:
            logger.warning(f"Unknown feature '{name}' for degradation")
            return
            
        feature = self.features[name]
        feature.enter_fallback_mode()
        
        # Record degradation
        self.degradation_history.append({
            'feature': name,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(error) if error else None
        })
        
        # Execute fallback handler if available
        if name in self.fallback_handlers:
            try:
                self.fallback_handlers[name](error)
            except Exception as e:
                logger.error(f"Fallback handler failed for '{name}': {e}")
                
    def restore_feature(self, name: str):
        """Restore a degraded feature"""
        if name in self.features:
            self.features[name].exit_fallback_mode()
            
    def is_degraded(self, name: str) -> bool:
        """Check if feature is degraded"""
        if name not in self.features:
            return False
        return self.features[name]._fallback_mode
        
    def get_status(self) -> Dict[str, Any]:
        """Get degradation status"""
        return {
            'features': {
                name: {
                    'enabled': feature.enabled,
                    'degraded': feature._fallback_mode
                }
                for name, feature in self.features.items()
            },
            'degradation_count': len(self.degradation_history),
            'recent_degradations': self.degradation_history[-10:]
        }


def with_graceful_degradation(
    feature_name: str,
    degradation_manager: GracefulDegradation,
    fallback_value: Any = None
):
    """
    Decorator for graceful degradation
    
    If the function fails, the feature is degraded and fallback value returned
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if feature is already degraded
            if degradation_manager.is_degraded(feature_name):
                logger.debug(f"Feature '{feature_name}' is degraded, returning fallback")
                return fallback_value
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Feature '{feature_name}' failed: {e}")
                degradation_manager.degrade_feature(feature_name, e)
                return fallback_value
                
        return wrapper
    return decorator


# ============================================================================
# Error Recovery Strategies
# ============================================================================

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    COMPENSATE = "compensate"


class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self):
        self.strategies: Dict[Type[Exception], RecoveryStrategy] = {
            NetworkError: RecoveryStrategy.RETRY,
            MemoryError: RecoveryStrategy.DEGRADE,
            ValidationError: RecoveryStrategy.FAIL_FAST,
            RateLimitError: RecoveryStrategy.RETRY,
            AuthenticationError: RecoveryStrategy.FAIL_FAST
        }
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        
    def register_strategy(
        self, 
        exception_type: Type[Exception], 
        strategy: RecoveryStrategy
    ):
        """Register recovery strategy for exception type"""
        self.strategies[exception_type] = strategy
        
    def register_handler(
        self,
        strategy: RecoveryStrategy,
        handler: Callable
    ):
        """Register handler for recovery strategy"""
        self.recovery_handlers[strategy] = handler
        
    def recover(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        """Execute recovery strategy for error"""
        # Find matching strategy
        strategy = None
        for exc_type, strat in self.strategies.items():
            if isinstance(error, exc_type):
                strategy = strat
                break
                
        if strategy is None:
            strategy = RecoveryStrategy.FAIL_FAST
            
        logger.info(f"Applying {strategy.value} strategy for {type(error).__name__}")
        
        # Execute recovery handler
        if strategy in self.recovery_handlers:
            try:
                return self.recovery_handlers[strategy](error, context)
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")
                raise error
        else:
            # Default behaviors
            if strategy == RecoveryStrategy.RETRY:
                raise error  # Let retry decorator handle
            elif strategy == RecoveryStrategy.FAIL_FAST:
                raise error
            else:
                logger.warning(f"No handler for strategy {strategy.value}")
                raise error


# ============================================================================
# Global Error Context
# ============================================================================

class ErrorContext:
    """Global error context for tracking errors across the application"""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
        self._lock = threading.Lock()
        
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error occurrence"""
        with self._lock:
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            error_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'error_type': error_type,
                'message': str(error),
                'context': context or {}
            }
            
            if isinstance(error, REVException):
                error_record.update(error.to_dict())
                
            self.errors.append(error_record)
            
            # Keep only last 1000 errors
            if len(self.errors) > 1000:
                self.errors = self.errors[-1000:]
                
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._lock:
            uptime = time.time() - self.start_time
            total_errors = sum(self.error_counts.values())
            
            return {
                'uptime_seconds': uptime,
                'total_errors': total_errors,
                'error_rate': total_errors / max(uptime, 1),
                'error_counts': dict(self.error_counts),
                'recent_errors': self.errors[-10:]
            }
            
    def reset(self):
        """Reset error tracking"""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()
            self.start_time = time.time()


# Global instances
error_context = ErrorContext()
degradation_manager = GracefulDegradation()
recovery_manager = ErrorRecoveryManager()


# ============================================================================
# Utility Functions
# ============================================================================

def handle_exception(
    error: Exception,
    context: Dict[str, Any] = None,
    reraise: bool = True
) -> Optional[Any]:
    """
    Central exception handler
    
    Args:
        error: The exception to handle
        context: Additional context
        reraise: Whether to re-raise the exception
    """
    # Record error
    error_context.record_error(error, context)
    
    # Log error with appropriate level
    if isinstance(error, ValidationError):
        logger.warning(f"Validation error: {error}")
    elif isinstance(error, (NetworkError, RateLimitError)):
        logger.warning(f"Recoverable error: {error}")
    else:
        logger.error(f"Error occurred: {error}", exc_info=True)
        
    # Try recovery
    try:
        return recovery_manager.recover(error, context)
    except Exception:
        if reraise:
            raise error
        return None


def safe_execute(
    func: Callable,
    *args,
    default_value: Any = None,
    error_handler: Optional[Callable[[Exception], Any]] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        error_handler: Custom error handler
        *args, **kwargs: Arguments for function
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            try:
                return error_handler(e)
            except Exception:
                pass
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default_value