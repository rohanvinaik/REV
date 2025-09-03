"""
Structured Logging and Monitoring Configuration
Supports JSON output, ELK stack, Prometheus metrics, and OpenTelemetry tracing
"""

import logging
import json
import sys
import time
import os
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import socket
import traceback

# Monitoring libraries (optional imports)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


# ============================================================================
# Structured Logging
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            '@timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': self.process_id,
            'hostname': self.hostname,
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
            
        # Add exception info
        if record.exc_info and self.include_traceback:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_obj)


class ContextualLogger:
    """Logger with contextual information"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}
        self._local = threading.local()
        
    def add_context(self, **kwargs):
        """Add permanent context to all logs"""
        self.context.update(kwargs)
        
    @contextmanager
    def temporary_context(self, **kwargs):
        """Add temporary context for a block"""
        if not hasattr(self._local, 'context_stack'):
            self._local.context_stack = []
            
        self._local.context_stack.append(kwargs)
        try:
            yield
        finally:
            self._local.context_stack.pop()
            
    def _get_extra_fields(self) -> Dict[str, Any]:
        """Get all context fields"""
        fields = dict(self.context)
        
        if hasattr(self._local, 'context_stack'):
            for ctx in self._local.context_stack:
                fields.update(ctx)
                
        return fields
        
    def _log(self, level: str, msg: str, *args, **kwargs):
        """Internal log method"""
        extra = kwargs.get('extra', {})
        extra['extra_fields'] = self._get_extra_fields()
        extra['extra_fields'].update(kwargs.get('fields', {}))
        kwargs['extra'] = extra
        
        # Remove 'fields' from kwargs as it's not a valid logging parameter
        kwargs.pop('fields', None)
        
        getattr(self.logger, level)(msg, *args, **kwargs)
        
    def debug(self, msg: str, *args, **kwargs):
        self._log('debug', msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        self._log('info', msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        self._log('warning', msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        self._log('error', msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        self._log('critical', msg, *args, **kwargs)


# ============================================================================
# ELK Stack Integration
# ============================================================================

class LogstashHandler(logging.Handler):
    """Send logs to Logstash via TCP/UDP"""
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 5959,
        protocol: str = 'tcp'
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.protocol = protocol
        self.socket = None
        self._connect()
        
    def _connect(self):
        """Establish connection to Logstash"""
        try:
            if self.protocol == 'tcp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            self.socket = None
            print(f"Failed to connect to Logstash: {e}")
            
    def emit(self, record: logging.LogRecord):
        """Send log record to Logstash"""
        if not self.socket:
            self._connect()
            
        if self.socket:
            try:
                # Format as JSON
                formatter = StructuredFormatter()
                msg = formatter.format(record)
                
                # Send to Logstash
                if self.protocol == 'tcp':
                    self.socket.send((msg + '\n').encode('utf-8'))
                else:  # UDP
                    self.socket.sendto(
                        msg.encode('utf-8'), 
                        (self.host, self.port)
                    )
            except Exception as e:
                self.handleError(record)
                
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
        super().close()


# ============================================================================
# Prometheus Metrics
# ============================================================================

class MetricsCollector:
    """Collect and expose Prometheus metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if not PROMETHEUS_AVAILABLE:
            self.enabled = False
            return
            
        self.enabled = True
        self.registry = registry or CollectorRegistry()
        
        # Define metrics
        self.request_count = Counter(
            'rev_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'rev_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.model_inference_duration = Histogram(
            'rev_model_inference_seconds',
            'Model inference duration in seconds',
            ['model_name', 'model_family'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'rev_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # process, gpu, system
            registry=self.registry
        )
        
        self.active_models = Gauge(
            'rev_active_models',
            'Number of active models',
            registry=self.registry
        )
        
        self.error_count = Counter(
            'rev_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.fingerprint_similarity = Summary(
            'rev_fingerprint_similarity',
            'Fingerprint similarity scores',
            ['model_family'],
            registry=self.registry
        )
        
        self.circuit_breaker_state = Gauge(
            'rev_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['component'],
            registry=self.registry
        )
        
    @contextmanager
    def time_operation(self, metric: Histogram, **labels):
        """Context manager to time operations"""
        if not self.enabled:
            yield
            return
            
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            metric.labels(**labels).observe(duration)
            
    def increment_counter(self, metric: Counter, value: int = 1, **labels):
        """Increment a counter metric"""
        if self.enabled:
            metric.labels(**labels).inc(value)
            
    def set_gauge(self, metric: Gauge, value: float, **labels):
        """Set a gauge metric"""
        if self.enabled:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
                
    def observe_summary(self, metric: Summary, value: float, **labels):
        """Observe a summary metric"""
        if self.enabled:
            metric.labels(**labels).observe(value)
            
    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics in text format"""
        if self.enabled:
            return generate_latest(self.registry)
        return b""


# ============================================================================
# OpenTelemetry Tracing
# ============================================================================

class TracingManager:
    """Manage distributed tracing with OpenTelemetry"""
    
    def __init__(
        self,
        service_name: str = "rev-system",
        otlp_endpoint: str = "localhost:4317",
        enabled: bool = True
    ):
        if not OPENTELEMETRY_AVAILABLE or not enabled:
            self.enabled = False
            self.tracer = None
            return
            
        self.enabled = True
        
        # Configure resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })
        
        # Configure tracer provider
        provider = TracerProvider(resource=resource)
        
        # Configure exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Create a trace span for an operation"""
        if not self.enabled:
            yield None
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
                
            try:
                yield span
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise
                
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to current span"""
        if self.enabled:
            span = trace.get_current_span()
            if span:
                span.add_event(name, attributes=attributes or {})
                
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        if self.enabled:
            span = trace.get_current_span()
            if span:
                span.set_attribute(key, str(value))


# ============================================================================
# Performance Monitoring
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        
    @contextmanager
    def measure(self, operation: str, **metadata):
        """Measure operation performance"""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                metadata=metadata
            )
            
            self._record_metric(metric)
            
    def _record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            
            # Trim history
            if len(self.metrics) > self.max_history:
                self.metrics = self.metrics[-self.max_history:]
                
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if operation:
                metrics = [m for m in self.metrics if m.operation == operation]
            else:
                metrics = list(self.metrics)
                
        if not metrics:
            return {}
            
        durations = [m.duration for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        
        return {
            'count': len(metrics),
            'success_rate': success_count / len(metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'p50_duration': sorted(durations)[len(durations) // 2],
            'p95_duration': sorted(durations)[int(len(durations) * 0.95)],
            'p99_duration': sorted(durations)[int(len(durations) * 0.99)]
        }


# ============================================================================
# Logging Configuration
# ============================================================================

class LoggingConfig:
    """Central logging configuration"""
    
    def __init__(
        self,
        level: str = "INFO",
        json_output: bool = True,
        console_output: bool = True,
        file_output: Optional[str] = None,
        logstash_host: Optional[str] = None,
        logstash_port: int = 5959,
        enable_prometheus: bool = True,
        enable_tracing: bool = True,
        otlp_endpoint: str = "localhost:4317"
    ):
        self.level = getattr(logging, level.upper())
        self.json_output = json_output
        self.console_output = console_output
        self.file_output = file_output
        self.logstash_host = logstash_host
        self.logstash_port = logstash_port
        self.enable_prometheus = enable_prometheus
        self.enable_tracing = enable_tracing
        self.otlp_endpoint = otlp_endpoint
        
        # Initialize components
        self.metrics_collector = None
        self.tracing_manager = None
        self.performance_monitor = PerformanceMonitor()
        
    def setup(self, service_name: str = "rev-system"):
        """Setup logging configuration"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.json_output:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            root_logger.addHandler(console_handler)
            
        # File handler
        if self.file_output:
            file_handler = logging.FileHandler(self.file_output)
            if self.json_output:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            root_logger.addHandler(file_handler)
            
        # Logstash handler
        if self.logstash_host:
            logstash_handler = LogstashHandler(
                host=self.logstash_host,
                port=self.logstash_port
            )
            root_logger.addHandler(logstash_handler)
            
        # Setup metrics collector
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self.metrics_collector = MetricsCollector()
            
        # Setup tracing
        if self.enable_tracing and OPENTELEMETRY_AVAILABLE:
            self.tracing_manager = TracingManager(
                service_name=service_name,
                otlp_endpoint=self.otlp_endpoint
            )
            
        return self
        
    def get_logger(self, name: str) -> ContextualLogger:
        """Get a contextual logger"""
        return ContextualLogger(logging.getLogger(name))
        
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics"""
        if self.metrics_collector:
            return self.metrics_collector.get_metrics()
        return b""


# ============================================================================
# Global Configuration
# ============================================================================

# Default configuration
default_config = LoggingConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_JSON", "true").lower() == "true",
    console_output=True,
    file_output=os.getenv("LOG_FILE"),
    logstash_host=os.getenv("LOGSTASH_HOST"),
    enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
    enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317")
)

# Setup on import
if os.getenv("AUTO_SETUP_LOGGING", "true").lower() == "true":
    default_config.setup()


def get_logger(name: str) -> ContextualLogger:
    """Get a logger with context support"""
    return default_config.get_logger(name)