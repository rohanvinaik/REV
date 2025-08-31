#!/usr/bin/env python3
"""
Comprehensive monitoring and diagnostics for behavioral probing in REV system.
Provides detailed logging, failure analysis, and diagnostic reporting.
"""

import logging
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class ProbeExecutionRecord:
    """Record of a single probe execution attempt."""
    timestamp: str
    probe_text: str
    layer_idx: int
    execution_time_ms: float
    device: str
    dtype: str
    success: bool
    error_message: str = ""
    divergence_score: float = 0.0
    tensor_shapes: Dict[str, str] = None
    memory_usage_mb: float = 0.0
    
    def __post_init__(self):
        if self.tensor_shapes is None:
            self.tensor_shapes = {}


@dataclass 
class FallbackEvent:
    """Record of a fallback to hardcoded sites."""
    timestamp: str
    reason: str
    layer_idx: int
    probe_count: int
    failed_probes: List[str]
    

class ProbeMonitor:
    """Comprehensive monitoring system for behavioral probe execution."""
    
    def __init__(self, log_file: str = "probe_execution.log", enable_html_reports: bool = True):
        self.log_file = log_file
        self.enable_html_reports = enable_html_reports
        self.executions: List[ProbeExecutionRecord] = []
        self.fallback_events: List[FallbackEvent] = []
        self.fallback_count = 0
        self.success_count = 0
        self.start_time = datetime.now()
        self.setup_logging()
        
        # Statistics tracking
        self.layer_stats: Dict[int, Dict[str, Any]] = {}
        self.device_stats: Dict[str, int] = {}
        self.probe_type_stats: Dict[str, Dict[str, Any]] = {}
        
    def setup_logging(self):
        """Configure logging with detailed format."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logger
        logger = logging.getLogger("ProbeMonitor")
        logger.setLevel(logging.DEBUG)
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_dir / self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        
    def log_probe_execution(self, record: ProbeExecutionRecord):
        """Log probe execution with full details."""
        self.executions.append(record)
        self._update_statistics(record)
        
        if record.success:
            self.success_count += 1
            self.logger.info(
                f"✓ PROBE SUCCESS: Layer {record.layer_idx:2d} | "
                f"Divergence: {record.divergence_score:.3f} | "
                f"Time: {record.execution_time_ms:5.1f}ms | "
                f"Device: {record.device} | "
                f"Probe: {record.probe_text[:30]}..."
            )
        else:
            self.logger.error(
                f"✗ PROBE FAILED: Layer {record.layer_idx:2d} | "
                f"Time: {record.execution_time_ms:5.1f}ms | "
                f"Device: {record.device} | "
                f"Error: {record.error_message} | "
                f"Probe: {record.probe_text[:30]}..."
            )
            
        # Log tensor shape information if available
        if record.tensor_shapes:
            self.logger.debug(f"  Tensor shapes: {record.tensor_shapes}")
            
        # Log memory usage if available
        if record.memory_usage_mb > 0:
            self.logger.debug(f"  Memory usage: {record.memory_usage_mb:.1f}MB")
    
    def log_fallback(self, reason: str, layer_idx: int = -1, probe_count: int = 0, failed_probes: List[str] = None):
        """Log when system falls back to hardcoded sites."""
        self.fallback_count += 1
        
        fallback_event = FallbackEvent(
            timestamp=datetime.now().isoformat(),
            reason=reason,
            layer_idx=layer_idx,
            probe_count=probe_count,
            failed_probes=failed_probes or []
        )
        
        self.fallback_events.append(fallback_event)
        
        self.logger.warning("⚠️" * 10)
        self.logger.warning(f"⚠️  FALLBACK TO HARDCODED SITES #{self.fallback_count}")
        self.logger.warning(f"⚠️  Reason: {reason}")
        if layer_idx >= 0:
            self.logger.warning(f"⚠️  Layer: {layer_idx}")
        if probe_count > 0:
            self.logger.warning(f"⚠️  Failed probes: {probe_count}")
        self.logger.warning("⚠️" * 10)
        
        # Generate immediate diagnostic report on fallback
        self._generate_fallback_diagnostics(fallback_event)
    
    def log_behavioral_analysis_start(self, model_info: Dict[str, Any]):
        """Log start of behavioral analysis with system info."""
        self.logger.info("🔬" * 20)
        self.logger.info("🔬 BEHAVIORAL ANALYSIS STARTED")
        self.logger.info("🔬" * 20)
        self.logger.info(f"Model: {model_info.get('name', 'Unknown')}")
        self.logger.info(f"Layers: {model_info.get('num_layers', 'Unknown')}")
        self.logger.info(f"Parameters: {model_info.get('num_parameters', 'Unknown')}")
        self.logger.info(f"Device: {model_info.get('device', 'Unknown')}")
        self.logger.info(f"Dtype: {model_info.get('dtype', 'Unknown')}")
        self.logger.info("🔬" * 20)
        
    def log_restriction_sites_discovered(self, sites: List[Any]):
        """Log discovered restriction sites."""
        if not sites:
            self.logger.warning("⚠️  NO RESTRICTION SITES DISCOVERED!")
            return
            
        self.logger.info(f"🎯 DISCOVERED {len(sites)} RESTRICTION SITES:")
        for i, site in enumerate(sites):
            divergence = getattr(site, 'behavioral_divergence', 0.0)
            layer = getattr(site, 'layer_idx', -1)
            site_type = getattr(site, 'site_type', 'unknown')
            confidence = getattr(site, 'confidence_score', 0.0)
            
            self.logger.info(
                f"  Site {i+1:2d}: Layer {layer:2d} | "
                f"Divergence: {divergence:.3f} | "
                f"Type: {site_type} | "
                f"Confidence: {confidence:.3f}"
            )
    
    def _update_statistics(self, record: ProbeExecutionRecord):
        """Update internal statistics."""
        # Layer statistics
        if record.layer_idx not in self.layer_stats:
            self.layer_stats[record.layer_idx] = {
                'executions': 0,
                'successes': 0,
                'total_time_ms': 0.0,
                'divergence_scores': []
            }
        
        stats = self.layer_stats[record.layer_idx]
        stats['executions'] += 1
        stats['total_time_ms'] += record.execution_time_ms
        
        if record.success:
            stats['successes'] += 1
            if record.divergence_score > 0:
                stats['divergence_scores'].append(record.divergence_score)
        
        # Device statistics
        self.device_stats[record.device] = self.device_stats.get(record.device, 0) + 1
        
        # Probe type statistics
        probe_type = self._extract_probe_type(record.probe_text)
        if probe_type not in self.probe_type_stats:
            self.probe_type_stats[probe_type] = {
                'count': 0,
                'success_rate': 0.0,
                'avg_divergence': 0.0,
                'divergences': []
            }
            
        type_stats = self.probe_type_stats[probe_type]
        type_stats['count'] += 1
        if record.success and record.divergence_score > 0:
            type_stats['divergences'].append(record.divergence_score)
    
    def _extract_probe_type(self, probe_text: str) -> str:
        """Extract probe type from probe text."""
        probe_lower = probe_text.lower()
        
        if any(word in probe_lower for word in ['calculate', 'compute', 'solve', '+', '-', '*', '/']):
            return 'mathematical'
        elif any(word in probe_lower for word in ['explain', 'describe', 'what', 'why', 'how']):
            return 'reasoning'
        elif any(word in probe_lower for word in ['translate', 'language', 'grammar', 'word']):
            return 'linguistic'
        elif any(word in probe_lower for word in ['remember', 'recall', 'memory', 'previous']):
            return 'memory'
        elif any(word in probe_lower for word in ['create', 'write', 'story', 'poem', 'creative']):
            return 'creative'
        else:
            return 'general'
    
    def _generate_fallback_diagnostics(self, fallback_event: FallbackEvent):
        """Generate diagnostic information when fallback occurs."""
        try:
            # Create diagnostics directory
            diag_dir = Path("diagnostics")
            diag_dir.mkdir(exist_ok=True)
            
            # Save fallback diagnostic
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            diag_file = diag_dir / f"fallback_diagnostic_{timestamp}.json"
            
            diagnostic_data = {
                "fallback_event": asdict(fallback_event),
                "recent_executions": [asdict(e) for e in self.executions[-10:]],
                "system_stats": self.generate_summary_stats(),
                "recommendations": self._generate_recommendations()
            }
            
            with open(diag_file, 'w') as f:
                json.dump(diagnostic_data, f, indent=2)
                
            self.logger.warning(f"⚠️  Fallback diagnostic saved to: {diag_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate fallback diagnostics: {e}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on execution patterns."""
        recommendations = []
        
        if len(self.executions) == 0:
            recommendations.append("No probe executions recorded - check if probes are being generated")
            return recommendations
        
        # Check success rate
        success_rate = self.success_count / len(self.executions)
        if success_rate < 0.5:
            recommendations.append(f"Low success rate ({success_rate:.1%}) - check device/dtype compatibility")
        
        # Check for device issues
        failed_executions = [e for e in self.executions if not e.success]
        device_errors = [e for e in failed_executions if 'device' in e.error_message.lower()]
        if len(device_errors) > 0:
            recommendations.append("Device-related errors detected - verify CUDA/MPS compatibility")
        
        # Check for tensor shape issues
        shape_errors = [e for e in failed_executions if 'shape' in e.error_message.lower()]
        if len(shape_errors) > 0:
            recommendations.append("Tensor shape errors detected - check model input dimensions")
        
        # Check execution times
        if len(self.executions) > 0:
            avg_time = np.mean([e.execution_time_ms for e in self.executions])
            if avg_time > 1000:  # > 1 second
                recommendations.append(f"Slow execution times ({avg_time:.0f}ms avg) - consider model optimization")
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        execution_times = [e.execution_time_ms for e in self.executions if e.success]
        divergence_scores = [e.divergence_score for e in self.executions if e.success and e.divergence_score > 0]
        
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "session_duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "total_executions": len(self.executions),
                "successful_executions": self.success_count,
                "failed_executions": len(self.executions) - self.success_count,
                "success_rate": self.success_count / len(self.executions) if self.executions else 0,
                "fallback_count": self.fallback_count,
                "using_behavioral_probing": self.fallback_count == 0 and self.success_count > 0
            },
            "performance": {
                "average_execution_time_ms": np.mean(execution_times) if execution_times else 0,
                "median_execution_time_ms": np.median(execution_times) if execution_times else 0,
                "max_execution_time_ms": max(execution_times) if execution_times else 0,
                "min_execution_time_ms": min(execution_times) if execution_times else 0
            },
            "divergence_analysis": {
                "total_scores": len(divergence_scores),
                "mean_divergence": np.mean(divergence_scores) if divergence_scores else 0,
                "std_divergence": np.std(divergence_scores) if divergence_scores else 0,
                "min_divergence": min(divergence_scores) if divergence_scores else 0,
                "max_divergence": max(divergence_scores) if divergence_scores else 0,
                "divergence_distribution": {
                    "low_count": len([d for d in divergence_scores if d < 0.3]),
                    "medium_count": len([d for d in divergence_scores if 0.3 <= d < 0.7]),
                    "high_count": len([d for d in divergence_scores if d >= 0.7])
                }
            },
            "layer_statistics": self._compute_layer_statistics(),
            "device_statistics": self.device_stats,
            "probe_type_statistics": self._compute_probe_type_statistics(),
            "fallback_events": [asdict(event) for event in self.fallback_events],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate quick summary statistics."""
        return {
            "total_executions": len(self.executions),
            "success_count": self.success_count,
            "fallback_count": self.fallback_count,
            "success_rate": self.success_count / len(self.executions) if self.executions else 0,
            "is_using_behavioral_probing": self.fallback_count == 0 and self.success_count > 0
        }
    
    def _compute_layer_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Compute detailed layer-wise statistics."""
        layer_stats = {}
        
        for layer_idx, stats in self.layer_stats.items():
            success_rate = stats['successes'] / stats['executions'] if stats['executions'] > 0 else 0
            avg_time = stats['total_time_ms'] / stats['executions'] if stats['executions'] > 0 else 0
            avg_divergence = np.mean(stats['divergence_scores']) if stats['divergence_scores'] else 0
            
            layer_stats[layer_idx] = {
                "executions": stats['executions'],
                "success_rate": success_rate,
                "average_time_ms": avg_time,
                "average_divergence": avg_divergence,
                "divergence_variance": np.var(stats['divergence_scores']) if len(stats['divergence_scores']) > 1 else 0
            }
            
        return layer_stats
    
    def _compute_probe_type_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute probe type statistics."""
        type_stats = {}
        
        for probe_type, stats in self.probe_type_stats.items():
            avg_divergence = np.mean(stats['divergences']) if stats['divergences'] else 0
            
            type_stats[probe_type] = {
                "count": stats['count'],
                "average_divergence": avg_divergence,
                "divergence_std": np.std(stats['divergences']) if len(stats['divergences']) > 1 else 0
            }
            
        return type_stats
    
    def save_report(self, filename: str = None) -> str:
        """Save diagnostic report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"probe_diagnostic_report_{timestamp}.json"
        
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / filename
        report = self.generate_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Diagnostic report saved to: {report_path}")
        
        # Generate HTML report if enabled
        if self.enable_html_reports:
            html_path = report_path.with_suffix('.html')
            self._generate_html_report(report, html_path)
            self.logger.info(f"📊 HTML report saved to: {html_path}")
        
        return str(report_path)
    
    def _generate_html_report(self, report: Dict[str, Any], html_path: Path):
        """Generate HTML visualization report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>REV Behavioral Probing Diagnostic Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                .stat {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>REV Behavioral Probing Diagnostic Report</h1>
                <p>Generated: {report['summary']['timestamp']}</p>
                <p class="{'success' if report['summary']['using_behavioral_probing'] else 'error'}">
                    Status: {'✓ Using Behavioral Probing' if report['summary']['using_behavioral_probing'] else '✗ Not Using Behavioral Probing'}
                </p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="stat">Total Executions: {report['summary']['total_executions']}</div>
                <div class="stat">Success Rate: {report['summary']['success_rate']:.1%}</div>
                <div class="stat">Fallback Count: {report['summary']['fallback_count']}</div>
                <div class="stat">Session Duration: {report['summary']['session_duration_minutes']:.1f} minutes</div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="stat">Average Execution Time: {report['performance']['average_execution_time_ms']:.1f}ms</div>
                <div class="stat">Median Execution Time: {report['performance']['median_execution_time_ms']:.1f}ms</div>
            </div>
            
            <div class="section">
                <h2>Divergence Analysis</h2>
                <div class="stat">Mean Divergence: {report['divergence_analysis']['mean_divergence']:.3f}</div>
                <div class="stat">Standard Deviation: {report['divergence_analysis']['std_divergence']:.3f}</div>
                <div class="stat">Divergence Range: {report['divergence_analysis']['min_divergence']:.3f} - {report['divergence_analysis']['max_divergence']:.3f}</div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)


# Global monitor instance for easy access
_global_monitor = None

def get_probe_monitor() -> ProbeMonitor:
    """Get global probe monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ProbeMonitor()
    return _global_monitor

def reset_probe_monitor():
    """Reset global probe monitor."""
    global _global_monitor
    _global_monitor = None