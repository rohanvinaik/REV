#!/usr/bin/env python3
"""
Production Deployment Example

Demonstrates how to deploy REV in production with monitoring, scaling, and reliability.
"""

import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

sys.path.append(str(Path(__file__).parent.parent))

# Import production components
from src.api.rest_service import app, ModelAnalysisRequest
from src.utils.error_handling import CircuitBreaker, GracefulDegradation
from src.utils.logging_config import LoggingConfig, MetricsCollector


def setup_production_logging():
    """
    Setup production-grade logging with structured output.
    """
    print("=" * 70)
    print("PRODUCTION LOGGING SETUP")
    print("=" * 70)
    
    # Configure structured JSON logging
    config = LoggingConfig(
        log_level="INFO",
        json_output=True,
        log_file="logs/rev_production.log",
        enable_rotation=True,
        max_bytes=10485760,  # 10MB
        backup_count=5
    )
    
    config.setup()
    logger = config.get_logger("production_demo")
    
    print("\n✅ Logging configured:")
    print(f"  Format: JSON structured")
    print(f"  Level: INFO")
    print(f"  File: logs/rev_production.log")
    print(f"  Rotation: 10MB max, 5 backups")
    
    # Example log entries
    logger.info("Production system started", fields={
        "version": "3.0.0",
        "environment": "production",
        "features": ["prompt_orchestration", "principled_features", "unified_fingerprints"]
    })
    
    logger.warning("High memory usage detected", fields={
        "memory_percent": 85.5,
        "action": "reducing_segment_size"
    })
    
    return logger


def demonstrate_circuit_breaker():
    """
    Demonstrate circuit breaker for fault tolerance.
    """
    print("\n" + "=" * 70)
    print("CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 70)
    
    # Configure circuit breaker
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=10.0,
        success_threshold=2
    )
    
    print("\n⚡ Circuit Breaker Configuration:")
    print(f"  Failure Threshold: 3 failures to open")
    print(f"  Recovery Timeout: 10 seconds")
    print(f"  Success Threshold: 2 successes to close")
    
    # Simulate API calls
    def unreliable_api_call(fail_rate=0.5):
        """Simulate an unreliable API."""
        import random
        if random.random() < fail_rate:
            raise Exception("API call failed")
        return {"status": "success"}
    
    print("\n🔄 Testing Circuit Breaker:")
    print("-" * 50)
    
    for i in range(10):
        try:
            result = breaker.call(unreliable_api_call, fail_rate=0.7)
            print(f"  Call {i+1}: ✅ Success - {breaker.state.value} state")
        except Exception as e:
            print(f"  Call {i+1}: ❌ Failed - {breaker.state.value} state - {str(e)}")
        
        time.sleep(0.5)
    
    print(f"\n📊 Final Statistics:")
    print(f"  State: {breaker.state.value}")
    print(f"  Failures: {breaker.failure_count}")
    print(f"  Success Rate: {(1 - breaker.failure_count/10)*100:.1f}%")


def demonstrate_graceful_degradation():
    """
    Demonstrate graceful feature degradation under load.
    """
    print("\n" + "=" * 70)
    print("GRACEFUL DEGRADATION DEMONSTRATION")
    print("=" * 70)
    
    degradation = GracefulDegradation()
    
    # Register features
    features = [
        "prompt_orchestration",
        "principled_features",
        "unified_fingerprints",
        "adversarial_testing",
        "deep_analysis"
    ]
    
    print("\n📋 Registered Features:")
    for feature in features:
        f = degradation.register_feature(feature)
        print(f"  {feature}: {'Enabled' if f.is_enabled() else 'Disabled'}")
    
    # Simulate system under stress
    print("\n🔥 Simulating System Stress:")
    print("-" * 50)
    
    stress_events = [
        ("High memory usage", ["deep_analysis", "adversarial_testing"]),
        ("API rate limit", ["prompt_orchestration"]),
        ("CPU overload", ["principled_features"])
    ]
    
    for event, affected in stress_events:
        print(f"\n  Event: {event}")
        for feature in affected:
            degradation.degrade_feature(feature, Exception(event))
            print(f"    Degraded: {feature}")
    
    print("\n📊 Feature Status After Stress:")
    for feature in features:
        status = "Enabled" if not degradation.is_degraded(feature) else "Degraded"
        print(f"  {feature}: {status}")
    
    # Recovery
    print("\n🔄 Recovery Phase:")
    time.sleep(1)
    
    for feature in features:
        if degradation.is_degraded(feature):
            degradation.restore_feature(feature)
            print(f"  Restored: {feature}")


def demonstrate_metrics_collection():
    """
    Demonstrate metrics collection for monitoring.
    """
    print("\n" + "=" * 70)
    print("METRICS COLLECTION DEMONSTRATION")
    print("=" * 70)
    
    collector = MetricsCollector(namespace="rev_demo")
    
    print("\n📊 Available Metrics:")
    print("  - Request count and latency")
    print("  - Error rates by type")
    print("  - Resource utilization")
    print("  - Model inference performance")
    
    # Simulate metric collection
    print("\n🔄 Simulating Operations:")
    print("-" * 50)
    
    import random
    
    for i in range(10):
        # Simulate API request
        latency = random.uniform(0.1, 2.0)
        collector.observe_latency("api_request", latency)
        
        # Random success/failure
        if random.random() > 0.8:
            collector.increment_counter("errors_total", labels={"type": "timeout"})
            print(f"  Request {i+1}: ❌ Timeout ({latency:.2f}s)")
        else:
            collector.increment_counter("requests_total")
            print(f"  Request {i+1}: ✅ Success ({latency:.2f}s)")
        
        # Memory gauge
        memory_gb = random.uniform(2.0, 4.0)
        collector.set_gauge("memory_usage_gb", memory_gb)
    
    # Get metrics
    metrics = collector.get_metrics()
    
    print("\n📈 Collected Metrics:")
    print(f"  Total Requests: {metrics.get('requests_total', 0)}")
    print(f"  Total Errors: {metrics.get('errors_total', 0)}")
    print(f"  Average Latency: {metrics.get('api_request_latency_avg', 0):.2f}s")
    print(f"  Current Memory: {metrics.get('memory_usage_gb', 0):.1f}GB")


def demonstrate_health_checks():
    """
    Demonstrate health check system.
    """
    print("\n" + "=" * 70)
    print("HEALTH CHECK SYSTEM")
    print("=" * 70)
    
    class HealthChecker:
        def __init__(self):
            self.checks = {}
        
        def register_check(self, name: str, check_fn):
            self.checks[name] = check_fn
        
        def run_checks(self) -> Dict:
            results = {
                "status": "healthy",
                "checks": {},
                "timestamp": time.time()
            }
            
            for name, check_fn in self.checks.items():
                try:
                    result = check_fn()
                    results["checks"][name] = {
                        "status": "pass" if result else "fail",
                        "message": "OK" if result else "Check failed"
                    }
                    if not result:
                        results["status"] = "degraded"
                except Exception as e:
                    results["checks"][name] = {
                        "status": "error",
                        "message": str(e)
                    }
                    results["status"] = "unhealthy"
            
            return results
    
    # Create health checker
    checker = HealthChecker()
    
    # Register checks
    checker.register_check("database", lambda: True)  # Mock DB check
    checker.register_check("redis", lambda: True)     # Mock Redis check
    checker.register_check("disk_space", lambda: True)  # Mock disk check
    checker.register_check("memory", lambda: False)   # Mock memory issue
    
    print("\n🏥 Running Health Checks:")
    print("-" * 50)
    
    results = checker.run_checks()
    
    # Display results
    status_emoji = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌"
    }
    
    print(f"\nOverall Status: {status_emoji[results['status']]} {results['status'].upper()}")
    
    print("\nIndividual Checks:")
    for check_name, check_result in results["checks"].items():
        status = check_result["status"]
        emoji = "✅" if status == "pass" else "⚠️" if status == "fail" else "❌"
        print(f"  {check_name}: {emoji} {status} - {check_result['message']}")


def demonstrate_load_balancing():
    """
    Demonstrate load balancing strategies.
    """
    print("\n" + "=" * 70)
    print("LOAD BALANCING STRATEGIES")
    print("=" * 70)
    
    class LoadBalancer:
        def __init__(self, instances: List[str]):
            self.instances = instances
            self.current = 0
            self.weights = {inst: 1.0 for inst in instances}
            self.request_counts = {inst: 0 for inst in instances}
        
        def round_robin(self) -> str:
            """Simple round-robin selection."""
            instance = self.instances[self.current]
            self.current = (self.current + 1) % len(self.instances)
            self.request_counts[instance] += 1
            return instance
        
        def weighted(self) -> str:
            """Weighted selection based on capacity."""
            import random
            total = sum(self.weights.values())
            r = random.uniform(0, total)
            upto = 0
            for instance, weight in self.weights.items():
                if upto + weight >= r:
                    self.request_counts[instance] += 1
                    return instance
                upto += weight
        
        def least_connections(self) -> str:
            """Select instance with fewest connections."""
            instance = min(self.request_counts, key=self.request_counts.get)
            self.request_counts[instance] += 1
            return instance
    
    # Create load balancer
    instances = ["rev-api-1", "rev-api-2", "rev-api-3"]
    lb = LoadBalancer(instances)
    
    # Set different weights (capacity)
    lb.weights = {
        "rev-api-1": 3.0,  # High capacity
        "rev-api-2": 2.0,  # Medium capacity
        "rev-api-3": 1.0   # Low capacity
    }
    
    print("\n🔄 Load Balancing Strategies:")
    print("-" * 50)
    
    # Test round-robin
    print("\n1. Round-Robin:")
    for i in range(6):
        instance = lb.round_robin()
        print(f"   Request {i+1} -> {instance}")
    
    # Reset counts
    lb.request_counts = {inst: 0 for inst in instances}
    
    # Test weighted
    print("\n2. Weighted (by capacity):")
    for i in range(6):
        instance = lb.weighted()
        print(f"   Request {i+1} -> {instance}")
    
    print("\n   Distribution:")
    for inst, count in lb.request_counts.items():
        weight = lb.weights[inst]
        print(f"   {inst}: {count} requests (weight: {weight})")
    
    # Test least connections
    print("\n3. Least Connections:")
    lb.request_counts = {"rev-api-1": 5, "rev-api-2": 3, "rev-api-3": 1}
    print("   Current connections: rev-api-1:5, rev-api-2:3, rev-api-3:1")
    
    for i in range(3):
        instance = lb.least_connections()
        print(f"   Request {i+1} -> {instance}")


def create_docker_compose_snippet():
    """
    Generate Docker Compose configuration for production.
    """
    print("\n" + "=" * 70)
    print("DOCKER COMPOSE CONFIGURATION")
    print("=" * 70)
    
    config = """
version: '3.8'

services:
  # REV API Service (scaled)
  rev-api:
    image: rev-system:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      - LOG_LEVEL=INFO
      - MEMORY_LIMIT_GB=4.0
      - ENABLE_METRICS=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rev-network

  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - rev-api
    networks:
      - rev-network

  # Monitoring Stack
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - rev-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    networks:
      - rev-network

networks:
  rev-network:
    driver: overlay
    """
    
    print("\n📝 Docker Compose Configuration:")
    print(config)
    
    print("\n💡 Usage:")
    print("  1. Save configuration to docker-compose.yml")
    print("  2. Run: docker-compose up -d")
    print("  3. Scale: docker-compose up -d --scale rev-api=5")
    print("  4. Monitor: http://localhost:3000 (Grafana)")


def main():
    """Main function."""
    
    print("=" * 70)
    print("REV PRODUCTION DEPLOYMENT EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate production-ready features:\n")
    
    # Run demonstrations
    logger = setup_production_logging()
    demonstrate_circuit_breaker()
    demonstrate_graceful_degradation()
    demonstrate_metrics_collection()
    demonstrate_health_checks()
    demonstrate_load_balancing()
    create_docker_compose_snippet()
    
    print("\n" + "=" * 70)
    print("Production deployment examples completed!")
    print("=" * 70)
    
    print("\n📚 Next Steps:")
    print("  1. Review DEPLOYMENT.md for complete setup")
    print("  2. Configure monitoring dashboards")
    print("  3. Set up alerting rules")
    print("  4. Test disaster recovery procedures")
    print("  5. Run load tests before production")


if __name__ == "__main__":
    main()