# REV Production Deployment Guide

## 🚀 Overview

This guide covers deploying the REV (Restriction Enzyme Verification) System v3.0 in production environments with enterprise-grade reliability, monitoring, and scalability.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [API Documentation](#api-documentation)
7. [Monitoring & Observability](#monitoring--observability)
8. [Error Recovery](#error-recovery)
9. [Security](#security)
10. [Performance Tuning](#performance-tuning)
11. [Troubleshooting](#troubleshooting)

## 🏃 Quick Start

### Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 16GB+ RAM minimum (64GB+ recommended)
- 500GB+ storage for models

### Rapid Deployment

```bash
# Clone repository
git clone https://github.com/rohanvinaik/REV.git
cd REV

# Install production dependencies
pip install -r requirements.txt -r requirements-prod.txt

# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# Verify deployment
curl http://localhost:8000/health
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                │
└─────────────┬───────────────────────┬──────────────────┘
              │                       │
    ┌─────────▼──────────┐  ┌────────▼──────────┐
    │   REV API Service   │  │  WebSocket Server │
    │   (FastAPI/REST)    │  │  (Real-time)      │
    └─────────┬──────────┘  └────────┬──────────┘
              │                       │
    ┌─────────▼───────────────────────▼──────────┐
    │          Application Layer                  │
    │  ┌──────────────┬──────────┬────────────┐ │
    │  │ Error Recovery│ Feature  │ Orchestration││
    │  │    System    │Extraction│   Engine    │ │
    │  └──────────────┴──────────┴────────────┘ │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │             Data Layer                      │
    │  ┌─────────┬──────────┬──────────┬───────┐│
    │  │  Redis  │PostgreSQL│Elasticsearch│MLflow││
    │  │ (Cache) │(Metadata)│  (Logs)   │(Exp)  ││
    │  └─────────┴──────────┴──────────┴───────┘│
    └──────────────────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │         Monitoring & Observability          │
    │  ┌──────────┬─────────┬─────────┬────────┐│
    │  │Prometheus│ Grafana │ Jaeger  │ Kibana ││
    │  │(Metrics) │(Dashboards)│(Traces)│(Logs) ││
    │  └──────────┴─────────┴─────────┴────────┘│
    └──────────────────────────────────────────────┘
```

## 💻 Local Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-prod.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run development server
python -m uvicorn src.api.rest_service:app --reload --port 8000
```

### Running Tests

```bash
# Unit tests
pytest tests/test_integration.py -v

# Integration tests
pytest tests/test_integration.py -v -m integration

# Load tests
pytest tests/test_integration.py -v -m slow

# Coverage report
pytest --cov=src --cov-report=html
```

## 🐳 Docker Deployment

### Build Images

```bash
# Production image
docker build -f deployment/Dockerfile -t rev-system:latest .

# Development image with debugging tools
docker build -f deployment/Dockerfile --target development -t rev-system:dev .
```

### Docker Compose Stack

```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f rev-api

# Scale API service
docker-compose -f deployment/docker-compose.yml up -d --scale rev-api=3

# Stop services
docker-compose -f deployment/docker-compose.yml down
```

### Service URLs

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **GraphQL**: http://localhost:8000/graphql
- **Grafana**: http://localhost:3000 (admin/admin)
- **Kibana**: http://localhost:5601
- **Jaeger**: http://localhost:16686
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9091

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Create namespace
kubectl create namespace rev-system
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n rev-system

# Get service endpoint
kubectl get svc rev-api-service -n rev-system

# View logs
kubectl logs -f deployment/rev-api -n rev-system

# Scale deployment
kubectl scale deployment rev-api --replicas=5 -n rev-system
```

### Helm Chart (Optional)

```bash
# Add Helm repository
helm repo add rev https://rev-system.github.io/charts

# Install REV
helm install rev rev/rev-system \
  --namespace rev-system \
  --set image.tag=latest \
  --set gpu.enabled=true \
  --set persistence.size=500Gi
```

## 📚 API Documentation

### REST Endpoints

#### Health Check
```bash
GET /health
```

#### Analyze Model
```bash
POST /analyze
Content-Type: application/json

{
  "model_path": "/path/to/model",
  "challenges": 50,
  "enable_principled_features": true,
  "enable_prompt_orchestration": true
}
```

#### Compare Fingerprints
```bash
POST /compare
Content-Type: application/json

{
  "fingerprint1": {...},
  "fingerprint2": {...},
  "method": "hamming"
}
```

### WebSocket API

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/analysis');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'start_analysis',
    model_path: '/path/to/model'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
};
```

### GraphQL API

```graphql
query GetAnalysis($requestId: String!) {
  analysisStatus(requestId: $requestId) {
    requestId
    modelInfo {
      name
      family
      architecture
    }
    confidence
    processingTime
  }
}

mutation StartAnalysis($modelPath: String!) {
  startAnalysis(modelPath: $modelPath, challenges: 50) {
    requestId
  }
}
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Key metrics exposed at `/metrics`:

- `rev_requests_total`: Total API requests
- `rev_request_duration_seconds`: Request latency
- `rev_model_inference_seconds`: Model inference time
- `rev_memory_usage_bytes`: Memory consumption
- `rev_errors_total`: Error counts by type
- `rev_circuit_breaker_state`: Circuit breaker status

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Copy dashboard files
cp deployment/monitoring/grafana/dashboards/*.json \
   /var/lib/grafana/dashboards/

# Or import via API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @deployment/monitoring/dashboards/rev-dashboard.json
```

### Distributed Tracing

View traces in Jaeger UI: http://localhost:16686

Enable tracing in code:
```python
from src.utils.logging_config import default_config

with default_config.tracing_manager.trace_operation(
    "model_analysis",
    model_name="llama-70b",
    challenges=50
):
    # Your code here
    pass
```

### Log Aggregation

Search logs in Kibana: http://localhost:5601

Query examples:
```
level:ERROR AND module:rev_pipeline
@timestamp:[now-1h TO now] AND message:*memory*
```

## 🛡️ Error Recovery

### Circuit Breaker Configuration

```python
# In src/utils/error_handling.py
breaker_config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,     # Try recovery after 60s
    success_threshold=2        # Close after 2 successes
)
```

### Graceful Degradation

Features automatically degrade on failure:

```python
# Register features for degradation
degradation_manager.register_feature("prompt_orchestration")
degradation_manager.register_feature("principled_features")
degradation_manager.register_feature("unified_fingerprints")
```

### Checkpoint Recovery

```bash
# Resume from checkpoint
python run_rev.py /path/to/model \
  --resume-from-checkpoint \
  --checkpoint-dir checkpoints/
```

### Memory Management

Automatic memory recovery triggers when:
- System memory > 90%
- GPU memory > 95%
- OOM errors detected

Actions taken:
1. Clear GPU cache
2. Reduce batch/segment size
3. Switch to CPU if needed
4. Enable memory-mapped loading

## 🔒 Security

### Authentication

Configure JWT authentication:

```bash
# Generate secret key
openssl rand -hex 32 > jwt_secret.key

# Set in environment
export JWT_SECRET=$(cat jwt_secret.key)
```

### Rate Limiting

Configure in `src/api/rest_service.py`:

```python
rate_limiter = RateLimiter(
    requests_per_minute=60,
    burst_size=10,
    per_user=True
)
```

### TLS/SSL

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes

# Configure Nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
}
```

## ⚡ Performance Tuning

### GPU Optimization

```bash
# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Enable TensorCore
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

### Memory Optimization

```python
# In run_rev.py
rev = REVUnified(
    memory_limit_gb=4.0,        # Limit per-segment memory
    enable_memory_mapping=True,  # Use mmap for large files
    cache_size_gb=10.0          # Disk cache size
)
```

### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET work_mem = '32MB';
```

### Redis Configuration

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Check memory usage
docker stats rev-api

# Increase memory limits
docker update --memory="8g" --memory-swap="16g" rev-api

# Or in docker-compose.yml:
services:
  rev-api:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi

# Enable GPU in compose
services:
  rev-api:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

#### API Timeout
```python
# Increase timeout in nginx.conf
proxy_read_timeout 300s;
proxy_connect_timeout 75s;

# In API code
@app.post("/analyze", timeout=300)
```

#### Database Connection Issues
```bash
# Check connectivity
docker exec -it rev-postgres psql -U rev -c "SELECT 1"

# Reset connections
docker restart rev-postgres rev-api
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with verbose output
python run_rev.py --debug --verbose

# API debug mode
uvicorn src.api.rest_service:app --reload --log-level debug
```

### Health Checks

```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:9091/metrics
curl http://localhost:9200/_cluster/health
```

## 📈 Scaling Guidelines

### Horizontal Scaling

```yaml
# docker-compose scale
docker-compose up -d --scale rev-api=5

# Kubernetes HPA
kubectl autoscale deployment rev-api \
  --min=2 --max=10 \
  --cpu-percent=70
```

### Vertical Scaling

```yaml
# Increase resources
services:
  rev-api:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
```

### Load Balancing

```nginx
upstream rev_backend {
    least_conn;
    server rev-api-1:8000 weight=3;
    server rev-api-2:8000 weight=2;
    server rev-api-3:8000 weight=1;
}
```

## 🔄 Backup & Recovery

### Database Backup

```bash
# Backup PostgreSQL
docker exec rev-postgres pg_dump -U rev rev_db > backup.sql

# Backup Redis
docker exec rev-redis redis-cli BGSAVE
docker cp rev-redis:/data/dump.rdb ./redis-backup.rdb

# Automated backup script
./scripts/backup.sh --all --compress
```

### Model Checkpoints

```bash
# Backup checkpoints
tar -czf checkpoints-$(date +%Y%m%d).tar.gz checkpoints/

# Restore from backup
tar -xzf checkpoints-20240101.tar.gz
```

## 🚨 Production Checklist

Before going to production:

- [ ] Configure environment variables
- [ ] Set up SSL/TLS certificates
- [ ] Configure authentication/authorization
- [ ] Set up monitoring dashboards
- [ ] Configure log aggregation
- [ ] Set up automated backups
- [ ] Configure rate limiting
- [ ] Set resource limits
- [ ] Enable health checks
- [ ] Configure auto-scaling
- [ ] Set up alerting rules
- [ ] Document runbooks
- [ ] Test disaster recovery
- [ ] Security audit
- [ ] Performance benchmarks

## 📝 Support

For issues or questions:
- GitHub Issues: https://github.com/rohanvinaik/REV/issues
- Documentation: https://rev-system.readthedocs.io
- Email: support@rev-system.ai

---
*REV Production Deployment Guide v3.0*