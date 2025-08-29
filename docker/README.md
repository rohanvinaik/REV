# REV/HBT Unified Verification System - Docker Deployment

This directory contains Docker configuration for deploying the REV/HBT unified verification system with multiple services, load balancing, and monitoring.

## Architecture

The system consists of the following services:

### Core Services

1. **Unified Coordinator** (2 replicas, 2GB RAM each)
   - Main API endpoint for verification requests
   - Handles mode selection (fast/robust/hybrid/auto)
   - Manages result caching and request routing

2. **REV Verifier** (3 replicas, 4GB RAM each)
   - Fast sequential testing using REV algorithm
   - Memory-bounded streaming verification
   - Optimized for low latency

3. **HBT Consensus** (5 replicas, 8GB RAM each)
   - Byzantine fault-tolerant consensus (3f+1, f=1)
   - Robust verification with high accuracy
   - Handles up to 1 Byzantine failure

### Supporting Services

4. **Redis** - Caching and message passing
5. **Nginx** - Load balancing and SSL termination
6. **Prometheus** - Metrics collection
7. **Grafana** - Visualization and dashboards

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional, for using Makefile commands)
- 32GB RAM recommended for full deployment

### Basic Commands

```bash
# Build all images
make build

# Start all services
make up

# Check health status
make health-check

# View logs
make logs

# Stop all services
make down
```

### Manual Docker Compose Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Scale services
docker-compose up -d --scale rev-verifier=5 --scale hbt-consensus=7

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Usage

### Verification Endpoint

```bash
# Automatic mode selection
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "model_a": "gpt-3.5-turbo",
    "model_b": "gpt-4",
    "challenges": ["What is 2+2?", "Explain gravity"],
    "mode": "auto",
    "max_latency_ms": 5000,
    "min_accuracy": 0.9,
    "priority": "balanced"
  }'

# Fast mode (REV)
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "model_a": "model-a",
    "model_b": "model-b",
    "challenges": ["test prompt"],
    "mode": "fast"
  }'

# Robust mode (HBT)
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "model_a": "model-a",
    "model_b": "model-b",
    "challenges": ["test prompt"],
    "mode": "robust",
    "consensus_threshold": 0.67
  }'

# Hybrid mode
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "model_a": "model-a",
    "model_b": "model-b",
    "challenges": ["test prompt"],
    "mode": "hybrid"
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Statistics

```bash
curl http://localhost:8000/stats
```

## Configuration

### Environment Variables

Each service can be configured via environment variables:

#### Unified Coordinator
- `MAX_MEMORY_MB`: Maximum memory usage (default: 2048)
- `CACHE_ENABLED`: Enable result caching (default: true)
- `MODE_SELECTION`: Default mode selection (default: auto)

#### REV Verifier
- `MAX_MEMORY_MB`: Maximum memory usage (default: 4096)
- `SEGMENT_SIZE`: Segment size for processing (default: 512)
- `BUFFER_SIZE`: Buffer size for streaming (default: 4)

#### HBT Consensus
- `MAX_MEMORY_MB`: Maximum memory usage (default: 8192)
- `NUM_VALIDATORS`: Number of validators (default: 5)
- `FAULT_TOLERANCE`: Byzantine fault tolerance (default: 1)
- `CONSENSUS_THRESHOLD`: Consensus threshold (default: 0.67)

### Scaling

Services can be scaled independently:

```bash
# Scale REV verifier for higher throughput
make scale-rev REPLICAS=5

# Scale HBT consensus (must be 3f+1)
make scale-hbt REPLICAS=7  # For f=2

# Or using docker-compose directly
docker-compose up -d --scale rev-verifier=5
```

### Resource Limits

Default resource limits per service:

| Service | Replicas | RAM/Instance | CPU/Instance | Total RAM | Total CPU |
|---------|----------|--------------|--------------|-----------|-----------|
| Coordinator | 2 | 2GB | 1.0 | 4GB | 2.0 |
| REV Verifier | 3 | 4GB | 2.0 | 12GB | 6.0 |
| HBT Consensus | 5 | 8GB | 4.0 | 40GB | 20.0 |
| Redis | 1 | 1GB | 0.5 | 1GB | 0.5 |
| Nginx | 1 | 256MB | 0.25 | 256MB | 0.25 |
| Prometheus | 1 | 1GB | 0.5 | 1GB | 0.5 |
| Grafana | 1 | 512MB | 0.5 | 512MB | 0.5 |

**Total**: ~60GB RAM, ~30 CPU cores (with default configuration)

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/admin)

Pre-configured dashboards:
- Service Health Overview
- Request Latency & Throughput
- Resource Usage (CPU, Memory, Network)
- Consensus Metrics
- Cache Hit Rates

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Key metrics:
- `verification_requests_total`: Total verification requests
- `verification_latency_seconds`: Request latency histogram
- `consensus_rounds_total`: HBT consensus rounds
- `cache_hit_ratio`: Cache hit rate
- `memory_usage_bytes`: Memory usage per service

### Logging

View logs for all services:
```bash
make logs
```

View logs for specific service:
```bash
make logs-service SERVICE=rev-verifier
```

## SSL/TLS Configuration

For production, update SSL certificates:

1. Place certificates in `docker/ssl/`:
   - `cert.pem`: SSL certificate
   - `key.pem`: Private key

2. Update `nginx.conf` with your domain

3. Restart nginx:
```bash
docker-compose restart nginx
```

## Backup & Restore

### Backup volumes
```bash
make backup
```

### Restore from backup
```bash
make restore BACKUP_DATE=20240101-120000
```

## Troubleshooting

### Service won't start
```bash
# Check logs
make logs-service SERVICE=<service-name>

# Check resource usage
make stats

# Restart service
make restart-service SERVICE=<service-name>
```

### High latency
- Check resource limits: `docker stats`
- Scale REV verifier: `make scale-rev REPLICAS=5`
- Check network: `docker network inspect rev-hbt-verification_rev-network`

### Consensus failures
- Ensure HBT replicas = 3f+1
- Check Byzantine node detection in logs
- Verify network connectivity between consensus nodes

### Memory issues
- Reduce segment size
- Decrease buffer size
- Scale horizontally instead of vertically

## Development

### Building individual images
```bash
make dev-build-rev        # Build REV verifier
make dev-build-hbt        # Build HBT consensus
make dev-build-coordinator # Build coordinator
```

### Accessing containers
```bash
make exec-rev        # Shell into REV verifier
make exec-hbt        # Shell into HBT consensus
make exec-coordinator # Shell into coordinator
```

### Running tests
```bash
make test      # Run integration tests
make benchmark # Run performance benchmark
```

## Performance Tuning

### REV Verifier (Fast Mode)
- Optimize for latency: Reduce segment size
- Increase parallelism: Scale replicas
- Enable caching: Use Redis effectively

### HBT Consensus (Robust Mode)
- Balance validators: Use 3f+1 formula
- Tune consensus threshold: 0.67-0.75 typical
- Monitor Byzantine detection

### Hybrid Mode
- Adjust weights: Balance speed vs accuracy
- Parallel execution: Leverage both modes
- Smart caching: Cache intermediate results

## Security Considerations

1. **API Keys**: Store securely, use environment variables
2. **Network**: Use internal Docker network for service communication
3. **SSL/TLS**: Always use HTTPS in production
4. **Rate Limiting**: Configure in nginx.conf
5. **Authentication**: Implement API key or OAuth2
6. **Monitoring**: Set up alerts for anomalies

## Support

For issues or questions:
- Check logs: `make logs`
- View metrics: `make monitoring`
- Health status: `make health-check`
- GitHub Issues: https://github.com/rohanvinaik/REV

## License

See LICENSE file in the project root.