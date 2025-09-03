# Multi-stage build for production REV deployment
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY fingerprint_library/ ./fingerprint_library/
COPY run_rev.py ./
COPY CLAUDE.md ./

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs /app/logs /app/experiments

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=42
ENV CUDA_LAUNCH_BLOCKING=1

# Create non-root user
RUN useradd -m -s /bin/bash revuser && \
    chown -R revuser:revuser /app

USER revuser

# Expose ports
EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.rest_service:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development tools
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy all code
COPY . .

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=true

# Run in development mode
CMD ["python", "-m", "uvicorn", "src.api.rest_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]