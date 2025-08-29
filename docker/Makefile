# Docker Makefile for REV/HBT Unified Verification System

.PHONY: help build up down restart logs clean test monitoring

# Variables
COMPOSE_FILE = docker-compose.yml
PROJECT_NAME = rev-hbt-verification
DOCKER_REGISTRY = localhost:5000

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)REV/HBT Docker Management$(NC)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build --parallel
	@echo "$(GREEN)Build complete!$(NC)"

build-no-cache: ## Build all Docker images without cache
	@echo "$(GREEN)Building Docker images (no cache)...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build --no-cache --parallel
	@echo "$(GREEN)Build complete!$(NC)"

up: ## Start all services
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "$(YELLOW)Waiting for services to be healthy...$(NC)"
	@sleep 10
	@make health-check

up-dev: ## Start services in development mode (with logs)
	@echo "$(GREEN)Starting services in development mode...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up

down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down
	@echo "$(GREEN)Services stopped!$(NC)"

restart: ## Restart all services
	@make down
	@make up

restart-service: ## Restart a specific service (use SERVICE=<name>)
	@echo "$(YELLOW)Restarting $(SERVICE)...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) restart $(SERVICE)
	@echo "$(GREEN)$(SERVICE) restarted!$(NC)"

logs: ## Show logs for all services
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f --tail=100

logs-service: ## Show logs for specific service (use SERVICE=<name>)
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f --tail=100 $(SERVICE)

ps: ## Show running containers
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps

health-check: ## Check health of all services
	@echo "$(GREEN)Checking service health...$(NC)"
	@docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps | grep -E "(health|Up)" || true
	@echo ""
	@echo "$(GREEN)Testing endpoints...$(NC)"
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && echo "✓ Unified Coordinator: $(GREEN)Healthy$(NC)" || echo "✗ Unified Coordinator: $(RED)Unhealthy$(NC)"
	@curl -f http://localhost:8001/health > /dev/null 2>&1 && echo "✓ REV Verifier: $(GREEN)Healthy$(NC)" || echo "✗ REV Verifier: $(RED)Unhealthy$(NC)"
	@curl -f http://localhost:8004/health > /dev/null 2>&1 && echo "✓ HBT Consensus: $(GREEN)Healthy$(NC)" || echo "✗ HBT Consensus: $(RED)Unhealthy$(NC)"
	@curl -f http://localhost:9090/-/healthy > /dev/null 2>&1 && echo "✓ Prometheus: $(GREEN)Healthy$(NC)" || echo "✗ Prometheus: $(RED)Unhealthy$(NC)"
	@curl -f http://localhost:3000/api/health > /dev/null 2>&1 && echo "✓ Grafana: $(GREEN)Healthy$(NC)" || echo "✗ Grafana: $(RED)Unhealthy$(NC)"

scale-rev: ## Scale REV verifier (use REPLICAS=<n>)
	@echo "$(YELLOW)Scaling REV verifier to $(REPLICAS) replicas...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d --scale rev-verifier=$(REPLICAS)
	@echo "$(GREEN)Scaled!$(NC)"

scale-hbt: ## Scale HBT consensus (use REPLICAS=<n>, must be 3f+1)
	@echo "$(YELLOW)Scaling HBT consensus to $(REPLICAS) replicas...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d --scale hbt-consensus=$(REPLICAS)
	@echo "$(GREEN)Scaled!$(NC)"

test: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	@python3 ../tests/test_docker_integration.py
	@echo "$(GREEN)Tests complete!$(NC)"

benchmark: ## Run performance benchmark
	@echo "$(GREEN)Running performance benchmark...$(NC)"
	@curl -X POST http://localhost:8000/verify \
		-H "Content-Type: application/json" \
		-d '{"model_a": "gpt-3.5-turbo", "model_b": "gpt-4", "challenges": ["What is 2+2?"], "mode": "auto"}' \
		| python3 -m json.tool
	@echo "$(GREEN)Benchmark complete!$(NC)"

monitoring: ## Open monitoring dashboards
	@echo "$(GREEN)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

clean: ## Clean up containers, volumes, and images
	@echo "$(RED)WARNING: This will remove all containers, volumes, and images!$(NC)"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down -v --rmi all
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-volumes: ## Clean up only volumes
	@echo "$(YELLOW)Removing volumes...$(NC)"
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down -v
	@echo "$(GREEN)Volumes removed!$(NC)"

push: ## Push images to registry
	@echo "$(GREEN)Pushing images to registry...$(NC)"
	docker tag rev/verifier:latest $(DOCKER_REGISTRY)/rev/verifier:latest
	docker tag rev/hbt-consensus:latest $(DOCKER_REGISTRY)/rev/hbt-consensus:latest
	docker tag rev/unified-coordinator:latest $(DOCKER_REGISTRY)/rev/unified-coordinator:latest
	docker push $(DOCKER_REGISTRY)/rev/verifier:latest
	docker push $(DOCKER_REGISTRY)/rev/hbt-consensus:latest
	docker push $(DOCKER_REGISTRY)/rev/unified-coordinator:latest
	@echo "$(GREEN)Push complete!$(NC)"

pull: ## Pull images from registry
	@echo "$(GREEN)Pulling images from registry...$(NC)"
	docker pull $(DOCKER_REGISTRY)/rev/verifier:latest
	docker pull $(DOCKER_REGISTRY)/rev/hbt-consensus:latest
	docker pull $(DOCKER_REGISTRY)/rev/unified-coordinator:latest
	docker tag $(DOCKER_REGISTRY)/rev/verifier:latest rev/verifier:latest
	docker tag $(DOCKER_REGISTRY)/rev/hbt-consensus:latest rev/hbt-consensus:latest
	docker tag $(DOCKER_REGISTRY)/rev/unified-coordinator:latest rev/unified-coordinator:latest
	@echo "$(GREEN)Pull complete!$(NC)"

exec-rev: ## Execute shell in REV verifier container
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec rev-verifier /bin/bash

exec-hbt: ## Execute shell in HBT consensus container
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec hbt-consensus /bin/bash

exec-coordinator: ## Execute shell in unified coordinator container
	docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) exec unified-coordinator /bin/bash

stats: ## Show resource usage statistics
	@echo "$(GREEN)Resource Usage Statistics:$(NC)"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
		$(shell docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) ps -q)

backup: ## Backup volumes
	@echo "$(GREEN)Backing up volumes...$(NC)"
	@mkdir -p backups
	docker run --rm -v $(PROJECT_NAME)_redis-data:/data -v $(PWD)/backups:/backup \
		alpine tar czf /backup/redis-data-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	docker run --rm -v $(PROJECT_NAME)_prometheus-data:/data -v $(PWD)/backups:/backup \
		alpine tar czf /backup/prometheus-data-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	docker run --rm -v $(PROJECT_NAME)_grafana-data:/data -v $(PWD)/backups:/backup \
		alpine tar czf /backup/grafana-data-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "$(GREEN)Backup complete!$(NC)"

restore: ## Restore volumes from backup (use BACKUP_DATE=<date>)
	@echo "$(YELLOW)Restoring volumes from backup...$(NC)"
	docker run --rm -v $(PROJECT_NAME)_redis-data:/data -v $(PWD)/backups:/backup \
		alpine tar xzf /backup/redis-data-$(BACKUP_DATE).tar.gz -C /data
	docker run --rm -v $(PROJECT_NAME)_prometheus-data:/data -v $(PWD)/backups:/backup \
		alpine tar xzf /backup/prometheus-data-$(BACKUP_DATE).tar.gz -C /data
	docker run --rm -v $(PROJECT_NAME)_grafana-data:/data -v $(PWD)/backups:/backup \
		alpine tar xzf /backup/grafana-data-$(BACKUP_DATE).tar.gz -C /data
	@echo "$(GREEN)Restore complete!$(NC)"

# Development helpers
dev-build-rev: ## Build only REV verifier image
	docker build -f Dockerfile.rev-verifier -t rev/verifier:latest ..

dev-build-hbt: ## Build only HBT consensus image
	docker build -f Dockerfile.hbt-consensus -t rev/hbt-consensus:latest ..

dev-build-coordinator: ## Build only unified coordinator image
	docker build -f Dockerfile.unified-coordinator -t rev/unified-coordinator:latest ..

# Default target
.DEFAULT_GOAL := help