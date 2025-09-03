"""
Advanced rate limiting for REV API endpoints.
Implements token bucket algorithm with Redis backend support.
"""

import time
import threading
import logging
import json
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size: int = 60  # seconds
    max_backoff: float = 60.0  # seconds
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    
    # Per-model limits
    model_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Per-user limits
    user_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    
    capacity: int
    refill_rate: float
    tokens: float
    last_refill: float
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    tokens_remaining: float
    retry_after: Optional[float] = None
    quota_reset: Optional[float] = None
    headers: Dict[str, str] = field(default_factory=dict)


class RateLimiterBackend(ABC):
    """Abstract backend for rate limiting."""
    
    @abstractmethod
    def check_rate_limit(
        self,
        key: str,
        cost: float = 1.0
    ) -> RateLimitResult:
        """Check if request is allowed."""
        pass
    
    @abstractmethod
    def reset_quota(self, key: str):
        """Reset quota for a key."""
        pass


class LocalRateLimiter(RateLimiterBackend):
    """Local in-memory rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize local rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.buckets: Dict[str, TokenBucket] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.backoff_state: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def check_rate_limit(
        self,
        key: str,
        cost: float = 1.0
    ) -> RateLimitResult:
        """
        Check if request is allowed using token bucket algorithm.
        
        Args:
            key: Unique identifier (user, model, etc.)
            cost: Cost of the request in tokens
            
        Returns:
            RateLimitResult with decision
        """
        with self.lock:
            bucket = self._get_or_create_bucket(key)
            
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - bucket.last_refill
            tokens_to_add = elapsed * bucket.refill_rate
            
            with bucket.lock:
                bucket.tokens = min(
                    bucket.capacity,
                    bucket.tokens + tokens_to_add
                )
                bucket.last_refill = now
                
                # Check if enough tokens available
                if bucket.tokens >= cost:
                    bucket.tokens -= cost
                    
                    # Record request
                    self.request_history[key].append(now)
                    
                    # Clear backoff if successful
                    if key in self.backoff_state:
                        del self.backoff_state[key]
                    
                    return RateLimitResult(
                        allowed=True,
                        tokens_remaining=bucket.tokens,
                        headers=self._generate_headers(bucket, now)
                    )
                else:
                    # Calculate retry time
                    tokens_needed = cost - bucket.tokens
                    retry_after = tokens_needed / bucket.refill_rate
                    
                    # Apply exponential backoff if needed
                    retry_after = self._apply_backoff(key, retry_after)
                    
                    return RateLimitResult(
                        allowed=False,
                        tokens_remaining=bucket.tokens,
                        retry_after=retry_after,
                        quota_reset=now + retry_after,
                        headers=self._generate_headers(bucket, now, retry_after)
                    )
    
    def reset_quota(self, key: str):
        """Reset quota for a key."""
        with self.lock:
            if key in self.buckets:
                bucket = self.buckets[key]
                with bucket.lock:
                    bucket.tokens = bucket.capacity
                    bucket.last_refill = time.time()
            
            if key in self.request_history:
                self.request_history[key].clear()
            
            if key in self.backoff_state:
                del self.backoff_state[key]
    
    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for key."""
        if key not in self.buckets:
            # Check for specific limits
            limits = self._get_limits_for_key(key)
            
            self.buckets[key] = TokenBucket(
                capacity=limits.get("burst_size", self.config.burst_size),
                refill_rate=limits.get("requests_per_second", self.config.requests_per_second),
                tokens=limits.get("burst_size", self.config.burst_size),
                last_refill=time.time()
            )
        
        return self.buckets[key]
    
    def _get_limits_for_key(self, key: str) -> Dict[str, Any]:
        """Get specific limits for a key."""
        # Check model-specific limits
        if key.startswith("model:"):
            model_name = key.split(":", 1)[1]
            if model_name in self.config.model_limits:
                return self.config.model_limits[model_name]
        
        # Check user-specific limits
        if key.startswith("user:"):
            user_id = key.split(":", 1)[1]
            if user_id in self.config.user_limits:
                return self.config.user_limits[user_id]
        
        # Return default limits
        return {
            "requests_per_second": self.config.requests_per_second,
            "burst_size": self.config.burst_size
        }
    
    def _apply_backoff(self, key: str, base_retry: float) -> float:
        """Apply exponential backoff for repeated failures."""
        if key not in self.backoff_state:
            self.backoff_state[key] = base_retry
        else:
            # Exponential backoff with jitter
            self.backoff_state[key] = min(
                self.backoff_state[key] * 2,
                self.config.max_backoff
            )
        
        # Add jitter (±10%)
        import random
        jitter = random.uniform(0.9, 1.1)
        
        return self.backoff_state[key] * jitter
    
    def _generate_headers(
        self,
        bucket: TokenBucket,
        now: float,
        retry_after: Optional[float] = None
    ) -> Dict[str, str]:
        """Generate rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(int(bucket.capacity)),
            "X-RateLimit-Remaining": str(int(bucket.tokens)),
            "X-RateLimit-Reset": str(int(now + bucket.capacity / bucket.refill_rate))
        }
        
        if retry_after is not None:
            headers["Retry-After"] = str(int(retry_after))
        
        return headers


class RedisRateLimiter(RateLimiterBackend):
    """Redis-backed distributed rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize Redis rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.redis_client = None
        
        # Try to import and connect to Redis
        try:
            import redis
            if config.redis_url:
                self.redis_client = redis.from_url(config.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for distributed rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to local rate limiting.")
            self.fallback = LocalRateLimiter(config)
    
    def check_rate_limit(
        self,
        key: str,
        cost: float = 1.0
    ) -> RateLimitResult:
        """
        Check rate limit using Redis.
        
        Args:
            key: Unique identifier
            cost: Cost of request
            
        Returns:
            RateLimitResult
        """
        if not self.redis_client:
            return self.fallback.check_rate_limit(key, cost)
        
        try:
            # Use Redis Lua script for atomic token bucket
            lua_script = """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local cost = tonumber(ARGV[3])
                local now = tonumber(ARGV[4])
                
                local bucket = redis.call('HGETALL', key)
                local tokens = capacity
                local last_refill = now
                
                if #bucket > 0 then
                    tokens = tonumber(bucket[2])
                    last_refill = tonumber(bucket[4])
                    
                    -- Refill tokens
                    local elapsed = now - last_refill
                    local tokens_to_add = elapsed * refill_rate
                    tokens = math.min(capacity, tokens + tokens_to_add)
                end
                
                if tokens >= cost then
                    tokens = tokens - cost
                    redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
                    redis.call('EXPIRE', key, 3600)
                    return {1, tokens}
                else
                    return {0, tokens}
                end
            """
            
            limits = self._get_limits_for_key(key)
            result = self.redis_client.eval(
                lua_script,
                1,
                key,
                limits["burst_size"],
                limits["requests_per_second"],
                cost,
                time.time()
            )
            
            allowed = result[0] == 1
            tokens_remaining = result[1]
            
            if allowed:
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=tokens_remaining
                )
            else:
                retry_after = (cost - tokens_remaining) / limits["requests_per_second"]
                return RateLimitResult(
                    allowed=False,
                    tokens_remaining=tokens_remaining,
                    retry_after=retry_after
                )
                
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return self.fallback.check_rate_limit(key, cost)
    
    def reset_quota(self, key: str):
        """Reset quota in Redis."""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Failed to reset quota in Redis: {e}")
        
        if hasattr(self, 'fallback'):
            self.fallback.reset_quota(key)
    
    def _get_limits_for_key(self, key: str) -> Dict[str, Any]:
        """Get limits for key (same as LocalRateLimiter)."""
        if key.startswith("model:"):
            model_name = key.split(":", 1)[1]
            if model_name in self.config.model_limits:
                return self.config.model_limits[model_name]
        
        if key.startswith("user:"):
            user_id = key.split(":", 1)[1]
            if user_id in self.config.user_limits:
                return self.config.user_limits[user_id]
        
        return {
            "requests_per_second": self.config.requests_per_second,
            "burst_size": self.config.burst_size
        }


class HierarchicalRateLimiter:
    """
    Hierarchical rate limiter with multiple levels of quotas.
    Supports user -> model -> global hierarchy.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize hierarchical rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        
        # Create backend based on configuration
        if config.enable_distributed and config.redis_url:
            self.backend = RedisRateLimiter(config)
        else:
            self.backend = LocalRateLimiter(config)
        
        # Track quota usage statistics
        self.stats = defaultdict(lambda: {
            "requests": 0,
            "accepted": 0,
            "rejected": 0,
            "tokens_consumed": 0
        })
    
    def check_rate_limit(
        self,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        operation: str = "inference",
        cost: float = 1.0
    ) -> RateLimitResult:
        """
        Check hierarchical rate limits.
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            operation: Type of operation
            cost: Cost in tokens
            
        Returns:
            RateLimitResult
        """
        checks = []
        
        # Check global limit
        global_key = f"global:{operation}"
        checks.append((global_key, self.backend.check_rate_limit(global_key, cost)))
        
        # Check model-specific limit
        if model_id:
            model_key = f"model:{model_id}:{operation}"
            checks.append((model_key, self.backend.check_rate_limit(model_key, cost)))
        
        # Check user-specific limit
        if user_id:
            user_key = f"user:{user_id}:{operation}"
            checks.append((user_key, self.backend.check_rate_limit(user_key, cost)))
        
        # Check combined user-model limit
        if user_id and model_id:
            combined_key = f"user:{user_id}:model:{model_id}:{operation}"
            checks.append((combined_key, self.backend.check_rate_limit(combined_key, cost)))
        
        # All checks must pass
        for key, result in checks:
            if not result.allowed:
                # Update statistics
                self.stats[key]["requests"] += 1
                self.stats[key]["rejected"] += 1
                
                return result
        
        # All checks passed - update statistics
        for key, _ in checks:
            self.stats[key]["requests"] += 1
            self.stats[key]["accepted"] += 1
            self.stats[key]["tokens_consumed"] += cost
        
        # Return the most restrictive result
        min_tokens = min(r.tokens_remaining for _, r in checks)
        return RateLimitResult(
            allowed=True,
            tokens_remaining=min_tokens
        )
    
    def get_quota_status(
        self,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current quota status.
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            Quota status dictionary
        """
        status = {
            "quotas": {},
            "statistics": {}
        }
        
        # Check various quota levels
        keys = ["global:inference"]
        
        if model_id:
            keys.append(f"model:{model_id}:inference")
        
        if user_id:
            keys.append(f"user:{user_id}:inference")
        
        if user_id and model_id:
            keys.append(f"user:{user_id}:model:{model_id}:inference")
        
        for key in keys:
            result = self.backend.check_rate_limit(key, 0)  # Check without consuming
            status["quotas"][key] = {
                "tokens_remaining": result.tokens_remaining,
                "allowed": result.allowed
            }
            
            if key in self.stats:
                status["statistics"][key] = dict(self.stats[key])
        
        return status
    
    def reset_user_quota(self, user_id: str):
        """Reset all quotas for a user."""
        keys = [
            f"user:{user_id}:inference",
            f"user:{user_id}:model:*:inference"  # Pattern for all models
        ]
        
        for key in keys:
            self.backend.reset_quota(key)
    
    def set_model_limits(
        self,
        model_id: str,
        requests_per_second: float,
        burst_size: int
    ):
        """
        Set specific limits for a model.
        
        Args:
            model_id: Model identifier
            requests_per_second: Rate limit
            burst_size: Burst capacity
        """
        self.config.model_limits[model_id] = {
            "requests_per_second": requests_per_second,
            "burst_size": burst_size
        }
    
    def set_user_limits(
        self,
        user_id: str,
        requests_per_second: float,
        burst_size: int
    ):
        """
        Set specific limits for a user.
        
        Args:
            user_id: User identifier
            requests_per_second: Rate limit
            burst_size: Burst capacity
        """
        self.config.user_limits[user_id] = {
            "requests_per_second": requests_per_second,
            "burst_size": burst_size
        }


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load.
    """
    
    def __init__(
        self,
        base_config: RateLimitConfig,
        load_threshold: float = 0.8,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            base_config: Base configuration
            load_threshold: System load threshold for adaptation
            adaptation_rate: Rate of adaptation
        """
        self.base_config = base_config
        self.load_threshold = load_threshold
        self.adaptation_rate = adaptation_rate
        
        self.current_config = RateLimitConfig(**base_config.__dict__)
        self.limiter = HierarchicalRateLimiter(self.current_config)
        
        # Track system metrics
        self.metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "request_latency": deque(maxlen=1000),
            "rejection_rate": deque(maxlen=100)
        }
        
        # Start adaptation thread
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
    
    def check_rate_limit(self, **kwargs) -> RateLimitResult:
        """Check rate limit with adaptation."""
        start = time.time()
        result = self.limiter.check_rate_limit(**kwargs)
        
        # Track latency
        latency = time.time() - start
        self.metrics["request_latency"].append(latency)
        
        return result
    
    def _adaptation_loop(self):
        """Background loop for adapting rate limits."""
        while True:
            time.sleep(10)  # Adapt every 10 seconds
            
            try:
                self._adapt_limits()
            except Exception as e:
                logger.error(f"Adaptation failed: {e}")
    
    def _adapt_limits(self):
        """Adapt rate limits based on system metrics."""
        # Get current system load
        import psutil
        
        cpu_usage = psutil.cpu_percent() / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        self.metrics["cpu_usage"].append(cpu_usage)
        self.metrics["memory_usage"].append(memory_usage)
        
        # Calculate average metrics
        avg_cpu = sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0
        avg_memory = sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
        avg_latency = sum(self.metrics["request_latency"]) / len(self.metrics["request_latency"]) if self.metrics["request_latency"] else 0
        
        # Determine if we should adapt
        system_load = max(avg_cpu, avg_memory)
        
        if system_load > self.load_threshold:
            # Reduce rate limits
            scale_factor = 1.0 - self.adaptation_rate
            self._scale_limits(scale_factor)
            logger.info(f"Reduced rate limits due to high load ({system_load:.1%})")
            
        elif system_load < self.load_threshold * 0.5 and avg_latency < 0.1:
            # Increase rate limits
            scale_factor = 1.0 + self.adaptation_rate
            self._scale_limits(scale_factor)
            logger.info(f"Increased rate limits due to low load ({system_load:.1%})")
    
    def _scale_limits(self, factor: float):
        """Scale all rate limits by factor."""
        # Scale global limits
        self.current_config.requests_per_second = max(
            1.0,
            min(1000.0, self.current_config.requests_per_second * factor)
        )
        
        self.current_config.burst_size = max(
            1,
            min(1000, int(self.current_config.burst_size * factor))
        )
        
        # Scale model and user limits
        for limits_dict in [self.current_config.model_limits, self.current_config.user_limits]:
            for key, limits in limits_dict.items():
                if "requests_per_second" in limits:
                    limits["requests_per_second"] *= factor
                if "burst_size" in limits:
                    limits["burst_size"] = int(limits["burst_size"] * factor)
        
        # Recreate limiter with new config
        self.limiter = HierarchicalRateLimiter(self.current_config)