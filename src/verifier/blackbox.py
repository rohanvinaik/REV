"""
Black-Box Verification Interface for API-based model comparison.

This module implements verification of models through their API interfaces,
extracting logits, generating response hypervectors, and managing rate limits.
"""

import time
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import requests
from collections import deque, defaultdict
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Avoid circular import - use TYPE_CHECKING
if TYPE_CHECKING:
    from .contamination import UnifiedContaminationDetector, ContaminationResult


@dataclass
class APIConfig:
    """Configuration for API access."""
    
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    rate_limit_rpm: int = 60  # Requests per minute
    cache_ttl: int = 3600  # Cache time-to-live in seconds


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class CachedResponse:
    """Cached API response."""
    
    prompt: str
    response: Dict[str, Any]
    logits: Optional[np.ndarray]
    timestamp: datetime
    model_name: str
    
    def is_valid(self, ttl: int) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() - self.timestamp < timedelta(seconds=ttl)


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_rpm: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_rpm: Maximum requests per minute
        """
        self.max_rpm = max_rpm
        self.requests = deque()
        self.min_interval = 60.0 / max_rpm  # Minimum seconds between requests
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        # Check if we need to wait
        if len(self.requests) >= self.max_rpm:
            # Wait until the oldest request is older than 1 minute
            wait_time = 60 - (now - self.requests[0]) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Check minimum interval
        if self.requests:
            time_since_last = now - self.requests[-1]
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
        
        # Record this request
        self.requests.append(time.time())


class BlackBoxVerifier:
    """
    Black-box model verification through API interfaces.
    
    Supports logit extraction, response hypervector generation,
    and comparison of models accessible only through APIs.
    """
    
    def __init__(
        self,
        configs: Dict[str, APIConfig],
        cache_dir: str = "/tmp/rev_api_cache"
    ):
        """
        Initialize black-box verifier.
        
        Args:
            configs: Dictionary mapping model names to API configurations
            cache_dir: Directory for response caching
        """
        self.configs = configs
        self.cache_dir = cache_dir
        
        # Initialize rate limiters for each API
        self.rate_limiters = {
            name: RateLimiter(config.rate_limit_rpm)
            for name, config in configs.items()
        }
        
        # Response cache
        self.cache: Dict[str, CachedResponse] = {}
        
        # Provider-specific handlers
        self.providers = self._init_providers()
        
        # Executor for parallel API calls
        self.executor = ThreadPoolExecutor(max_workers=len(configs))
    
    def _init_providers(self) -> Dict[ModelProvider, callable]:
        """Initialize provider-specific API handlers."""
        return {
            ModelProvider.OPENAI: self._call_openai,
            ModelProvider.ANTHROPIC: self._call_anthropic,
            ModelProvider.HUGGINGFACE: self._call_huggingface,
            ModelProvider.COHERE: self._call_cohere,
            ModelProvider.LOCAL: self._call_local,
            ModelProvider.CUSTOM: self._call_custom
        }
    
    def _get_cache_key(self, model_name: str, prompt: str) -> str:
        """Generate cache key for a prompt."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{model_name}_{prompt_hash}"
    
    def _call_openai(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "logprobs": logprobs,
            "top_logprobs": 5 if logprobs else None
        }
        
        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def _call_anthropic(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        response = requests.post(
            f"{config.base_url}/messages",
            headers=headers,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def _call_huggingface(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call HuggingFace Inference API."""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"{config.base_url}/models/{config.model_name}",
            headers=headers,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def _call_cohere(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call Cohere API."""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config.model_name,
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "return_likelihoods": "ALL" if logprobs else "NONE"
        }
        
        response = requests.post(
            f"{config.base_url}/generate",
            headers=headers,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def _call_local(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call local model server."""
        # Assumes a local server with OpenAI-compatible API
        data = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "logprobs": 5 if logprobs else 0
        }
        
        response = requests.post(
            config.base_url,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def _call_custom(
        self,
        config: APIConfig,
        prompt: str,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """Call custom API endpoint."""
        # Generic implementation for custom endpoints
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        data = {
            "prompt": prompt,
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        response = requests.post(
            config.base_url,
            headers=headers,
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def extract_logits(
        self,
        response: Dict[str, Any],
        provider: ModelProvider
    ) -> Optional[np.ndarray]:
        """
        Extract logits from API response.
        
        Args:
            response: API response dictionary
            provider: Model provider type
            
        Returns:
            Logits as numpy array or None if not available
        """
        logits = None
        
        if provider == ModelProvider.OPENAI:
            # Extract from OpenAI response
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    # Extract top logprobs
                    content = choice["logprobs"]["content"]
                    if content:
                        token_logprobs = []
                        for token_data in content:
                            if "top_logprobs" in token_data:
                                # Get logprobs for top tokens
                                probs = [lp["logprob"] for lp in token_data["top_logprobs"]]
                                token_logprobs.append(probs)
                        if token_logprobs:
                            logits = np.array(token_logprobs)
        
        elif provider == ModelProvider.COHERE:
            # Extract from Cohere response
            if "generations" in response and response["generations"]:
                gen = response["generations"][0]
                if "token_likelihoods" in gen:
                    logits = np.array([
                        tl["likelihood"] for tl in gen["token_likelihoods"]
                    ])
        
        elif provider == ModelProvider.HUGGINGFACE:
            # HuggingFace may return scores
            if isinstance(response, list) and response:
                if "score" in response[0]:
                    logits = np.array([response[0]["score"]])
        
        # Convert to standard format if needed
        if logits is not None and len(logits.shape) == 1:
            logits = logits.reshape(-1, 1)
        
        return logits
    
    def generate_response_hypervector(
        self,
        response: Dict[str, Any],
        provider: ModelProvider,
        dimension: int = 10000
    ) -> np.ndarray:
        """
        Generate hypervector from API response.
        
        Args:
            response: API response dictionary
            provider: Model provider type
            dimension: Hypervector dimension
            
        Returns:
            Response hypervector
        """
        # Extract logits if available
        logits = self.extract_logits(response, provider)
        
        if logits is not None:
            # Use logits to generate hypervector
            flat_logits = logits.flatten()
            
            # Pad or truncate to dimension
            if len(flat_logits) < dimension:
                # Pad with zeros
                padded = np.zeros(dimension)
                padded[:len(flat_logits)] = flat_logits
                flat_logits = padded
            else:
                # Truncate
                flat_logits = flat_logits[:dimension]
            
            # Normalize
            norm = np.linalg.norm(flat_logits)
            if norm > 0:
                flat_logits = flat_logits / norm
            
            return flat_logits
        
        else:
            # Fallback: use response text embedding
            text = self._extract_text(response, provider)
            
            # Simple hash-based embedding
            text_hash = hashlib.sha256(text.encode()).digest()
            np.random.seed(int.from_bytes(text_hash[:4], 'big'))
            hypervector = np.random.randn(dimension)
            np.random.seed(None)
            
            # Normalize
            hypervector = hypervector / np.linalg.norm(hypervector)
            
            return hypervector
    
    def _extract_text(self, response: Dict[str, Any], provider: ModelProvider) -> str:
        """Extract text from API response."""
        text = ""
        
        if provider == ModelProvider.OPENAI:
            if "choices" in response and response["choices"]:
                text = response["choices"][0].get("message", {}).get("content", "")
        
        elif provider == ModelProvider.ANTHROPIC:
            if "content" in response and response["content"]:
                text = response["content"][0].get("text", "")
        
        elif provider == ModelProvider.COHERE:
            if "generations" in response and response["generations"]:
                text = response["generations"][0].get("text", "")
        
        elif provider == ModelProvider.HUGGINGFACE:
            if isinstance(response, list) and response:
                text = response[0].get("generated_text", "")
        
        else:
            # Generic extraction
            text = str(response.get("text", response.get("output", "")))
        
        return text
    
    def call_api_with_retry(
        self,
        model_name: str,
        prompt: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Call API with retry logic and caching.
        
        Args:
            model_name: Name of model to call
            prompt: Input prompt
            use_cache: Whether to use cached responses
            
        Returns:
            API response dictionary
        """
        # Check cache
        cache_key = self._get_cache_key(model_name, prompt)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.is_valid(self.configs[model_name].cache_ttl):
                return cached.response
        
        # Get configuration
        config = self.configs[model_name]
        
        # Determine provider
        provider = self._detect_provider(config)
        
        # Rate limiting
        self.rate_limiters[model_name].wait_if_needed()
        
        # Retry logic
        last_error = None
        for attempt in range(config.max_retries):
            try:
                # Call appropriate API
                handler = self.providers.get(provider, self._call_custom)
                response = handler(config, prompt)
                
                # Cache response
                self.cache[cache_key] = CachedResponse(
                    prompt=prompt,
                    response=response,
                    logits=self.extract_logits(response, provider),
                    timestamp=datetime.now(),
                    model_name=model_name
                )
                
                return response
                
            except Exception as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        raise RuntimeError(f"API call failed after {config.max_retries} attempts: {last_error}")
    
    def _detect_provider(self, config: APIConfig) -> ModelProvider:
        """Detect provider from configuration."""
        base_url = config.base_url.lower()
        
        if "openai.com" in base_url or "openai" in base_url:
            return ModelProvider.OPENAI
        elif "anthropic.com" in base_url or "anthropic" in base_url:
            return ModelProvider.ANTHROPIC
        elif "huggingface.co" in base_url or "huggingface" in base_url:
            return ModelProvider.HUGGINGFACE
        elif "cohere.ai" in base_url or "cohere" in base_url:
            return ModelProvider.COHERE
        elif "localhost" in base_url or "127.0.0.1" in base_url:
            return ModelProvider.LOCAL
        else:
            return ModelProvider.CUSTOM
    
    def compare_models(
        self,
        prompt: str,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same prompt.
        
        Args:
            prompt: Input prompt
            model_names: List of model names to compare (or all if None)
            
        Returns:
            Comparison results including hypervectors and similarities
        """
        if model_names is None:
            model_names = list(self.configs.keys())
        
        # Parallel API calls
        futures = {}
        for model_name in model_names:
            future = self.executor.submit(self.call_api_with_retry, model_name, prompt)
            futures[future] = model_name
        
        # Collect results
        responses = {}
        hypervectors = {}
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                response = future.result()
                responses[model_name] = response
                
                # Generate hypervector
                provider = self._detect_provider(self.configs[model_name])
                hypervectors[model_name] = self.generate_response_hypervector(
                    response, provider
                )
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                responses[model_name] = None
                hypervectors[model_name] = np.zeros(10000)
        
        # Compute pairwise similarities
        similarities = {}
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                if hypervectors[model_a] is not None and hypervectors[model_b] is not None:
                    similarity = np.dot(hypervectors[model_a], hypervectors[model_b]) / (
                        np.linalg.norm(hypervectors[model_a]) * 
                        np.linalg.norm(hypervectors[model_b]) + 1e-8
                    )
                    similarities[f"{model_a}_vs_{model_b}"] = float(similarity)
        
        return {
            "prompt": prompt,
            "responses": responses,
            "hypervectors": {k: v.tolist() if v is not None else None 
                           for k, v in hypervectors.items()},
            "similarities": similarities
        }
    
    async def compare_models_async(
        self,
        prompt: str,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously compare multiple models.
        
        Args:
            prompt: Input prompt
            model_names: List of model names to compare
            
        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = list(self.configs.keys())
        
        async def call_model(session, model_name):
            config = self.configs[model_name]
            provider = self._detect_provider(config)
            
            # Rate limiting
            self.rate_limiters[model_name].wait_if_needed()
            
            # Prepare request based on provider
            headers = {"Authorization": f"Bearer {config.api_key}"}
            data = {
                "model": config.model_name,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature
            }
            
            async with session.post(
                config.base_url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                return await response.json()
        
        # Create session and make parallel calls
        async with aiohttp.ClientSession() as session:
            tasks = [call_model(session, name) for name in model_names]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for model_name, response in zip(model_names, responses):
            if isinstance(response, Exception):
                print(f"Error with {model_name}: {response}")
                results[model_name] = None
            else:
                results[model_name] = response
        
        return results
    
    def get_model_response(
        self,
        prompt: str,
        config: APIConfig
    ) -> Dict[str, Any]:
        """
        Get model response for contamination detection.
        
        Args:
            prompt: Input prompt
            config: API configuration for the model
            
        Returns:
            Response dictionary with text and optional logits
        """
        # Detect provider
        provider = self._detect_provider(config)
        
        # Call API
        handler = self.providers.get(provider, self._call_custom)
        response = handler(config, prompt)
        
        # Extract components
        text = self._extract_text(response, provider)
        logits = self.extract_logits(response, provider)
        
        return {
            'text': text,
            'logits': logits,
            'raw_response': response,
            'provider': provider.value
        }
    
    def detect_contamination_api(
        self,
        target_model_name: str,
        reference_model_names: List[str],
        challenges: Optional[List[str]] = None,
        contamination_detector: Optional["UnifiedContaminationDetector"] = None
    ) -> "ContaminationResult":
        """
        Detect contamination in API-accessible models.
        
        Args:
            target_model_name: Name of target model to check
            reference_model_names: Names of reference models
            challenges: Test challenges for analysis
            contamination_detector: Optional detector instance
            
        Returns:
            ContaminationResult with detection details
        """
        # Initialize contamination detector if not provided
        if contamination_detector is None:
            from .contamination import UnifiedContaminationDetector
            contamination_detector = UnifiedContaminationDetector(
                blackbox_verifier=self
            )
        
        # Prepare API configs
        api_configs = {
            target_model_name: self.configs[target_model_name]
        }
        for ref_name in reference_model_names:
            if ref_name in self.configs:
                api_configs[ref_name] = self.configs[ref_name]
        
        # Run contamination detection using API
        result = contamination_detector.detect_contamination(
            model=None,  # No direct model access
            reference_models=[None] * len(reference_model_names),
            model_id=target_model_name,
            reference_ids=reference_model_names,
            challenges=challenges,
            use_api=True,
            api_configs=api_configs
        )
        
        return result
    
    def batch_contamination_check(
        self,
        model_names: List[str],
        reference_model_names: Optional[List[str]] = None,
        challenges: Optional[List[str]] = None
    ) -> Dict[str, "ContaminationResult"]:
        """
        Check contamination for multiple models in batch.
        
        Args:
            model_names: List of models to check
            reference_model_names: Reference models (uses all others if None)
            challenges: Test challenges
            
        Returns:
            Dictionary mapping model names to contamination results
        """
        if reference_model_names is None:
            # Use all other models as references
            reference_model_names = [
                name for name in self.configs.keys() 
                if name not in model_names
            ]
        
        # Initialize detector once for efficiency
        from .contamination import UnifiedContaminationDetector
        contamination_detector = UnifiedContaminationDetector(
            blackbox_verifier=self,
            cache_signatures=True  # Cache for efficiency
        )
        
        results = {}
        
        # Check each model
        for model_name in model_names:
            print(f"Checking contamination for {model_name}...")
            
            try:
                result = self.detect_contamination_api(
                    target_model_name=model_name,
                    reference_model_names=reference_model_names,
                    challenges=challenges,
                    contamination_detector=contamination_detector
                )
                results[model_name] = result
                
                # Print summary
                print(f"  - Type: {result.contamination_type.value}")
                print(f"  - Score: {result.contamination_score:.3f}")
                print(f"  - Confidence: {result.confidence:.3f}")
                
            except Exception as e:
                print(f"  - Error: {e}")
                results[model_name] = None
        
        return results
    
    def generate_contamination_report(
        self,
        results: Dict[str, "ContaminationResult"]
    ) -> str:
        """
        Generate comprehensive contamination report for multiple models.
        
        Args:
            results: Dictionary of contamination results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("API MODEL CONTAMINATION ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"\nModels Analyzed: {len(results)}")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        
        # Summary statistics
        contaminated = sum(
            1 for r in results.values() 
            if r and r.is_contaminated()
        )
        report.append(f"Contaminated Models: {contaminated}/{len(results)}")
        
        report.append("\n" + "-"*70)
        report.append("INDIVIDUAL MODEL RESULTS:")
        report.append("-"*70)
        
        # Sort by contamination score
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].contamination_score if x[1] else 0,
            reverse=True
        )
        
        for model_name, result in sorted_results:
            report.append(f"\n{model_name}:")
            
            if result is None:
                report.append("  Status: ERROR")
                continue
            
            report.append(f"  Contamination Type: {result.contamination_type.value}")
            report.append(f"  Score: {result.contamination_score:.3f}")
            report.append(f"  Confidence: {result.confidence:.3f}")
            report.append(f"  Contaminated: {'YES' if result.is_contaminated() else 'NO'}")
            
            if result.suspicious_models:
                report.append(f"  Related Models: {', '.join(result.suspicious_models[:3])}")
            
            if result.artifact_patterns:
                report.append(f"  Artifacts Detected: {len(result.artifact_patterns)}")
        
        # Relationship analysis
        report.append("\n" + "-"*70)
        report.append("CONTAMINATION RELATIONSHIPS:")
        report.append("-"*70)
        
        # Find clusters of related models
        relationships = defaultdict(list)
        for model_name, result in results.items():
            if result and result.suspicious_models:
                for suspicious in result.suspicious_models:
                    relationships[suspicious].append(model_name)
        
        for related_model, affected in relationships.items():
            if len(affected) > 1:
                report.append(f"\n{related_model} potentially contaminated:")
                for model in affected:
                    report.append(f"  - {model}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self.cache.clear()