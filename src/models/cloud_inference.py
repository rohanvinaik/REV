"""
Cloud-based inference for large models via API endpoints.
Supports OpenAI, Anthropic, HuggingFace Inference API, and custom endpoints.
"""

import os
import time
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud inference providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    REPLICATE = "replicate"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class CloudInferenceConfig:
    """Configuration for cloud inference."""
    provider: CloudProvider = CloudProvider.HUGGINGFACE
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    model_id: str = ""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 300  # 5 minutes default
    retry_attempts: int = 3
    retry_delay: float = 2.0


class CloudInferenceManager:
    """
    Manages cloud-based inference for large models.
    
    Features:
    - Multiple provider support (OpenAI, Anthropic, HuggingFace, etc.)
    - Automatic rate limiting and retry logic
    - Response caching for efficiency
    - Cost tracking and optimization
    - Fallback mechanisms
    """
    
    def __init__(self, config: CloudInferenceConfig):
        """Initialize cloud inference manager."""
        self.config = config
        self.session = requests.Session()
        self._setup_provider()
        
        # Cache for responses
        self.response_cache = {}
        self.total_tokens_used = 0
        self.total_api_calls = 0
        
    def _setup_provider(self):
        """Setup provider-specific configuration."""
        # Get API key from environment if not provided
        if not self.config.api_key:
            if self.config.provider == CloudProvider.OPENAI:
                self.config.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.config.provider == CloudProvider.ANTHROPIC:
                self.config.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.config.provider == CloudProvider.HUGGINGFACE:
                # Try environment variable first, then cached token
                self.config.api_key = os.environ.get("HF_TOKEN")
                if not self.config.api_key:
                    from pathlib import Path
                    token_file = Path.home() / ".cache" / "huggingface" / "token"
                    if token_file.exists():
                        self.config.api_key = token_file.read_text().strip()
            elif self.config.provider == CloudProvider.TOGETHER:
                self.config.api_key = os.environ.get("TOGETHER_API_KEY")
            elif self.config.provider == CloudProvider.REPLICATE:
                self.config.api_key = os.environ.get("REPLICATE_API_TOKEN")
        
        # Set default endpoints
        if not self.config.endpoint_url:
            if self.config.provider == CloudProvider.OPENAI:
                self.config.endpoint_url = "https://api.openai.com/v1/chat/completions"
            elif self.config.provider == CloudProvider.ANTHROPIC:
                self.config.endpoint_url = "https://api.anthropic.com/v1/messages"
            elif self.config.provider == CloudProvider.HUGGINGFACE:
                self.config.endpoint_url = f"https://api-inference.huggingface.co/models/{self.config.model_id}"
            elif self.config.provider == CloudProvider.TOGETHER:
                self.config.endpoint_url = "https://api.together.xyz/v1/completions"
            elif self.config.provider == CloudProvider.OLLAMA:
                # Ollama runs locally by default
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                self.config.endpoint_url = f"{ollama_host}/api/generate"
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using cloud inference.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Check cache first
        cache_key = f"{self.config.provider}:{self.config.model_id}:{prompt}"
        if cache_key in self.response_cache:
            logger.info("[CLOUD] Using cached response")
            return self.response_cache[cache_key]
        
        # Prepare request based on provider
        if self.config.provider == CloudProvider.OPENAI:
            response = self._openai_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.ANTHROPIC:
            response = self._anthropic_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.HUGGINGFACE:
            response = self._huggingface_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.TOGETHER:
            response = self._together_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.REPLICATE:
            response = self._replicate_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.OLLAMA:
            response = self._ollama_request(prompt, **kwargs)
        elif self.config.provider == CloudProvider.CUSTOM:
            response = self._custom_request(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # Cache response
        self.response_cache[cache_key] = response
        self.total_api_calls += 1
        
        return response
    
    def _openai_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_id or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }
        
        return self._make_request(headers, data)
    
    def _anthropic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Anthropic API request."""
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_id or "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }
        
        return self._make_request(headers, data)
    
    def _huggingface_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle HuggingFace Inference API request."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "do_sample": True
            }
        }
        
        return self._make_request(headers, data)
    
    def _together_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Together AI API request."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_id,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }
        
        return self._make_request(headers, data)
    
    def _replicate_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Replicate API request."""
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Replicate uses a different API structure
        data = {
            "version": self.config.model_id,
            "input": {
                "prompt": prompt,
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p)
            }
        }
        
        # First create a prediction
        response = self.session.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=data,
            timeout=self.config.timeout
        )
        
        if response.status_code != 201:
            raise Exception(f"Replicate API error: {response.text}")
        
        prediction = response.json()
        prediction_id = prediction["id"]
        
        # Poll for completion
        while True:
            response = self.session.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Replicate API error: {response.text}")
            
            result = response.json()
            
            if result["status"] == "succeeded":
                return {
                    "success": True,
                    "response": result["output"],
                    "provider": "replicate",
                    "model": self.config.model_id,
                    "tokens_used": None  # Replicate doesn't provide token count
                }
            elif result["status"] == "failed":
                raise Exception(f"Replicate prediction failed: {result.get('error')}")
            
            time.sleep(2)  # Wait before polling again
    
    def _ollama_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle Ollama local API request."""
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p)
            }
        }

        try:
            logger.info(f"[OLLAMA] Making request to {self.config.endpoint_url} for model {self.config.model_id}")

            response = self.session.post(
                self.config.endpoint_url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")

                # Ollama provides token counts in eval_count
                tokens = result.get("eval_count", None)
                if tokens:
                    self.total_tokens_used += tokens

                return {
                    "success": True,
                    "response": text,
                    "provider": "ollama",
                    "model": self.config.model_id,
                    "tokens_used": tokens,
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "eval_duration": result.get("eval_duration")
                }
            else:
                logger.warning(f"[OLLAMA] Request failed with status {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "details": response.text
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Is it running? Try: ollama serve"
            }
        except Exception as e:
            logger.error(f"[OLLAMA] Request exception: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _custom_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Handle custom endpoint request."""
        headers = {
            "Content-Type": "application/json"
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }

        return self._make_request(headers, data)
    
    def _make_request(self, headers: Dict, data: Dict) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"[CLOUD] Making request to {self.config.provider} (attempt {attempt + 1})")
                
                response = self.session.post(
                    self.config.endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response based on provider format
                    if self.config.provider == CloudProvider.OPENAI:
                        text = result["choices"][0]["message"]["content"]
                        tokens = result.get("usage", {}).get("total_tokens", 0)
                    elif self.config.provider == CloudProvider.ANTHROPIC:
                        text = result["content"][0]["text"]
                        tokens = result.get("usage", {}).get("output_tokens", 0)
                    elif self.config.provider == CloudProvider.HUGGINGFACE:
                        # HF can return different formats
                        if isinstance(result, list):
                            text = result[0].get("generated_text", "")
                        else:
                            text = result.get("generated_text", "")
                        tokens = None
                    elif self.config.provider == CloudProvider.TOGETHER:
                        text = result["choices"][0]["text"]
                        tokens = result.get("usage", {}).get("total_tokens", 0)
                    else:
                        text = result.get("text", result.get("response", ""))
                        tokens = result.get("tokens", None)
                    
                    if tokens:
                        self.total_tokens_used += tokens
                    
                    return {
                        "success": True,
                        "response": text,
                        "provider": self.config.provider.value,
                        "model": self.config.model_id,
                        "tokens_used": tokens
                    }
                else:
                    logger.warning(f"[CLOUD] Request failed with status {response.status_code}: {response.text}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        return {
                            "success": False,
                            "error": f"API request failed: {response.status_code}",
                            "details": response.text
                        }
                        
            except Exception as e:
                logger.error(f"[CLOUD] Request exception: {str(e)}")
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    return {
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "success": False,
            "error": "Max retry attempts exceeded"
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "cache_size": len(self.response_cache),
            "provider": self.config.provider.value,
            "model": self.config.model_id
        }


def list_ollama_models() -> List[Dict[str, Any]]:
    """
    List available models in local Ollama instance.

    Returns:
        List of model info dictionaries
    """
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        logger.warning(f"Could not list Ollama models: {e}")
        return []


def create_ollama_inference(model_id: str) -> CloudInferenceManager:
    """
    Create an Ollama inference manager for a specific model.

    Args:
        model_id: Ollama model name (e.g., "smollm:360m", "tinyllama", "gemma2:2b")

    Returns:
        Configured CloudInferenceManager for Ollama
    """
    config = CloudInferenceConfig(
        provider=CloudProvider.OLLAMA,
        model_id=model_id,
        timeout=120  # Ollama can be slow on first load
    )
    return CloudInferenceManager(config)


def create_cloud_inference(
    provider: str = "huggingface",
    model_id: str = "",
    api_key: Optional[str] = None
) -> CloudInferenceManager:
    """
    Factory function to create cloud inference manager.
    
    Args:
        provider: Cloud provider name
        model_id: Model identifier
        api_key: API key (uses environment if not provided)
        
    Returns:
        Configured CloudInferenceManager
    """
    config = CloudInferenceConfig(
        provider=CloudProvider(provider.lower()),
        model_id=model_id,
        api_key=api_key
    )
    
    return CloudInferenceManager(config)