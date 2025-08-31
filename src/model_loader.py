"""
Model loader for local LLMs with memory-efficient loading.
"""

import os
import torch
import gc
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import logging

logger = logging.getLogger(__name__)


class LocalModelLoader:
    """Load and manage local LLM models with memory-efficient strategies."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        use_flash_attention: bool = False
    ):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to the local model directory
            device: Device to load model on ("auto", "cuda", "cpu", etc.)
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            max_memory: Memory mapping for multi-GPU setup
            torch_dtype: Data type for model weights
            trust_remote_code: Whether to trust remote code in model config
            use_flash_attention: Whether to use flash attention if available
        """
        self.model_path = Path(model_path)
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_flash_attention = use_flash_attention
        
        self.model = None
        self.tokenizer = None
        self.config = None
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
            
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if needed."""
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
        return None
        
    def _determine_torch_dtype(self) -> torch.dtype:
        """Determine the torch dtype to use."""
        if self.torch_dtype is not None:
            return self.torch_dtype
            
        # Check config for recommended dtype
        config_path = self.model_path / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                dtype_str = config.get("torch_dtype", "float32")
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                return dtype_map.get(dtype_str, torch.float32)
                
        return torch.float32
        
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": str(self.model_path),
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self._determine_torch_dtype(),
            "low_cpu_mem_usage": True,  # Enable low CPU memory usage
        }
        
        # Set device map carefully
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self.device in ["cuda", "mps"]:
            # For GPU, use auto device map
            model_kwargs["device_map"] = "auto"
        else:
            # For CPU, don't set device_map to avoid loading entire model at once
            pass
        
        # Add quantization config if needed
        quantization_config = self._get_quantization_config()
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            
        # Add memory constraints if specified
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory
            
        # Add flash attention if requested and available
        if self.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
            
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try fallback loading without advanced features
            logger.info("Attempting fallback loading...")
            model_kwargs.pop("use_flash_attention_2", None)
            model_kwargs.pop("quantization_config", None)
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
        # Move to eval mode
        self.model.eval()
        
        return self.model, self.tokenizer
        
    def create_pipeline(
        self,
        task: str = "text-generation",
        **pipeline_kwargs
    ):
        """
        Create a transformers pipeline for the model.
        
        Args:
            task: The task for the pipeline
            **pipeline_kwargs: Additional kwargs for the pipeline
            
        Returns:
            Transformers pipeline
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        return pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            **pipeline_kwargs
        )
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        if self.model.device.type != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
        
    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded from memory")


class Yi34BLoader(LocalModelLoader):
    """Specialized loader for Yi-34B model."""
    
    def __init__(
        self,
        model_path: str = "/Users/rohanvinaik/LLM_models/yi-34b",
        **kwargs
    ):
        """
        Initialize Yi-34B loader with optimized settings.
        
        Args:
            model_path: Path to Yi-34B model
            **kwargs: Additional arguments for LocalModelLoader
        """
        # Set Yi-34B specific defaults
        kwargs.setdefault("torch_dtype", torch.float16)  # Use float16 for better MPS compatibility
        kwargs.setdefault("trust_remote_code", True)
        
        # For Mac, use CPU offloading strategy for large model
        if not kwargs.get("device"):
            # Use CPU with careful memory management for Yi-34B
            kwargs["device"] = "cpu"
                
        super().__init__(model_path, **kwargs)
        
    def load_model_for_rev(self, memory_limit_gb: float = 16.0):
        """
        Load Yi-34B optimized for REV pipeline.
        
        Args:
            memory_limit_gb: Memory limit in GB
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Calculate optimal loading strategy based on memory
        model_size_gb = 68  # Yi-34B is ~68GB in bfloat16
        
        if memory_limit_gb < model_size_gb / 4:
            logger.warning(f"Memory limit {memory_limit_gb}GB is very low for Yi-34B")
            self.load_in_4bit = True
        elif memory_limit_gb < model_size_gb / 2:
            logger.info("Using 8-bit quantization for memory efficiency")
            self.load_in_8bit = True
            
        return self.load_model()