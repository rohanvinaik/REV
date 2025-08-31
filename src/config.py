"""
Configuration management for REV framework.

Loads configuration from YAML files and environment variables.
Environment variables take precedence and should be prefixed with REV_.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for REV framework.
    
    Loads configuration from config/paths.yaml and environment variables.
    Environment variables prefixed with REV_ override YAML settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (default: config/paths.yaml)
        """
        if config_path is None:
            # Find config relative to this file
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "paths.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_defaults()
        
        with open(self.config_path) as f:
            raw_config = f.read()
        
        # Substitute environment variables
        template = Template(raw_config)
        substituted = template.safe_substitute(os.environ)
        
        # Parse YAML
        config = yaml.safe_load(substituted)
        
        # Expand paths
        config = self._expand_paths(config)
        
        return config
    
    def _expand_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand ~ and relative paths in configuration."""
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._expand_paths(value)
            elif isinstance(value, str) and ('/' in value or '~' in value):
                # Expand user home and make absolute
                expanded = os.path.expanduser(value)
                if not os.path.isabs(expanded) and expanded != value:
                    expanded = os.path.abspath(expanded)
                config[key] = expanded
        return config
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration if no config file exists."""
        return {
            'models': {
                'cache_dir': os.path.expanduser('~/.cache/huggingface/hub'),
                'local_models': './models',
                'test_models': {
                    'tiny': 'sshleifer/tiny-gpt2',
                    'small': 'distilgpt2',
                    'medium': 'gpt2'
                }
            },
            'data': {
                'test_data': './test_data',
                'results': './results',
                'checkpoints': './checkpoints'
            },
            'api': {
                'redis_url': 'redis://localhost:6379/0',
                'host': '0.0.0.0',
                'port': 8000,
                'jwt_secret': 'development-secret-change-in-production'
            },
            'system': {
                'max_segment_memory_mb': 512,
                'device': 'cpu',
                'num_workers': 4,
                'debug': False
            },
            'consensus': {
                'num_validators': 5,
                'fault_tolerance': 1,
                'timeout_ms': 5000
            },
            'hdc': {
                'dimension': 10000,
                'sparsity': 0.01,
                'enable_lut': True,
                'enable_simd': True
            },
            'security': {
                'rate_limit_rps': 100,
                'token_expiry_seconds': 3600,
                'auth_enabled': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.cache_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get path to a model, checking local directory first.
        
        Args:
            model_name: Name of model or path
            
        Returns:
            Full path to model
        """
        # Check if it's already a path
        if '/' in model_name or os.path.exists(model_name):
            return os.path.expanduser(model_name)
        
        # Check local models directory
        local_models = self.get('models.local_models', './models')
        local_path = os.path.join(local_models, model_name)
        if os.path.exists(local_path):
            return local_path
        
        # Check test models
        test_models = self.get('models.test_models', {})
        if model_name in test_models:
            return test_models[model_name]
        
        # Return as-is (probably a HuggingFace model ID)
        return model_name
    
    def setup_logging(self):
        """Configure logging based on configuration."""
        level = self.get('logging.level', 'INFO')
        format_str = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.get('logging.file')
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str
        )
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)
    
    def get_device(self) -> str:
        """Get compute device based on configuration and availability."""
        device = self.get('system.device', 'cpu')
        
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                return 'cpu'
        elif device == 'mps':
            import torch
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, using CPU")
                return 'cpu'
        
        return device
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path})"


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def reset_config(config_path: Optional[str] = None):
    """Reset global configuration with optional new path."""
    global _config
    _config = Config(config_path)
    return _config