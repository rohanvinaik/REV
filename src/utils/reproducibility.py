"""
Reproducibility and Experiment Tracking
Ensures experiments are reproducible and properly tracked
"""

import os
import sys
import json
import hashlib
import random
import numpy as np
import torch
import platform
import subprocess
import shutil
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import yaml
import logging

# Experiment tracking libraries (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Seed Management
# ============================================================================

@dataclass
class SeedConfig:
    """Configuration for random seed management"""
    master_seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: int
    cuda_deterministic: bool = True
    cuda_benchmark: bool = False
    
    @classmethod
    def from_master_seed(cls, master_seed: int) -> 'SeedConfig':
        """Generate all seeds from a master seed"""
        # Use master seed to generate other seeds deterministically
        rng = random.Random(master_seed)
        return cls(
            master_seed=master_seed,
            python_seed=rng.randint(0, 2**31 - 1),
            numpy_seed=rng.randint(0, 2**31 - 1),
            torch_seed=rng.randint(0, 2**31 - 1)
        )


class SeedManager:
    """Manage random seeds across all libraries"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"No seed provided, using random seed: {seed}")
            
        self.config = SeedConfig.from_master_seed(seed)
        self._original_state = None
        
    def set_all_seeds(self):
        """Set seeds for all random number generators"""
        # Save original state
        self._save_state()
        
        # Python random
        random.seed(self.config.python_seed)
        
        # NumPy
        np.random.seed(self.config.numpy_seed)
        
        # PyTorch
        torch.manual_seed(self.config.torch_seed)
        torch.cuda.manual_seed(self.config.torch_seed)
        torch.cuda.manual_seed_all(self.config.torch_seed)
        
        # CUDA settings for determinism
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.config.cuda_deterministic
            torch.backends.cudnn.benchmark = self.config.cuda_benchmark
            
        # Environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(self.config.python_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.info(f"Set all seeds with master seed: {self.config.master_seed}")
        
    def _save_state(self):
        """Save current RNG states"""
        self._original_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else None
        }
        
    def restore_state(self):
        """Restore original RNG states"""
        if self._original_state:
            random.setstate(self._original_state['python'])
            np.random.set_state(self._original_state['numpy'])
            torch.set_rng_state(self._original_state['torch'])
            
            if self._original_state['torch_cuda'] and torch.cuda.is_available():
                for i, state in enumerate(self._original_state['torch_cuda']):
                    torch.cuda.set_rng_state(state, i)
                    
    def get_config(self) -> Dict[str, Any]:
        """Get seed configuration as dictionary"""
        return asdict(self.config)


# ============================================================================
# Environment Snapshot
# ============================================================================

@dataclass
class EnvironmentSnapshot:
    """Snapshot of the computing environment"""
    timestamp: str
    platform: Dict[str, str]
    python: Dict[str, str]
    packages: Dict[str, str]
    environment_variables: Dict[str, str]
    git_info: Dict[str, str]
    hardware: Dict[str, Any]
    seeds: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save snapshot to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnvironmentSnapshot':
        """Load snapshot from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class EnvironmentCapture:
    """Capture environment information for reproducibility"""
    
    @staticmethod
    def capture(seed_config: Optional[SeedConfig] = None) -> EnvironmentSnapshot:
        """Capture current environment snapshot"""
        
        # Platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        # Python information
        python_info = {
            'version': sys.version,
            'executable': sys.executable,
            'prefix': sys.prefix,
            'path': json.dumps(sys.path)
        }
        
        # Package versions
        packages = {}
        try:
            import pkg_resources
            for dist in pkg_resources.working_set:
                packages[dist.key] = dist.version
        except Exception as e:
            logger.warning(f"Could not capture package versions: {e}")
            
        # Environment variables (filtered for security)
        safe_env_vars = {
            k: v for k, v in os.environ.items()
            if not any(secret in k.upper() for secret in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD'])
        }
        
        # Git information
        git_info = EnvironmentCapture._get_git_info()
        
        # Hardware information
        hardware_info = {
            'cpu_count': os.cpu_count(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        }
        
        # Add memory information
        try:
            import psutil
            hardware_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
            
        # Seeds
        seeds = asdict(seed_config) if seed_config else {}
        
        return EnvironmentSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            platform=platform_info,
            python=python_info,
            packages=packages,
            environment_variables=safe_env_vars,
            git_info=git_info,
            hardware=hardware_info,
            seeds=seeds
        )
        
    @staticmethod
    def _get_git_info() -> Dict[str, str]:
        """Get git repository information"""
        git_info = {}
        
        try:
            # Get current commit hash
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info['commit'] = commit
            
            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info['branch'] = branch
            
            # Check if working directory is clean
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info['clean'] = len(status) == 0
            
            # Get remote URL
            remote = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info['remote'] = remote
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git information not available")
            
        return git_info


# ============================================================================
# Experiment Tracking
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    version: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    seed_config: SeedConfig
    environment: EnvironmentSnapshot
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'tags': self.tags,
            'parameters': self.parameters,
            'seed_config': asdict(self.seed_config),
            'environment': self.environment.to_dict()
        }


class ExperimentTracker:
    """Track experiments with multiple backends"""
    
    def __init__(
        self,
        backend: str = "local",  # local, mlflow, wandb
        project_name: str = "rev-experiments",
        experiment_name: Optional[str] = None,
        config: Optional[ExperimentConfig] = None
    ):
        self.backend = backend
        self.project_name = project_name
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config
        self.run_id = None
        
        # Local tracking directory
        self.local_dir = Path("experiments") / self.experiment_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        self._init_backend()
        
    def _init_backend(self):
        """Initialize tracking backend"""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.project_name)
            mlflow.start_run(run_name=self.experiment_name)
            self.run_id = mlflow.active_run().info.run_id
            
            # Log configuration
            if self.config:
                mlflow.log_params(self.config.parameters)
                for tag in self.config.tags:
                    mlflow.set_tag(tag, True)
                    
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config.parameters if self.config else {},
                tags=self.config.tags if self.config else []
            )
            self.run_id = wandb.run.id
            
        elif self.backend == "local":
            # Generate run ID
            self.run_id = hashlib.sha256(
                f"{self.project_name}_{self.experiment_name}_{datetime.now()}".encode()
            ).hexdigest()[:16]
            
            # Save configuration
            if self.config:
                config_path = self.local_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2, default=str)
                    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.log_params(params)
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.config.update(params)
        else:
            # Local logging
            params_path = self.local_dir / "parameters.json"
            existing = {}
            if params_path.exists():
                with open(params_path, 'r') as f:
                    existing = json.load(f)
            existing.update(params)
            with open(params_path, 'w') as f:
                json.dump(existing, f, indent=2)
                
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
        else:
            # Local logging
            metrics_path = self.local_dir / "metrics.jsonl"
            with open(metrics_path, 'a') as f:
                record = {'step': step, 'timestamp': datetime.utcnow().isoformat()}
                record.update(metrics)
                f.write(json.dumps(record) + '\n')
                
    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None):
        """Log an artifact file"""
        path = Path(path)
        name = name or path.name
        
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(path))
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.save(str(path))
        else:
            # Local logging
            artifact_dir = self.local_dir / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            shutil.copy2(path, artifact_dir / name)
            
    def log_model(self, model: Any, name: str):
        """Log a model"""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            if hasattr(model, 'save_pretrained'):
                # HuggingFace model
                model_dir = self.local_dir / "models" / name
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir)
                mlflow.log_artifact(str(model_dir))
            else:
                # PyTorch model
                mlflow.pytorch.log_model(model, name)
                
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.save(f"{name}/*")
        else:
            # Local logging
            model_dir = self.local_dir / "models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(model_dir)
            else:
                torch.save(model, model_dir / "model.pt")
                
    def finish(self):
        """Finish experiment tracking"""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.end_run()
        elif self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.finish()
            
        # Save final summary
        summary = {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'backend': self.backend,
            'finished_at': datetime.utcnow().isoformat()
        }
        
        with open(self.local_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class Checkpoint:
    """Checkpoint for resumable experiments"""
    step: int
    epoch: int
    state: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    
    def save(self, path: Union[str, Path]):
        """Save checkpoint to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'step': self.step,
            'epoch': self.epoch,
            'state': self.state,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }
        
        torch.save(checkpoint_data, path)
        logger.info(f"Saved checkpoint to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Checkpoint':
        """Load checkpoint from disk"""
        checkpoint_data = torch.load(path)
        return cls(**checkpoint_data)


class CheckpointManager:
    """Manage experiment checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "loss",
        metric_mode: str = "min"  # min or max
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')
        self.checkpoints: List[Path] = []
        
    def save(
        self,
        step: int,
        epoch: int,
        state: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Path:
        """Save a checkpoint"""
        checkpoint = Checkpoint(
            step=step,
            epoch=epoch,
            state=state,
            metrics=metrics,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step{step}.pt"
        checkpoint.save(checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Save best checkpoint
        if self.save_best and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_best = (
                (self.metric_mode == "min" and metric_value < self.best_metric) or
                (self.metric_mode == "max" and metric_value > self.best_metric)
            )
            
            if is_best:
                self.best_metric = metric_value
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                shutil.copy2(checkpoint_path, best_path)
                logger.info(f"Saved best checkpoint with {self.metric_name}={metric_value}")
                
        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                
        return checkpoint_path
        
    def load_latest(self) -> Optional[Checkpoint]:
        """Load the latest checkpoint"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.split('step')[1])
        )
        
        if checkpoints:
            return Checkpoint.load(checkpoints[-1])
        return None
        
    def load_best(self) -> Optional[Checkpoint]:
        """Load the best checkpoint"""
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        if best_path.exists():
            return Checkpoint.load(best_path)
        return None


# ============================================================================
# Docker Support
# ============================================================================

def generate_dockerfile(
    base_image: str = "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    requirements_file: str = "requirements.txt"
) -> str:
    """Generate Dockerfile for reproducible environment"""
    
    dockerfile = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY {requirements_file} .

# Install Python dependencies
RUN pip install --no-cache-dir -r {requirements_file}

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=42

# Entry point
ENTRYPOINT ["python", "run_rev.py"]
"""
    
    return dockerfile


def freeze_requirements(output_file: str = "requirements.txt"):
    """Freeze current environment requirements"""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            
        logger.info(f"Froze requirements to {output_file}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to freeze requirements: {e}")


# ============================================================================
# Reproducibility Manager
# ============================================================================

class ReproducibilityManager:
    """Central manager for reproducibility features"""
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        seed: Optional[int] = None,
        tracking_backend: str = "local",
        checkpoint_dir: Optional[str] = None
    ):
        # Set up seeds
        self.seed_manager = SeedManager(seed)
        self.seed_manager.set_all_seeds()
        
        # Capture environment
        self.environment = EnvironmentCapture.capture(self.seed_manager.config)
        
        # Set up experiment tracking
        self.config = ExperimentConfig(
            name=experiment_name or "rev_experiment",
            version="1.0.0",
            description="REV System Experiment",
            tags=["rev", "fingerprinting"],
            parameters={},
            seed_config=self.seed_manager.config,
            environment=self.environment
        )
        
        self.tracker = ExperimentTracker(
            backend=tracking_backend,
            project_name="rev-system",
            experiment_name=experiment_name,
            config=self.config
        )
        
        # Set up checkpoint manager
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = CheckpointManager(
                self.tracker.local_dir / "checkpoints"
            )
            
    def save_experiment(self, path: Optional[Union[str, Path]] = None):
        """Save complete experiment configuration"""
        if path is None:
            path = self.tracker.local_dir / "experiment.json"
        else:
            path = Path(path)
            
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
            
        # Also save environment snapshot separately
        self.environment.save(path.parent / "environment.json")
        
        logger.info(f"Saved experiment configuration to {path}")
        
    def create_docker_files(self):
        """Create Docker files for reproducibility"""
        # Freeze requirements
        requirements_path = self.tracker.local_dir / "requirements.txt"
        freeze_requirements(str(requirements_path))
        
        # Generate Dockerfile
        dockerfile_content = generate_dockerfile()
        dockerfile_path = self.tracker.local_dir / "Dockerfile"
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
            
        # Create docker-compose.yml
        compose_content = """version: '3.8'

services:
  rev:
    build: .
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
        
        compose_path = self.tracker.local_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
            
        logger.info(f"Created Docker files in {self.tracker.local_dir}")
        
    def finish(self):
        """Finish experiment and save final artifacts"""
        self.save_experiment()
        self.create_docker_files()
        self.tracker.finish()
        logger.info("Experiment finished and all artifacts saved")