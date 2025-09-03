#!/usr/bin/env python3
"""
Diagnostic Fingerprinting Script

Creates lightweight diagnostic fingerprints for each model architecture family
by analyzing the smallest representative model from each family.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime

@dataclass
class ArchitecturalProfile:
    """Diagnostic fingerprint of a model architecture"""
    architecture_family: str
    model_name: str
    model_path: str
    
    # Structural components
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    
    # Architecture specifics
    activation_function: str
    attention_type: str  # "multi_head", "multi_query", "grouped_query"
    has_bias: bool
    
    # Layer patterns
    layer_norm_type: str  # "pre", "post", "sandwich"
    residual_pattern: str  # "standard", "parallel", "sequential"
    
    # Optional fields
    rope_scaling: Optional[Dict] = None
    
    # Special features
    has_vision: bool = False
    is_mixture_of_experts: bool = False
    num_experts: Optional[int] = None
    
    # Diagnostic signature
    structural_hash: str = ""
    scan_timestamp: str = ""
    file_size_gb: float = 0.0


class DiagnosticScanner:
    """Quick diagnostic scanner for model architectures"""
    
    # Smallest representative model for each architecture family
    ARCHITECTURE_REPRESENTATIVES = {
        "gpt2": "gpt2",  # 124M - Classic GPT-2
        "gpt-neo": "gpt-neo-125m",  # 125M - EleutherAI GPT variant
        "pythia": "models--EleutherAI--pythia-70m",  # 70M - Smallest Pythia
        "llama": "llama-2-7b-hf",  # 7B - Smallest Llama
        "falcon": "falcon-7b",  # 7B - Smallest Falcon
        "mistral": "mistral_for_colab",  # Likely 7B
        "mixtral": "mixtral-8x7b-v0.1",  # MoE architecture
        "phi": "phi-2",  # 2.7B - Microsoft's small model
        "qwen": "Qwen2.5-72B-Q4",  # Quantized but represents Qwen
        "yi": "yi-34b",  # 34B - Yi architecture
        "vicuna": "vicuna-7b-v1.5",  # 7B - Vicuna (Llama variant)
        "zephyr": "zephyr-7b-beta-final",  # 7B - Zephyr (Mistral variant)
        "deepseek": "DeepSeek-R1-UD-IQ1_M",  # DeepSeek architecture
        "dialogpt": "models--microsoft--DialoGPT-small",  # Small GPT for dialog
    }
    
    def __init__(self, models_dir: str = "/Users/rohanvinaik/LLM_models"):
        self.models_dir = Path(models_dir)
        self.profiles = {}
        
    def scan_all_architectures(self) -> Dict[str, ArchitecturalProfile]:
        """Scan all unique architectures"""
        print("🔍 Starting Diagnostic Architecture Scan")
        print("=" * 60)
        
        for arch_family, model_name in self.ARCHITECTURE_REPRESENTATIVES.items():
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                print(f"\n📊 Scanning {arch_family} architecture: {model_name}")
                try:
                    profile = self.scan_model(model_path, arch_family)
                    self.profiles[arch_family] = profile
                    self._print_diagnostic(profile)
                except Exception as e:
                    print(f"  ⚠️ Error scanning {model_name}: {e}")
            else:
                print(f"  ⏭️ Skipping {arch_family} - model not found: {model_path}")
        
        return self.profiles
    
    def scan_model(self, model_path: Path, arch_family: str) -> ArchitecturalProfile:
        """Perform diagnostic scan on a model"""
        
        # Look for config.json
        config_path = model_path / "config.json"
        if not config_path.exists():
            # Try finding it in subdirectories
            configs = list(model_path.glob("**/config.json"))
            if configs:
                config_path = configs[0]
            else:
                raise FileNotFoundError(f"No config.json found in {model_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract architectural components
        profile = ArchitecturalProfile(
            architecture_family=arch_family,
            model_name=model_path.name,
            model_path=str(model_path),
            
            # Basic structure
            num_layers=config.get("num_hidden_layers", config.get("n_layer", 0)),
            hidden_size=config.get("hidden_size", config.get("n_embd", 0)),
            num_attention_heads=config.get("num_attention_heads", config.get("n_head", 0)),
            intermediate_size=config.get("intermediate_size", config.get("n_inner", 0)),
            vocab_size=config.get("vocab_size", 0),
            max_position_embeddings=config.get("max_position_embeddings", config.get("n_positions", 0)),
            
            # Architecture details
            activation_function=config.get("hidden_act", config.get("activation_function", "gelu")),
            attention_type=self._detect_attention_type(config),
            has_bias=config.get("use_bias", config.get("bias", False)),
            rope_scaling=config.get("rope_scaling"),
            
            # Layer patterns
            layer_norm_type=self._detect_layer_norm_type(config),
            residual_pattern=self._detect_residual_pattern(config),
            
            # Special features
            has_vision="vision" in model_path.name.lower(),
            is_mixture_of_experts=self._is_moe(config),
            num_experts=config.get("num_local_experts"),
            
            # Metadata
            scan_timestamp=datetime.now().isoformat(),
            file_size_gb=self._get_model_size(model_path)
        )
        
        # Generate structural hash
        profile.structural_hash = self._generate_structural_hash(profile)
        
        return profile
    
    def _detect_attention_type(self, config: Dict) -> str:
        """Detect the type of attention mechanism"""
        if config.get("num_key_value_heads"):
            n_kv = config["num_key_value_heads"]
            n_heads = config.get("num_attention_heads", 0)
            if n_kv == n_heads:
                return "multi_head"
            elif n_kv == 1:
                return "multi_query"
            else:
                return "grouped_query"
        return "multi_head"  # Default
    
    def _detect_layer_norm_type(self, config: Dict) -> str:
        """Detect layer normalization pattern"""
        if config.get("layer_norm_epsilon") or config.get("layer_norm_eps"):
            if config.get("pre_norm", True):
                return "pre"
            return "post"
        return "unknown"
    
    def _detect_residual_pattern(self, config: Dict) -> str:
        """Detect residual connection pattern"""
        if config.get("parallel_residual", False):
            return "parallel"
        return "standard"
    
    def _is_moe(self, config: Dict) -> bool:
        """Check if model is Mixture of Experts"""
        return "num_local_experts" in config or "moe" in config.get("architectures", [""])[0].lower()
    
    def _get_model_size(self, model_path: Path) -> float:
        """Get approximate model size in GB"""
        total_size = 0
        for file_path in model_path.glob("**/*.safetensors"):
            total_size += file_path.stat().st_size
        for file_path in model_path.glob("**/*.bin"):
            total_size += file_path.stat().st_size
        return total_size / (1024**3)  # Convert to GB
    
    def _generate_structural_hash(self, profile: ArchitecturalProfile) -> str:
        """Generate a hash representing the architecture structure"""
        key_components = f"{profile.num_layers}-{profile.hidden_size}-{profile.num_attention_heads}-{profile.attention_type}"
        return hashlib.sha256(key_components.encode()).hexdigest()[:16]
    
    def _print_diagnostic(self, profile: ArchitecturalProfile):
        """Print diagnostic summary"""
        print(f"  ✓ Layers: {profile.num_layers}")
        print(f"  ✓ Hidden: {profile.hidden_size}")
        print(f"  ✓ Heads: {profile.num_attention_heads}")
        print(f"  ✓ Attention: {profile.attention_type}")
        print(f"  ✓ Activation: {profile.activation_function}")
        if profile.is_mixture_of_experts:
            print(f"  ✓ MoE: {profile.num_experts} experts")
        print(f"  ✓ Size: {profile.file_size_gb:.1f}GB")
        print(f"  ✓ Hash: {profile.structural_hash}")
    
    def save_profiles(self, output_path: str = "diagnostic_profiles.json"):
        """Save all profiles to JSON"""
        output_data = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_dir": str(self.models_dir),
                "num_architectures": len(self.profiles)
            },
            "architectures": {}
        }
        
        for arch_name, profile in self.profiles.items():
            output_data["architectures"][arch_name] = asdict(profile)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved diagnostic profiles to {output_path}")
        return output_path
    
    def compare_architectures(self):
        """Generate comparison matrix of architectures"""
        print("\n" + "=" * 60)
        print("ARCHITECTURE COMPARISON MATRIX")
        print("=" * 60)
        
        # Create comparison table
        headers = ["Architecture", "Layers", "Hidden", "Heads", "Attention", "Special"]
        rows = []
        
        for arch_name, profile in sorted(self.profiles.items()):
            special = []
            if profile.is_mixture_of_experts:
                special.append(f"MoE({profile.num_experts})")
            if profile.has_vision:
                special.append("Vision")
            if profile.rope_scaling:
                special.append("RoPE")
                
            rows.append([
                arch_name,
                str(profile.num_layers),
                str(profile.hidden_size),
                str(profile.num_attention_heads),
                profile.attention_type,
                ", ".join(special) if special else "-"
            ])
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def main():
    """Run diagnostic fingerprinting"""
    scanner = DiagnosticScanner()
    
    # Scan all architectures
    profiles = scanner.scan_all_architectures()
    
    # Save profiles
    output_path = scanner.save_profiles("fingerprint_library/diagnostic_profiles.json")
    
    # Show comparison
    scanner.compare_architectures()
    
    print(f"\n✅ Diagnostic scan complete!")
    print(f"📁 Profiles saved to: {output_path}")
    print(f"🏗️ Architectures scanned: {len(profiles)}")
    
    return profiles


if __name__ == "__main__":
    main()