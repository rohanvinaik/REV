#!/usr/bin/env python3
"""
Download LLaMA 3.1 405B model from Hugging Face.
Requires HuggingFace account with access to Meta's LLaMA models.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
import argparse

def download_llama_405b(token=None, model_id="meta-llama/Llama-3.1-405B", cache_dir="/Users/rohanvinaik/LLM_models"):
    """
    Download LLaMA 3.1 405B model.
    
    Note: This is a MASSIVE model (~800GB) and requires:
    1. Accepted license agreement on HuggingFace
    2. Sufficient disk space
    3. Stable internet connection
    """
    
    print("="*80)
    print("LLaMA 3.1 405B Download Script")
    print("="*80)
    print(f"Model: {model_id}")
    print(f"Destination: {cache_dir}")
    print(f"Estimated size: ~800GB")
    print("="*80)
    
    # Check available space
    import shutil
    stat = shutil.disk_usage(cache_dir)
    available_gb = stat.free / (1024**3)
    
    print(f"\nAvailable disk space: {available_gb:.1f}GB")
    
    if available_gb < 850:
        print("❌ Insufficient disk space! Need at least 850GB free.")
        return False
    
    # Login to HuggingFace if token provided
    if token:
        print("\nLogging in to HuggingFace...")
        login(token=token)
    else:
        print("\n⚠️ No token provided. Trying to use cached credentials...")
        print("If download fails, run: huggingface-cli login")
    
    # Confirm before starting massive download
    response = input("\n⚠️ This will download ~800GB. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Download cancelled.")
        return False
    
    try:
        print("\nStarting download (this will take a LONG time)...")
        print("You can interrupt and resume later if needed.\n")
        
        # Download with resume capability
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=Path(cache_dir) / "llama-3.1-405b",
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # Parallel downloads
        )
        
        print(f"\n✅ Successfully downloaded to: {local_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nPossible issues:")
        print("1. You need to accept the license at: https://huggingface.co/meta-llama/Llama-3.1-405B")
        print("2. You need to be logged in: huggingface-cli login")
        print("3. Network connection issues")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download LLaMA 3.1 405B model")
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-405B", 
                       help="Model ID on HuggingFace")
    parser.add_argument("--cache-dir", default="/Users/rohanvinaik/LLM_models",
                       help="Directory to save the model")
    
    args = parser.parse_args()
    
    # Alternative: Try the Instruct version which might be more accessible
    alt_models = [
        "meta-llama/Llama-3.1-405B",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B",
        "meta-llama/Meta-Llama-3.1-405B-Instruct"
    ]
    
    print("Available model variants:")
    for i, model in enumerate(alt_models, 1):
        print(f"{i}. {model}")
    
    choice = input("\nSelect model variant (1-4) or press Enter for default: ")
    
    if choice and choice.isdigit() and 1 <= int(choice) <= 4:
        model_id = alt_models[int(choice)-1]
    else:
        model_id = args.model_id
    
    success = download_llama_405b(
        token=args.token,
        model_id=model_id,
        cache_dir=args.cache_dir
    )
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. The model is downloaded but will need quantization for practical use")
        print("2. Run REV with 4-bit quantization:")
        print(f"   python run_rev_complete.py {args.cache_dir}/llama-3.1-405b --quantize 4bit --challenges 1")
        print("\n⚠️ WARNING: Even with quantization, this will be VERY slow on CPU")
        print("   Consider using a smaller variant like LLaMA 3.1 70B for testing")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())