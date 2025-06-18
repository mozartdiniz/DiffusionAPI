#!/usr/bin/env python3
"""
List Available Models and LoRAs

This script lists all available models and LoRAs that will be tested for compatibility.
Use this to verify what will be tested before running the full compatibility test.

Usage:
    python list_available_models_loras.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import from diffusionapi
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import after adding to path
try:
    from diffusionapi.generate import get_model_labels, resolve_model_path
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the DiffusionAPI root directory")
    sys.exit(1)

def find_models():
    """Find all available models."""
    models = []
    
    # Get models from the model labels system
    try:
        model_labels = get_model_labels()
        for model_info in model_labels:
            models.append({
                "name": model_info["name"],
                "label": model_info["label"],
                "path": resolve_model_path(model_info["name"])
            })
    except Exception as e:
        print(f"Warning: Could not get model labels: {e}")
    
    # Check the stable_diffusion/models directory
    models_dir = Path("stable_diffusion/models")
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Check if it contains model files
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                if model_files:
                    models.append({
                        "name": model_dir.name,
                        "label": model_dir.name,
                        "path": str(model_dir)
                    })
    
    # Also check the models directory in root
    root_models_dir = Path("models")
    if root_models_dir.exists():
        for model_dir in root_models_dir.iterdir():
            if model_dir.is_dir():
                # Check if it contains model files
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                if model_files:
                    models.append({
                        "name": model_dir.name,
                        "label": model_dir.name,
                        "path": str(model_dir)
                    })
    
    return models

def find_lora_files():
    """Find all LoRA files in the loras directory."""
    loras = []
    
    # Check the stable_diffusion/loras directory first
    loras_dir = Path("stable_diffusion/loras")
    if loras_dir.exists():
        # Common LoRA file extensions
        extensions = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        
        for ext in extensions:
            for lora_file in loras_dir.glob(ext):
                loras.append({
                    "name": lora_file.stem,
                    "path": str(lora_file),
                    "extension": lora_file.suffix
                })
    
    # Also check the root loras directory if it exists
    root_loras_dir = Path("loras")
    if root_loras_dir.exists():
        extensions = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        
        for ext in extensions:
            for lora_file in root_loras_dir.glob(ext):
                loras.append({
                    "name": lora_file.stem,
                    "path": str(lora_file),
                    "extension": lora_file.suffix
                })
    
    if not loras:
        print(f"Warning: No LoRA files found in stable_diffusion/loras or loras directories")
    
    return sorted(loras, key=lambda x: x["name"])

def main():
    print("=" * 60)
    print("AVAILABLE MODELS AND LoRAs")
    print("=" * 60)
    
    # Find models
    models = find_models()
    print(f"\n📁 MODELS ({len(models)} found):")
    print("-" * 40)
    
    if not models:
        print("❌ No models found!")
        print("   Make sure you have models in the 'models' directory or configured in the model labels.")
    else:
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['label']}")
            print(f"     Name: {model['name']}")
            print(f"     Path: {model['path']}")
            print()
    
    # Find LoRAs
    loras = find_lora_files()
    print(f"🎨 LoRAs ({len(loras)} found):")
    print("-" * 40)
    
    if not loras:
        print("❌ No LoRAs found!")
        print("   Make sure you have LoRA files in the 'loras' directory.")
    else:
        for i, lora in enumerate(loras, 1):
            print(f"{i:2d}. {lora['name']}")
            print(f"     File: {lora['path']}")
            print(f"     Type: {lora['extension']}")
            print()
    
    # Summary
    total_combinations = len(models) * len(loras)
    print("=" * 60)
    print("SUMMARY:")
    print(f"   Models: {len(models)}")
    print(f"   LoRAs: {len(loras)}")
    print(f"   Total combinations to test: {total_combinations}")
    
    if total_combinations > 0:
        print(f"\n🚀 Ready to run compatibility test!")
        print(f"   Run: python test_lora_compatibility.py")
        print(f"   This will test all {total_combinations} combinations.")
    else:
        print(f"\n⚠️  No combinations to test!")
        print(f"   Make sure you have both models and LoRAs available.")

if __name__ == "__main__":
    main() 