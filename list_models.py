#!/usr/bin/env python3
"""
Simple script to list available models with their user-friendly labels.
"""

from diffusionapi.generate import get_model_labels

if __name__ == "__main__":
    print("🎨 DiffusionAPI - Available Models")
    print("=" * 50)
    
    labels = get_model_labels()
    print("Available models:")
    print("=================")
    
    for model_name, label in sorted(labels.items(), key=lambda x: x[1]):
        print(f"  {label:20} -> {model_name}")
    
    print("\n💡 Usage: Use these friendly labels in your API requests!")
    print("   Example: 'Amanatsu' instead of 'John6666__amanatsu-illustrious-v11-sdxl'") 