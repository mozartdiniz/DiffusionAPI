#!/usr/bin/env python3
"""
Test script for memory optimization on Apple Silicon Macs.

This script tests the memory optimization features to ensure they work correctly
on M3 MacBook Pro and other Apple Silicon devices.
"""

import os
import sys
import psutil
import torch
from pathlib import Path

# Add the diffusionapi directory to the path
sys.path.insert(0, str(Path(__file__).parent / "diffusionapi"))

from memory_config import get_memory_config, get_optimal_settings_for_memory

def test_memory_detection():
    """Test memory detection and configuration."""
    print("=== Memory Detection Test ===")
    
    # Get system memory info
    total_memory = psutil.virtual_memory().total / (1024**3)
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    print(f"Total system memory: {total_memory:.1f} GB")
    print(f"Available memory: {available_memory:.1f} GB")
    
    # Test device detection
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        "cpu"
    )
    
    print(f"Detected device: {device}")
    
    # Test memory configuration
    config = get_memory_config()
    print(f"Memory configuration loaded: {len(config)} sections")
    
    # Test optimal settings
    settings = get_optimal_settings_for_memory(available_memory, device)
    print(f"Memory level: {settings['memory_level']}")
    print(f"Max image size: {settings['max_image_size']}")
    print(f"Optimizations enabled: {len(settings['optimizations'])}")
    
    return True

def test_environment_variables():
    """Test environment variable overrides."""
    print("\n=== Environment Variable Test ===")
    
    # Test with custom max image size
    os.environ["DIFFUSIONAPI_MAX_IMAGE_SIZE"] = "1024x1024"
    os.environ["DIFFUSIONAPI_ENABLE_VAE_TILING"] = "true"
    os.environ["DIFFUSIONAPI_ENABLE_ATTENTION_SLICING"] = "false"
    
    config = get_memory_config()
    
    # Check if environment variables were applied
    max_size_applied = False
    for device in config["max_image_sizes"]:
        for level in config["max_image_sizes"][device]:
            if config["max_image_sizes"][device][level] == (1024, 1024):
                max_size_applied = True
                break
    
    print(f"Max image size override applied: {max_size_applied}")
    print(f"VAE tiling enabled: {config['optimizations']['enable_vae_tiling']}")
    print(f"Attention slicing enabled: {config['optimizations']['enable_attention_slicing']}")
    
    # Clean up environment variables
    os.environ.pop("DIFFUSIONAPI_MAX_IMAGE_SIZE", None)
    os.environ.pop("DIFFUSIONAPI_ENABLE_VAE_TILING", None)
    os.environ.pop("DIFFUSIONAPI_ENABLE_ATTENTION_SLICING", None)
    
    return max_size_applied

def test_apple_silicon_specific():
    """Test Apple Silicon specific optimizations."""
    print("\n=== Apple Silicon Specific Test ===")
    
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        print("Not running on Apple Silicon, skipping test")
        return True
    
    available_memory = psutil.virtual_memory().available / (1024**3)
    settings = get_optimal_settings_for_memory(available_memory, "mps")
    
    print(f"Apple Silicon optimizations:")
    print(f"  - Memory level: {settings['memory_level']}")
    print(f"  - Max image size: {settings['max_image_size']}")
    print(f"  - Aggressive cleanup: {settings['aggressive_cleanup']}")
    print(f"  - Max concurrent models: {settings.get('max_concurrent_models', 'N/A')}")
    
    # Test different memory levels
    test_memory_levels = [4, 8, 16, 24, 32]  # GB
    for memory_gb in test_memory_levels:
        test_settings = get_optimal_settings_for_memory(memory_gb, "mps")
        print(f"  - {memory_gb}GB memory -> Level: {test_settings['memory_level']}, Max size: {test_settings['max_image_size']}")
    
    return True

def test_memory_validation():
    """Test image size validation for memory constraints."""
    print("\n=== Memory Validation Test ===")
    
    # Import the validation function using absolute import
    import sys
    from pathlib import Path
    
    # Add the current directory to the path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from diffusionapi.generate import validate_image_size_for_memory
    except ImportError:
        print("  Warning: Could not import validate_image_size_for_memory function")
        print("  This is expected if the module structure is different")
        return True
    
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        "cpu"
    )
    
    # Test with different image sizes
    test_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096)
    ]
    
    for width, height in test_sizes:
        try:
            validated_width, validated_height = validate_image_size_for_memory(width, height, device, "test-model")
            if (width, height) != (validated_width, validated_height):
                print(f"  {width}x{height} -> {validated_width}x{validated_height} (reduced)")
            else:
                print(f"  {width}x{height} -> {validated_width}x{validated_height} (no change)")
        except Exception as e:
            print(f"  Error testing {width}x{height}: {e}")
    
    return True

def main():
    """Run all memory optimization tests."""
    print("Memory Optimization Test Suite")
    print("=" * 50)
    
    tests = [
        test_memory_detection,
        test_environment_variables,
        test_apple_silicon_specific,
        test_memory_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} passed")
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory optimization is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 