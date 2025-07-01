#!/usr/bin/env python3
"""
Test script to verify GPU detection works correctly.
"""

import torch
import sys

def test_gpu_detection():
    """Test GPU detection logic."""
    print("=== Testing GPU Detection ===\n")
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    
    # Test the device detection logic
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("✅ CUDA available, using NVIDIA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "cpu"
        dtype = torch.float32
        print("✅ Apple Silicon detected, using CPU for compatibility")
    else:
        device = "cpu"
        dtype = torch.float32
        print("✅ No GPU available, using CPU")
    
    print(f"Selected device: {device}")
    print(f"Selected dtype: {dtype}")
    
    # Test device functionality
    try:
        if device == "cuda":
            test_tensor = torch.zeros(1, device=device)
            print("✅ CUDA device test successful")
        else:
            test_tensor = torch.zeros(1, device=device)
            print("✅ CPU device test successful")
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_gpu_detection()
    if success:
        print("\n🎉 GPU detection test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 GPU detection test FAILED!")
        sys.exit(1) 