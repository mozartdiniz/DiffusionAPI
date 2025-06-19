#!/usr/bin/env python3
"""
Test script to demonstrate metadata functionality.
This script shows how to save and read metadata in the same format as Stable Diffusion web UI.
"""

import os
from PIL import Image
import numpy as np
from diffusionapi.metadata import (
    create_infotext, 
    save_image_with_metadata, 
    read_metadata_from_image,
    parse_infotext
)

def create_test_image(width=512, height=512):
    """Create a test image with some content."""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(255 * x / width),  # Red gradient
                int(255 * y / height), # Green gradient
                128                     # Blue constant
            ]
    
    return Image.fromarray(img_array)

def test_metadata_functionality():
    """Test the metadata functionality with different image formats."""
    
    print("=== Testing DiffusionAPI Metadata Functionality ===\n")
    
    # Create test parameters
    test_params = {
        "prompt": "a beautiful landscape with mountains and trees, masterpiece, best quality",
        "negative_prompt": "blurry, low quality, distorted",
        "steps": 30,
        "sampler_name": "Euler a",
        "scheduler": "Karras",
        "cfg_scale": 7.0,
        "seed": 123456789,
        "width": 512,
        "height": 512,
        "model_name": "stable-diffusion-v1-5",
        "model_hash": "a1b2c3d4e5f6",
        "vae_name": "vae-ft-mse-840000-ema-pruned",
        "vae_hash": "f6e5d4c3b2a1",
        "denoising_strength": 0.75,
        "clip_skip": 2,
        "tiling": False,
        "restore_faces": True,
        "extra_generation_params": {
            "LoRAs": "anime_style:0.8, detail_enhancer:0.6",
            "Memory before": "2048.5 MB",
            "Memory after": "3072.1 MB",
            "Generation time": "12.34s",
        },
        "user": "test_user",
        "version": "DiffusionAPI v1.0"
    }
    
    # Create infotext
    geninfo = create_infotext(**test_params)
    print("Generated infotext:")
    print(geninfo)
    print("\n" + "="*80 + "\n")
    
    # Create test image
    test_image = create_test_image(512, 512)
    
    # Test saving with metadata in different formats
    test_formats = [
        ("test_image_with_metadata.png", ".png"),
        ("test_image_with_metadata.jpg", ".jpg"),
        ("test_image_with_metadata.webp", ".webp"),
    ]
    
    for filename, extension in test_formats:
        print(f"Testing {extension.upper()} format...")
        
        # Save image with metadata
        save_image_with_metadata(
            image=test_image,
            filename=filename,
            geninfo=geninfo,
            extension=extension,
            jpeg_quality=85
        )
        
        # Read back the metadata
        with Image.open(filename) as img:
            read_geninfo, other_metadata = read_metadata_from_image(img)
        
        print(f"  Saved: {filename}")
        print(f"  Read back metadata: {read_geninfo[:100]}...")
        
        # Parse the infotext back to parameters
        if read_geninfo:
            parsed_params = parse_infotext(read_geninfo)
            print(f"  Parsed parameters: {len(parsed_params)} items")
            print(f"  Prompt: {parsed_params.get('prompt', 'N/A')[:50]}...")
            print(f"  Seed: {parsed_params.get('Seed', 'N/A')}")
            print(f"  Steps: {parsed_params.get('Steps', 'N/A')}")
        else:
            print("  No metadata found!")
        
        print()
    
    # Test compatibility with Stable Diffusion web UI format
    print("=== Testing Web UI Compatibility ===\n")
    
    # Example infotext from Stable Diffusion web UI
    webui_infotext = """a beautiful landscape with mountains and trees, masterpiece, best quality
Negative prompt: blurry, low quality, distorted
Steps: 30, Sampler: Euler a, Schedule type: Karras, CFG scale: 7.0, Seed: 123456789, Size: 512x512, Model: stable-diffusion-v1-5, Model hash: a1b2c3d4e5f6, VAE: vae-ft-mse-840000-ema-pruned, VAE hash: f6e5d4c3b2a1, Denoising strength: 0.75, Clip skip: 2, Face restoration: CodeFormer, User: test_user, Version: DiffusionAPI v1.0"""
    
    print("Web UI format infotext:")
    print(webui_infotext)
    print()
    
    # Parse it
    parsed_webui = parse_infotext(webui_infotext)
    print("Parsed Web UI infotext:")
    for key, value in parsed_webui.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")
    print("Check the generated image files to see the embedded metadata!")
    print("You can open these images in Stable Diffusion web UI to verify compatibility.")

if __name__ == "__main__":
    test_metadata_functionality() 