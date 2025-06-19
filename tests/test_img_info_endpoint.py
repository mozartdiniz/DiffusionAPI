#!/usr/bin/env python3
"""
Test script for the img-info endpoint.
This script tests the /img-info endpoint with various image formats and metadata.
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
from diffusionapi.metadata import create_infotext, save_image_with_metadata
import io

# API endpoint URL
API_URL = "http://localhost:8001/img-info"

def create_test_image_with_metadata():
    """Create a test image with metadata and return it as base64."""
    
    # Create a simple test image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # Create metadata
    geninfo = create_infotext(
        prompt="a beautiful sunset over mountains, masterpiece, best quality",
        negative_prompt="blurry, low quality, distorted",
        steps=30,
        sampler_name="Euler a",
        scheduler="Karras",
        cfg_scale=7.0,
        seed=123456789,
        width=512,
        height=512,
        model_name="stable-diffusion-v1-5",
        model_hash="a1b2c3d4e5f6",
        vae_name="vae-ft-mse-840000-ema-pruned",
        vae_hash="f6e5d4c3b2a1",
        denoising_strength=0.75,
        clip_skip=2,
        tiling=False,
        restore_faces=True,
        extra_generation_params={
            "LoRAs": "anime_style:0.8, detail_enhancer:0.6",
            "Memory before": "2048.5 MB",
            "Memory after": "3072.1 MB",
            "Generation time": "12.34s",
        },
        user="test_user",
        version="DiffusionAPI v1.0"
    )
    
    # Save image with metadata to bytes
    img_bytes = io.BytesIO()
    
    # Create PNG info with metadata
    from PIL import PngImagePlugin
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('parameters', geninfo)
    
    # Save to bytes
    image.save(img_bytes, format='PNG', pnginfo=pnginfo)
    img_bytes.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return image_base64, geninfo

def test_img_info_endpoint():
    """Test the img-info endpoint."""
    
    print("=== Testing img-info Endpoint ===\n")
    
    # Test 1: Image with metadata
    print("1. Testing with image containing metadata...")
    
    try:
        # Create test image with metadata
        image_base64, expected_geninfo = create_test_image_with_metadata()
        
        # Send request to endpoint
        payload = {"image": image_base64}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print(f"Status: {result.get('status')}")
            print(f"Info length: {len(result.get('info', ''))}")
            print(f"Parameters found: {len(result.get('parameters', {}))}")
            
            # Check if metadata was extracted correctly
            if result.get('info'):
                print("✓ Metadata extracted successfully")
                print(f"Raw info: {result['info'][:100]}...")
                
                # Check some key parameters
                params = result.get('parameters', {})
                if 'prompt' in params:
                    print(f"✓ Prompt: {params['prompt'][:50]}...")
                if 'Seed' in params:
                    print(f"✓ Seed: {params['Seed']}")
                if 'Steps' in params:
                    print(f"✓ Steps: {params['Steps']}")
            else:
                print("✗ No metadata found")
                
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print()
    
    # Test 2: Image without metadata
    print("2. Testing with image without metadata...")
    
    try:
        # Create a simple image without metadata
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Convert to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Send request
        payload = {"image": image_base64}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print(f"Status: {result.get('status')}")
            
            if not result.get('info'):
                print("✓ Correctly detected no metadata")
            else:
                print("✗ Unexpected metadata found")
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print()
    
    # Test 3: Invalid base64 data
    print("3. Testing with invalid base64 data...")
    
    try:
        payload = {"image": "invalid_base64_data"}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 400:
            print("✓ Correctly rejected invalid base64 data")
            result = response.json()
            print(f"Error: {result.get('detail')}")
        else:
            print(f"✗ Expected 400 error, got {response.status_code}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print()
    
    # Test 4: Missing image data
    print("4. Testing with missing image data...")
    
    try:
        payload = {}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 400:
            print("✓ Correctly rejected missing image data")
            result = response.json()
            print(f"Error: {result.get('detail')}")
        else:
            print(f"✗ Expected 400 error, got {response.status_code}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print()
    
    # Test 5: Data URL format
    print("5. Testing with data URL format...")
    
    try:
        # Create test image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Convert to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Create data URL
        data_url = f"data:image/png;base64,{image_base64}"
        
        # Send request
        payload = {"image": data_url}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            print("✓ Successfully handled data URL format")
            result = response.json()
            print(f"Status: {result.get('status')}")
        else:
            print(f"✗ Failed to handle data URL format: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print("\n=== Test Complete ===")

def test_with_real_image():
    """Test with a real image file if available."""
    
    print("=== Testing with Real Image File ===\n")
    
    # Look for test images created by previous scripts
    test_files = [
        "test_image_with_metadata.png",
        "example_output.png",
        "custom_example.jpg",
        "webui_compatible.png"
    ]
    
    for filename in test_files:
        try:
            if not os.path.exists(filename):
                continue
                
            print(f"Testing with {filename}...")
            
            # Read and encode the image
            with open(filename, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Send request
            payload = {"image": image_base64}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success! Status: {result.get('status')}")
                
                if result.get('info'):
                    print(f"✓ Metadata found: {len(result.get('info', ''))} characters")
                    params = result.get('parameters', {})
                    print(f"✓ Parameters: {len(params)} items")
                    
                    # Show some key parameters
                    if 'prompt' in params:
                        print(f"  Prompt: {params['prompt'][:50]}...")
                    if 'Seed' in params:
                        print(f"  Seed: {params['Seed']}")
                    if 'Steps' in params:
                        print(f"  Steps: {params['Steps']}")
                else:
                    print("✗ No metadata found")
            else:
                print(f"✗ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error testing {filename}: {e}")
        
        print()

if __name__ == "__main__":
    import os
    
    print("Make sure the DiffusionAPI server is running on http://localhost:8000")
    print("You can start it with: python -m uvicorn diffusionapi.main:app --reload")
    print()
    
    # Test the endpoint
    test_img_info_endpoint()
    
    # Test with real image files if available
    test_with_real_image() 