#!/usr/bin/env python3
"""
Test script for img2img functionality
"""

import requests
import base64
import json
import time
from PIL import Image
import io

# API endpoint
API_BASE = "http://localhost:8000"

def create_test_image(width=512, height=512):
    """Create a simple test image"""
    # Create a simple gradient image
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    for x in range(width):
        for y in range(height):
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = 128
            pixels[x, y] = (r, g, b)
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_img2img():
    """Test img2img functionality"""
    print("=== Testing img2img functionality ===")
    
    # Create a test image
    test_image = create_test_image(512, 512)
    image_b64 = image_to_base64(test_image)
    
    # Prepare the request
    payload = {
        "prompt": "a beautiful landscape with mountains and trees, highly detailed",
        "negative_prompt": "blurry, low quality, distorted",
        "image": image_b64,
        "steps": 20,
        "cfg_scale": 7.0,
        "denoising_strength": 0.75,
        "width": 512,
        "height": 512,
        "model": "stable-diffusion-v1-5",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png"
    }
    
    print(f"Payload keys: {list(payload.keys())}")
    print(f"Image data length: {len(image_b64)} characters")
    
    try:
        # Send the request
        print("Sending img2img request...")
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Job ID: {result.get('job_id')}")
            print(f"Status: {result.get('status')}")
            
            # Poll for completion
            job_id = result['job_id']
            print(f"Polling for job completion...")
            
            max_attempts = 60  # 5 minutes max
            for attempt in range(max_attempts):
                time.sleep(5)  # Wait 5 seconds between polls
                
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    progress = status_data.get('progress', 0)
                    state = status_data.get('state', 'unknown')
                    
                    print(f"Progress: {progress:.1f}% - State: {state}")
                    
                    if state == 'done':
                        print("✅ Job completed successfully!")
                        print(f"Final image size: {status_data.get('width')}x{status_data.get('height')}")
                        print(f"Generation time: {status_data.get('generation_time_sec')}s")
                        return True
                    elif state == 'error':
                        print(f"❌ Job failed: {status_data.get('error')}")
                        return False
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    return False
            
            print("❌ Job timed out")
            return False
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

def test_img2img_with_resize():
    """Test img2img with resize parameters"""
    print("\n=== Testing img2img with resize ===")
    
    # Create a test image
    test_image = create_test_image(256, 256)  # Smaller input image
    image_b64 = image_to_base64(test_image)
    
    # Prepare the request with resize
    payload = {
        "prompt": "a futuristic cityscape, neon lights, cyberpunk style",
        "negative_prompt": "blurry, low quality",
        "image": image_b64,
        "steps": 15,
        "cfg_scale": 7.0,
        "denoising_strength": 0.6,
        "resize_mode": "just resize",
        "width": 512,
        "height": 512,
        "model": "stable-diffusion-v1-5",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png"
    }
    
    try:
        # Send the request
        print("Sending img2img request with resize...")
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Job ID: {result.get('job_id')}")
            
            # Poll for completion
            job_id = result['job_id']
            print(f"Polling for job completion...")
            
            max_attempts = 60
            for attempt in range(max_attempts):
                time.sleep(5)
                
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    progress = status_data.get('progress', 0)
                    state = status_data.get('state', 'unknown')
                    
                    print(f"Progress: {progress:.1f}% - State: {state}")
                    
                    if state == 'done':
                        print("✅ Job completed successfully!")
                        print(f"Input size: 256x256, Output size: {status_data.get('width')}x{status_data.get('height')}")
                        return True
                    elif state == 'error':
                        print(f"❌ Job failed: {status_data.get('error')}")
                        return False
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    return False
            
            print("❌ Job timed out")
            return False
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    print("Starting img2img tests...")
    
    # Test basic img2img
    success1 = test_img2img()
    
    # Test img2img with resize
    success2 = test_img2img_with_resize()
    
    print(f"\n=== Test Results ===")
    print(f"Basic img2img: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Img2img with resize: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!") 