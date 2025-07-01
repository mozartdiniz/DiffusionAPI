#!/usr/bin/env python3
"""
Test script to verify that the img2img fix works correctly.
"""

import requests
import base64
import json
from PIL import Image
import io
import sys
import time

# Configuration
API_BASE = "http://localhost:7866"

def create_test_image(width=256, height=256):
    """Create a simple test image."""
    image = Image.new('RGB', (width, height), color='blue')
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

def test_img2img_fix():
    """Test img2img fix."""
    print("=== Testing Img2Img Fix ===\n")
    
    # Create a simple test image
    test_image = create_test_image(256, 256)
    image_base64 = image_to_base64(test_image)
    
    # Prepare the request
    payload = {
        "prompt": "a beautiful landscape, high quality, detailed",
        "negative_prompt": "blurry, low quality, distorted",
        "image": f"data:image/png;base64,{image_base64}",
        "steps": 3,  # Use very few steps for quick test
        "cfg_scale": 7.0,
        "denoising_strength": 0.5,
        "width": 256,
        "height": 256,
        "model": "John6666__amanatsu-illustrious-v11-sdxl",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png"
    }
    
    try:
        print("Sending img2img request...")
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get("job_id")
            print(f"✅ Request successful! Job ID: {job_id}")
            
            # Poll for completion
            print("Polling for completion...")
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(2)
                
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    progress = status_data.get("progress", 0)
                    
                    print(f"Status: {status}, Progress: {progress}")
                    
                    if status == "completed":
                        print("✅ Image generation completed successfully!")
                        return True
                    elif status == "error":
                        error_detail = status_data.get("detail", "Unknown error")
                        print(f"❌ Generation failed: {error_detail}")
                        return False
                
                if attempt == max_attempts - 1:
                    print("❌ Timeout waiting for completion")
                    return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_img2img_fix()
    if success:
        print("\n🎉 Img2Img fix test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Img2Img fix test FAILED!")
        sys.exit(1) 