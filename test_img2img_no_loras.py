#!/usr/bin/env python3
"""
Test script to verify that img2img works without LoRAs.
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
    image = Image.new('RGB', (width, height), color='red')
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

def test_img2img_without_loras():
    """Test img2img without LoRAs."""
    print("=== Testing Img2Img Without LoRAs ===\n")
    
    # Create a simple test image
    test_image = create_test_image(256, 256)
    image_b64 = image_to_base64(test_image)
    print(f"✅ Created test image: {test_image.size}")
    
    # Prepare the request WITHOUT LoRAs
    payload = {
        "prompt": "a simple test image, high quality",
        "negative_prompt": "blurry, low quality",
        "image": image_b64,
        "steps": 5,  # Use minimal steps for quick test
        "cfg_scale": 7.0,
        "denoising_strength": 0.5,
        "width": 512,
        "height": 512,
        "model": "John6666__amanatsu-illustrious-v11-sdxl",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png",
        # NO LoRAs in the payload
    }
    
    try:
        # Send the request
        print("Sending img2img request WITHOUT LoRAs...")
        print("This should work without any LoRA compatibility issues.")
        
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
                        
                        # Save the image for verification
                        if status_data.get('image'):
                            image_data = base64.b64decode(status_data['image'])
                            output_filename = "img2img_no_loras_test.png"
                            with open(output_filename, 'wb') as f:
                                f.write(image_data)
                            print(f"✅ Image saved as: {output_filename}")
                        
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
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("DiffusionAPI Img2Img No-LoRAs Test")
    print("=" * 50)
    print("This test verifies that img2img works without LoRAs.")
    print("This should prevent the LoRA compatibility issues.")
    print()
    
    # Test img2img without LoRAs
    if test_img2img_without_loras():
        print("\n" + "=" * 50)
        print("✅ Img2Img without LoRAs test completed successfully!")
        print("The img2img endpoint now ignores LoRAs to prevent compatibility issues.")
    else:
        print("\n" + "=" * 50)
        print("❌ Img2Img without LoRAs test failed")
        sys.exit(1) 