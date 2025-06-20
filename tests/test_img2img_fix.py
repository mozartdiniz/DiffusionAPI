#!/usr/bin/env python3
"""
Quick test to verify img2img fix
"""

import requests
import base64
import json
import time
from PIL import Image
import io

# API endpoint
API_BASE = "http://localhost:8000"

def create_simple_test_image():
    """Create a simple test image"""
    image = Image.new('RGB', (256, 256), color=(128, 128, 128))
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_img2img_fix():
    """Test img2img with different resize modes"""
    print("=== Testing img2img fix ===")
    
    # Create a test image
    test_image = create_simple_test_image()
    image_b64 = image_to_base64(test_image)
    
    # Test cases
    test_cases = [
        {
            "name": "Default resize mode",
            "payload": {
                "prompt": "a simple test image",
                "image": image_b64,
                "steps": 5,  # Quick test
                "denoising_strength": 0.5
            }
        },
        {
            "name": "Just resize mode",
            "payload": {
                "prompt": "a simple test image",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resize_mode": "just resize"
            }
        },
        {
            "name": "Crop and resize mode",
            "payload": {
                "prompt": "a simple test image",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resize_mode": "crop and resize"
            }
        },
        {
            "name": "Resize and fill mode",
            "payload": {
                "prompt": "a simple test image",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resize_mode": "resize and fill"
            }
        },
        {
            "name": "With resize_to",
            "payload": {
                "prompt": "a simple test image",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resize_to": {"width": 512, "height": 512}
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['name']} ---")
        
        try:
            # Send the request
            response = requests.post(f"{API_BASE}/img2img", json=test_case['payload'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Request successful! Job ID: {result.get('job_id')}")
                
                # Quick status check
                job_id = result['job_id']
                time.sleep(2)  # Wait a bit
                
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    state = status_data.get('state', 'unknown')
                    error = status_data.get('error')
                    
                    if error:
                        print(f"❌ Job failed: {error}")
                    elif state == 'done':
                        print(f"✅ Job completed successfully!")
                    else:
                        print(f"⏳ Job status: {state}")
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_img2img_fix() 