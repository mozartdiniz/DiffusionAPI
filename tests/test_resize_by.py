#!/usr/bin/env python3
"""
Test script to verify resize_by functionality
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

def test_resize_by_formats():
    """Test different resize_by formats"""
    print("=== Testing resize_by functionality ===")
    
    # Create a test image (512x512)
    test_image = create_test_image(512, 512)
    image_b64 = image_to_base64(test_image)
    
    # Test cases with different resize_by formats
    test_cases = [
        {
            "name": "resize_by as object with width/height",
            "payload": {
                "prompt": "a beautiful landscape",
                "image": image_b64,
                "steps": 5,  # Quick test
                "denoising_strength": 0.5,
                "resize_by": {"width": 3.0, "height": 3.0}  # Should result in 1536x1536
            },
            "expected_size": (1536, 1536)
        },
        {
            "name": "resize_by as simple number (scale factor)",
            "payload": {
                "prompt": "a beautiful landscape",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resize_by": 2.0  # Should result in 1024x1024
            },
            "expected_size": (1024, 1024)
        },
        {
            "name": "resized_by as object (alternative name)",
            "payload": {
                "prompt": "a beautiful landscape",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resized_by": {"width": 2.5, "height": 2.5}  # Should result in 1280x1280
            },
            "expected_size": (1280, 1280)
        },
        {
            "name": "resized_by as simple number (alternative name)",
            "payload": {
                "prompt": "a beautiful landscape",
                "image": image_b64,
                "steps": 5,
                "denoising_strength": 0.5,
                "resized_by": 1.5  # Should result in 768x768
            },
            "expected_size": (768, 768)
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['name']} ---")
        print(f"Expected size: {test_case['expected_size']}")
        
        try:
            # Send the request
            response = requests.post(f"{API_BASE}/img2img", json=test_case['payload'])
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Request successful! Job ID: {result.get('job_id')}")
                
                # Poll for completion
                job_id = result['job_id']
                print(f"Polling for job completion...")
                
                max_attempts = 30  # 2.5 minutes max
                for attempt in range(max_attempts):
                    time.sleep(5)  # Wait 5 seconds between polls
                    
                    status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        progress = status_data.get('progress', 0)
                        state = status_data.get('state', 'unknown')
                        
                        print(f"Progress: {progress:.1f}% - State: {state}")
                        
                        if state == 'done':
                            actual_width = status_data.get('width')
                            actual_height = status_data.get('height')
                            expected_width, expected_height = test_case['expected_size']
                            
                            print(f"✅ Job completed successfully!")
                            print(f"Expected: {expected_width}x{expected_height}")
                            print(f"Actual: {actual_width}x{actual_height}")
                            
                            if actual_width == expected_width and actual_height == expected_height:
                                print(f"✅ Size matches expected!")
                            else:
                                print(f"❌ Size mismatch!")
                            
                            break
                        elif state == 'error':
                            print(f"❌ Job failed: {status_data.get('error')}")
                            break
                    else:
                        print(f"❌ Failed to get status: {status_response.status_code}")
                        break
                else:
                    print("❌ Job timed out")
                    
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_resize_by_formats() 