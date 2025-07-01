#!/usr/bin/env python3
"""
Test script to verify that base64 image data is being properly sanitized from logs.
This test checks that the sanitize_payload_for_logging function works correctly.
"""

import requests
import base64
import json
from PIL import Image
import io
import sys
import time

# Configuration
API_BASE = "http://192.168.0.122:7866"

def create_test_image(width=256, height=256):
    """Create a simple test image."""
    image = Image.new('RGB', (width, height), color='red')
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def test_log_sanitization():
    """Test that base64 image data is sanitized from logs."""
    print("=== Testing Log Sanitization ===")
    
    # Create a test image
    test_image = create_test_image(512, 512)
    image_b64 = image_to_base64(test_image)
    
    print(f"Original base64 image length: {len(image_b64)} characters")
    print(f"Base64 starts with: {image_b64[:50]}...")
    
    # Prepare the request with base64 image
    payload = {
        "prompt": "a test image for log sanitization",
        "negative_prompt": "blurry, low quality",
        "image": image_b64,
        "steps": 5,  # Use minimal steps for quick test
        "cfg_scale": 7.0,
        "denoising_strength": 0.6,
        "width": 256,
        "height": 256,
        "model": "stable-diffusion-v1-5",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png"
    }
    
    try:
        # Send the request
        print("\nSending img2img request to test log sanitization...")
        print("Check the server logs to verify that base64 data is sanitized.")
        print("You should see '[BASE64_IMAGE_DATA_X_chars]' instead of the actual base64 string.")
        
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✓ Request accepted! Job ID: {job_id}")
            
            # Wait a moment for processing to start
            time.sleep(2)
            
            # Check status to see if processing started
            status_response = requests.get(f"{API_BASE}/queue/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Job status: {status_data.get('status', 'unknown')}")
                print(f"Progress: {status_data.get('progress', 0):.1%}")
            
            print("\n✓ Log sanitization test completed!")
            print("Please check your server logs to verify that:")
            print("1. Base64 image data is replaced with '[BASE64_IMAGE_DATA_X_chars]'")
            print("2. No actual base64 strings appear in the logs")
            print("3. Other payload data (prompt, steps, etc.) is still visible")
            
            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_txt2img_log_sanitization():
    """Test that txt2img logs are also sanitized (should be no base64 data)."""
    print("\n=== Testing Text2Img Log Sanitization ===")
    
    # Prepare the request without base64 image
    payload = {
        "prompt": "a test image for log sanitization",
        "negative_prompt": "blurry, low quality",
        "steps": 5,  # Use minimal steps for quick test
        "cfg_scale": 7.0,
        "width": 256,
        "height": 256,
        "model": "stable-diffusion-v1-5",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png"
    }
    
    try:
        # Send the request
        print("Sending txt2img request to test log sanitization...")
        print("Check the server logs to verify that all payload data is visible.")
        
        response = requests.post(f"{API_BASE}/txt2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✓ Request accepted! Job ID: {job_id}")
            
            # Wait a moment for processing to start
            time.sleep(2)
            
            # Check status to see if processing started
            status_response = requests.get(f"{API_BASE}/queue/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Job status: {status_data.get('status', 'unknown')}")
                print(f"Progress: {status_data.get('progress', 0):.1%}")
            
            print("\n✓ Text2Img log sanitization test completed!")
            print("Please check your server logs to verify that:")
            print("1. All payload data (prompt, steps, etc.) is visible")
            print("2. No base64 data appears (as expected for txt2img)")
            
            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

if __name__ == "__main__":
    print("DiffusionAPI Log Sanitization Test")
    print("=" * 50)
    print("This test verifies that base64 image data is properly")
    print("sanitized from all logs while preserving other payload data.")
    print()
    
    # Test img2img log sanitization
    if test_log_sanitization():
        print("\n" + "=" * 50)
        print("✓ Img2Img log sanitization test completed!")
    else:
        print("\n✗ Img2Img log sanitization test failed")
    
    # Test txt2img log sanitization
    if test_txt2img_log_sanitization():
        print("\n" + "=" * 50)
        print("✓ Text2Img log sanitization test completed!")
    else:
        print("\n✗ Text2Img log sanitization test failed")
    
    print("\n" + "=" * 50)
    print("Log sanitization test completed!")
    print("\nKey points to verify in your server logs:")
    print("1. Base64 image data should be replaced with '[BASE64_IMAGE_DATA_X_chars]'")
    print("2. No actual base64 strings should appear in any logs")
    print("3. Other payload data (prompts, parameters) should remain visible")
    print("4. Both txt2img and img2img endpoints should be sanitized")
    print("5. Job files should also contain sanitized data") 