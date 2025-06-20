#!/usr/bin/env python3
"""
Simple test script for a single img2img case using original.png
"""

import requests
import base64
import json
import time
from PIL import Image
import io
import os

# API endpoint
API_BASE = "http://192.168.0.122:7866"

def load_original_image():
    """Load the original.png image for testing"""
    try:
        # Try to load from current directory first
        if os.path.exists("original.png"):
            image = Image.open("original.png")
            print(f"✅ Loaded original.png: {image.size} ({image.mode})")
            return image
        elif os.path.exists("tests/original.png"):
            image = Image.open("tests/original.png")
            print(f"✅ Loaded tests/original.png: {image.size} ({image.mode})")
            return image
        else:
            print("❌ original.png not found in current directory or tests/")
            print("Please place original.png in the current directory or tests/ directory")
            return None
    except Exception as e:
        print(f"❌ Error loading original.png: {e}")
        return None

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_single_img2img():
    """Test a single img2img case with CFG 3.0"""
    print("=== Testing single img2img case with CFG 3.0 ===")
    
    # Load the original image
    original_image = load_original_image()
    if original_image is None:
        print("❌ Cannot proceed without original.png")
        return
    
    # Convert to base64
    image_b64 = image_to_base64(original_image)
    print(f"✅ Converted image to base64 (length: {len(image_b64)} characters)")
    
    # Test case 1: Standard img2img with just resize
    print("\n--- Test Case 1: Standard img2img with just resize ---")
    payload1 = {
        "prompt": "gorgeous blonde woman, in a kitchen, dark blue lace robe, medium breasts, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
        "negative_prompt": "ugly, old, beedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
        "image": image_b64,
        "steps": 25,
        "cfg_scale": 7.0,
        "denoising_strength": 0.5,  # Moderate transformation - should preserve structure
        "resize_mode": "just resize",
        "model": "models--John6666--ilustmix-v6-sdxl",
        "fileType": "png"
    }
    
    print(f"📤 Sending standard img2img request with denoising_strength: {payload1['denoising_strength']}...")
    run_img2img_test(payload1, "img2img_standard_result.png")
    
    # Test case 2: img2img with resize_by to double the image size
    print("\n--- Test Case 2: img2img with resize_by (2x) ---")
    payload2 = {
        "prompt": "gorgeous blonde woman, in a kitchen, dark blue lace robe, medium breasts, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
        "negative_prompt": "ugly, old, beedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
        "image": image_b64,
        "steps": 25,
        "cfg_scale": 7.0,
        "denoising_strength": 0.5,
        "resize_by": {
            "width": 2.0,
            "height": 2.0
        },
        "model": "models--John6666--ilustmix-v6-sdxl",
        "fileType": "png"
    }
    
    print(f"📤 Sending img2img request with resize_by 2x (denoising_strength: {payload2['denoising_strength']})...")
    print(f"📏 Original size: {original_image.size} -> Expected size: ({original_image.size[0] * 2}, {original_image.size[1] * 2})")
    run_img2img_test(payload2, "img2img_resize_by_2x_result.png")


def run_img2img_test(payload, output_filename):
    """Run a single img2img test with the given payload"""
    try:
        # Send the request
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful! Job ID: {result.get('job_id')}")
            
            # Poll for completion
            job_id = result['job_id']
            print(f"⏳ Polling for job completion...")
            
            max_attempts = 60  # 5 minutes max
            for attempt in range(max_attempts):
                time.sleep(5)  # Wait 5 seconds between polls
                
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    progress = status_data.get('progress', 0)
                    state = status_data.get('state', 'unknown')
                    
                    print(f"📊 Progress: {progress:.1f}% - State: {state}")
                    
                    if state == 'done':
                        print(f"🎉 Job completed successfully!")
                        print(f"📏 Final image size: {status_data.get('width')}x{status_data.get('height')}")
                        print(f"⏱️  Generation time: {status_data.get('generation_time_sec')}s")
                        
                        # Save the image
                        if status_data.get('image'):
                            image_data = base64.b64decode(status_data['image'])
                            with open(output_filename, 'wb') as f:
                                f.write(image_data)
                            print(f"💾 Image saved as: {output_filename}")
                            
                            # Also save the job output path for reference
                            if status_data.get('output_path'):
                                print(f"📁 Also saved at: {status_data.get('output_path')}")
                        
                        print(f"✅ Test completed: {output_filename}")
                        return True
                    elif state == 'error':
                        print(f"❌ Job failed: {status_data.get('error')}")
                        return False
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    return False
            else:
                print("⏰ Job timed out")
                return False
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_single_img2img() 