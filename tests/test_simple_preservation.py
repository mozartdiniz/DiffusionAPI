#!/usr/bin/env python3
"""
Simple test for image preservation with very low denoising_strength.
"""

import requests
import time
import base64
from PIL import Image
import io

API_BASE = "http://192.168.0.122:7866"

def test_simple_preservation():
    """Test simple image preservation."""
    print("=== Simple Image Preservation Test ===")
    
    # Load the original image
    try:
        original_image = Image.open("original.png")
        print(f"✅ Loaded original.png: {original_image.size}")
    except Exception as e:
        print(f"❌ Error loading original.png: {e}")
        return
    
    # Convert to base64
    buffer = io.BytesIO()
    original_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    print(f"✅ Converted image to base64")
    
    # Test with very low denoising_strength
    payload = {
        "prompt": "gorgeous blonde woman, in a kitchen, dark blue lace robe, medium breasts, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
        "negative_prompt": "ugly, old, beedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
        "image": image_b64,
        "steps": 25,
        "cfg_scale": 7.0,
        "denoising_strength": 0.05,  # Very low for maximum preservation
        "resize_mode": "just resize",
        "model": "models--John6666--ilustmix-v6-sdxl",
        "fileType": "png"
    }
    
    print(f"📤 Sending img2img request with denoising_strength: {payload['denoising_strength']}...")
    
    try:
        # Send the request
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        response.raise_for_status()
        
        result = response.json()
        job_id = result.get("job_id")
        print(f"✅ Request successful! Job ID: {job_id}")
        
        # Poll for completion
        print("⏳ Polling for job completion...")
        max_attempts = 30  # 2.5 minutes max
        attempts = 0
        
        while attempts < max_attempts:
            try:
                status_response = requests.get(f"{API_BASE}/queue/{job_id}")
                status_response.raise_for_status()
                status_data = status_response.json()
                
                progress = status_data.get("progress", 0)
                status = status_data.get("status", "unknown")
                
                print(f"📊 Progress: {progress:.1%} - State: {status}")
                
                if status == "done":
                    print("🎉 Job completed successfully!")
                    
                    # Get image data
                    image_b64_result = status_data.get("image")
                    if image_b64_result:
                        # Decode and save the result
                        image_data = base64.b64decode(image_b64_result)
                        result_image = Image.open(io.BytesIO(image_data))
                        
                        # Save result
                        output_filename = "preservation_test_0.05.png"
                        result_image.save(output_filename)
                        print(f"💾 Result saved as: {output_filename}")
                        
                        # Basic comparison
                        if result_image.size == original_image.size:
                            print(f"✅ Image dimensions preserved: {result_image.size}")
                        else:
                            print(f"⚠️  Image dimensions changed: {original_image.size} -> {result_image.size}")
                        
                        print("🔍 Compare preservation_test_0.05.png with original.png to see the preservation effect!")
                        break
                    else:
                        print("❌ No image data in response")
                        break
                elif status == "error":
                    print(f"❌ Job failed: {status_data.get('detail', 'Unknown error')}")
                    break
                
                time.sleep(5)  # Wait 5 seconds between polls
                attempts += 1
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error polling status: {e}")
                attempts += 1
                time.sleep(5)
        
        if attempts >= max_attempts:
            print("❌ Timeout waiting for job completion")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_simple_preservation() 