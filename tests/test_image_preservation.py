#!/usr/bin/env python3
"""
Test to verify image preservation with very low denoising_strength.
"""

import requests
import time
import base64
from PIL import Image
import io

API_BASE = "http://192.168.0.122:7866"

def load_original_image():
    """Load the original test image."""
    try:
        image = Image.open("original.png")
        print(f"✅ Loaded original.png: {image.size} ({image.mode})")
        return image
    except Exception as e:
        print(f"❌ Error loading original.png: {e}")
        return None

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_image_preservation():
    """Test image preservation with very low denoising_strength."""
    print("=== Testing Image Preservation ===")
    
    # Load the original image
    original_image = load_original_image()
    if original_image is None:
        print("❌ Cannot proceed without original.png")
        return
    
    # Convert to base64
    image_b64 = image_to_base64(original_image)
    print(f"✅ Converted image to base64 (length: {len(image_b64)} characters)")
    
    # Test cases with very low denoising_strength
    test_cases = [
        {
            "name": "Very Low Denoising (0.05)",
            "denoising_strength": 0.05,
            "expected_ssim": 0.9  # Should be very similar
        },
        {
            "name": "Low Denoising (0.1)",
            "denoising_strength": 0.1,
            "expected_ssim": 0.8  # Should be quite similar
        },
        {
            "name": "Medium Denoising (0.3)",
            "denoising_strength": 0.3,
            "expected_ssim": 0.5  # Should be moderately similar
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # Prepare the request
        payload = {
            "prompt": "gorgeous blonde woman, in a kitchen, dark blue lace robe, medium breasts, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
            "negative_prompt": "ugly, old, beedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
            "image": image_b64,
            "steps": 25,
            "cfg_scale": 7.0,
            "denoising_strength": test_case["denoising_strength"],
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
            max_attempts = 60  # 5 minutes max
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
                            
                            # Save with descriptive name
                            output_filename = f"preservation_test_{test_case['denoising_strength']:.2f}.png"
                            result_image.save(output_filename)
                            print(f"💾 Result saved as: {output_filename}")
                            
                            # Basic comparison
                            if result_image.size == original_image.size:
                                print(f"✅ Image dimensions preserved: {result_image.size}")
                            else:
                                print(f"⚠️  Image dimensions changed: {original_image.size} -> {result_image.size}")
                            
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
    test_image_preservation() 