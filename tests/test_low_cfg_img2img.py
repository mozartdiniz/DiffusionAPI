#!/usr/bin/env python3
"""
Test script to verify low CFG img2img behavior using original.png
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

def test_low_cfg_img2img():
    """Test img2img with low CFG values using original.png"""
    print("=== Testing low CFG img2img behavior with original.png ===")
    
    # Load the original image
    original_image = load_original_image()
    if original_image is None:
        print("❌ Cannot proceed without original.png")
        return
    
    # Convert to base64
    image_b64 = image_to_base64(original_image)
    print(f"✅ Converted image to base64 (length: {len(image_b64)} characters)")
    
    # Test cases with different CFG values
    test_cases = [
        {
            "name": "Very low CFG (2.0) - should re-render with model style",
            "payload": {
                "prompt": "gorgeous blonde woman, in a kitchen, open dark blue lace robe, artistic nude, medium breasts, pointy nipples, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
                "negative_prompt": "ugly, old, bedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
                "image": image_b64,
                "steps": 20,
                "cfg_scale": 2.0,
                "denoising_strength": 0.3,
                "resize_mode": "just resize",
                "model": "models--John6666--ilustmix-v6-sdxl",
                "fileType": "png"
            }
        },
        {
            "name": "Low CFG (3.0) - should re-render with model style",
            "payload": {
                "prompt": "gorgeous blonde woman, in a kitchen, open dark blue lace robe, artistic nude, medium breasts, pointy nipples, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
                "negative_prompt": "ugly, old, bedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
                "image": image_b64,
                "steps": 20,
                "cfg_scale": 3.0,
                "denoising_strength": 0.3,
                "resize_mode": "just resize",
                "model": "models--John6666--ilustmix-v6-sdxl",
                "fileType": "png"
            }
        },
        {
            "name": "Medium CFG (7.0) - should show more transformation",
            "payload": {
                "prompt": "gorgeous blonde woman, in a kitchen, open dark blue lace robe, artistic nude, medium breasts, pointy nipples, cigarette in mouth, holding coffee mug, photorealistic, morning light, golden hour, soft natural lighting, smooth skin, dreamy mood, seductive, delicate facial features, cinematic lighting, charming and lazy",
                "negative_prompt": "ugly, old, bedroom, bad anatomy, distorted face, deformed body, bad skin texture, masculine features, poorly drawn eyes, multiple limbs, blurry, lowres, watermark, extra fingers, cartoonish, jpeg artifacts, out of frame, cropped, loli, child, fat, grotesque, mutated",
                "image": image_b64,
                "steps": 20,
                "cfg_scale": 7.0,
                "denoising_strength": 0.3,
                "resize_mode": "just resize",
                "model": "models--John6666--ilustmix-v6-sdxl",
                "fileType": "png"
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
                            print(f"✅ Job completed successfully!")
                            print(f"CFG Scale: {test_case['payload']['cfg_scale']}")
                            print(f"Denoising Strength: {test_case['payload']['denoising_strength']}")
                            print(f"Final image size: {status_data.get('width')}x{status_data.get('height')}")
                            print(f"Generation time: {status_data.get('generation_time_sec')}s")
                            
                            # Save the image for comparison
                            if status_data.get('image'):
                                image_data = base64.b64decode(status_data['image'])
                                output_filename = f"img2img_cfg{test_case['payload']['cfg_scale']}_from_original.png"
                                with open(output_filename, 'wb') as f:
                                    f.write(image_data)
                                print(f"✅ Image saved as: {output_filename}")
                                
                                # Also save the job output path for reference
                                if status_data.get('output_path'):
                                    print(f"📁 Also saved at: {status_data.get('output_path')}")
                            
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
    print("Generated images:")
    print("- img2img_cfg2.0_from_original.png (very low CFG)")
    print("- img2img_cfg3.0_from_original.png (low CFG)")
    print("- img2img_cfg7.0_from_original.png (medium CFG)")
    print("\nCompare these with original.png to see the re-rendering effect!")

if __name__ == "__main__":
    test_low_cfg_img2img() 