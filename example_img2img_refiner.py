#!/usr/bin/env python3
"""
Example usage of img2img endpoint with refiner support.
This example demonstrates how to use the same refiner parameters
for both text2img and img2img endpoints.
"""

import requests
import base64
import json
from PIL import Image
import io
import time

# Configuration
API_BASE = "http://192.168.0.122:7866"

def create_sample_image():
    """Create a sample image for img2img processing."""
    # Create a simple gradient image
    width, height = 512, 512
    image = Image.new('RGB', (width, height))
    
    for y in range(height):
        for x in range(width):
            # Create a gradient from red to blue
            r = int(255 * (1 - x / width))
            g = 0
            b = int(255 * (x / width))
            image.putpixel((x, y), (r, g, b))
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def wait_for_completion(job_id):
    """Wait for job completion and return the result."""
    print(f"Waiting for job {job_id} to complete...")
    
    while True:
        response = requests.get(f"{API_BASE}/queue/{job_id}")
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            progress = data.get('progress', 0)
            phase = data.get('current_phase', 'unknown')
            
            print(f"Progress: {progress:.1%} - Status: {status} - Phase: {phase}")
            
            if status == 'success':
                print("✓ Job completed successfully!")
                return data
            elif status == 'error':
                print(f"✗ Job failed: {data.get('error', 'Unknown error')}")
                return None
            
            time.sleep(1)
        else:
            print(f"Failed to get status: {response.status_code}")
            return None

def example_img2img_with_refiner():
    """Example of using img2img with refiner support."""
    print("=== Img2Img with Refiner Example ===")
    
    # Create a sample input image
    input_image = create_sample_image()
    image_b64 = image_to_base64(input_image)
    
    # Prepare the request with refiner parameters
    # Note: These are the same parameters used for text2img
    payload = {
        "prompt": "a futuristic cityscape with neon lights, cyberpunk style, high quality, detailed",
        "negative_prompt": "blurry, low quality, distorted, ugly",
        "image": image_b64,
        "steps": 25,
        "cfg_scale": 7.5,
        "denoising_strength": 0.7,  # How much to change the input image
        "width": 512,
        "height": 512,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL model required for refiner
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png",
        "jpeg_quality": 95,
        # Refiner parameters (same as text2img)
        "refiner_checkpoint": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "refiner_switch_at": 0.8,  # Switch to refiner at 80% of steps
        "seed": 42  # Optional: for reproducible results
    }
    
    print("Request parameters:")
    print(f"  - Model: {payload['model']}")
    print(f"  - Refiner: {payload['refiner_checkpoint']}")
    print(f"  - Refiner switch at: {payload['refiner_switch_at']}")
    print(f"  - Denoising strength: {payload['denoising_strength']}")
    print(f"  - Steps: {payload['steps']}")
    print(f"  - CFG Scale: {payload['cfg_scale']}")
    
    try:
        # Send the request
        print("\nSending img2img request with refiner...")
        response = requests.post(f"{API_BASE}/img2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✓ Request accepted! Job ID: {job_id}")
            
            # Wait for completion
            final_result = wait_for_completion(job_id)
            
            if final_result:
                print(f"\n✓ Generation completed!")
                print(f"Output path: {final_result.get('output_path')}")
                print(f"Generation time: {final_result.get('generation_time_sec', 'N/A')} seconds")
                print(f"Memory usage: {final_result.get('memory_before_mb', 'N/A')} MB → {final_result.get('memory_after_mb', 'N/A')} MB")
                print(f"Final seed: {final_result.get('seed', 'N/A')}")
                
                # Save the result image
                if 'image' in final_result:
                    image_data = base64.b64decode(final_result['image'])
                    with open('img2img_refiner_result.png', 'wb') as f:
                        f.write(image_data)
                    print("✓ Result image saved as 'img2img_refiner_result.png'")
                
                return True
            else:
                print("✗ Job failed")
                return False
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def example_text2img_with_refiner():
    """Example of using text2img with refiner for comparison."""
    print("\n=== Text2Img with Refiner Example (for comparison) ===")
    
    # Prepare the request with the same refiner parameters
    payload = {
        "prompt": "a futuristic cityscape with neon lights, cyberpunk style, high quality, detailed",
        "negative_prompt": "blurry, low quality, distorted, ugly",
        "steps": 25,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "sampler_name": "DPM++ 2M Karras",
        "fileType": "png",
        "jpeg_quality": 95,
        # Same refiner parameters as img2img
        "refiner_checkpoint": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "refiner_switch_at": 0.8,
        "seed": 42
    }
    
    print("Request parameters (same as img2img except for image input):")
    print(f"  - Model: {payload['model']}")
    print(f"  - Refiner: {payload['refiner_checkpoint']}")
    print(f"  - Refiner switch at: {payload['refiner_switch_at']}")
    print(f"  - Steps: {payload['steps']}")
    print(f"  - CFG Scale: {payload['cfg_scale']}")
    
    try:
        # Send the request
        print("\nSending text2img request with refiner...")
        response = requests.post(f"{API_BASE}/txt2img", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✓ Request accepted! Job ID: {job_id}")
            
            # Wait for completion
            final_result = wait_for_completion(job_id)
            
            if final_result:
                print(f"\n✓ Generation completed!")
                print(f"Output path: {final_result.get('output_path')}")
                print(f"Generation time: {final_result.get('generation_time_sec', 'N/A')} seconds")
                print(f"Memory usage: {final_result.get('memory_before_mb', 'N/A')} MB → {final_result.get('memory_after_mb', 'N/A')} MB")
                print(f"Final seed: {final_result.get('seed', 'N/A')}")
                
                # Save the result image
                if 'image' in final_result:
                    image_data = base64.b64decode(final_result['image'])
                    with open('text2img_refiner_result.png', 'wb') as f:
                        f.write(image_data)
                    print("✓ Result image saved as 'text2img_refiner_result.png'")
                
                return True
            else:
                print("✗ Job failed")
                return False
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

if __name__ == "__main__":
    print("DiffusionAPI Img2Img Refiner Example")
    print("=" * 50)
    print("This example demonstrates how to use refiner support")
    print("with both img2img and text2img endpoints using the")
    print("same API signature.")
    print()
    
    # Run img2img example
    if example_img2img_with_refiner():
        print("\n" + "=" * 50)
        print("✓ Img2Img with refiner example completed successfully!")
    else:
        print("\n✗ Img2Img with refiner example failed")
    
    # Run text2img example for comparison
    if example_text2img_with_refiner():
        print("\n" + "=" * 50)
        print("✓ Text2Img with refiner example completed successfully!")
    else:
        print("\n✗ Text2Img with refiner example failed")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nKey points demonstrated:")
    print("1. Both endpoints accept the same refiner parameters")
    print("2. refiner_checkpoint: specifies the SDXL refiner model")
    print("3. refiner_switch_at: controls when to switch to refiner (0.8 = 80% of steps)")
    print("4. Only SDXL models support refiners")
    print("5. The API signature is consistent between img2img and text2img") 