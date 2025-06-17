#!/usr/bin/env python3
"""
Example usage of the DiffusionAPI with user-friendly model labels.
"""

import requests
import json

# API base URL
API_BASE = "http://localhost:8000"

def list_models():
    """List all available models with their labels."""
    response = requests.get(f"{API_BASE}/models/aliases")
    if response.status_code == 200:
        data = response.json()
        print("🎨 Available Models:")
        print("=" * 50)
        for model in data["models"]:
            print(f"  {model['label']:20} -> {model['name']}")
    else:
        print(f"Error: {response.status_code}")

def generate_image(prompt, model_label="PlantMilk (HempII)", width=1024, height=1024):
    """Generate an image using a user-friendly model label."""
    
    payload = {
        "prompt": prompt,
        "model": model_label,  # Use the friendly label instead of full name
        "width": width,
        "height": height,
        "steps": 20,
        "cfg_scale": 7.0,
        "seed": -1,  # Random seed
        "fileType": "jpg"
    }
    
    print(f"🎨 Generating image with model '{model_label}'...")
    print(f"📝 Prompt: {prompt}")
    
    # Submit the generation request
    response = requests.post(f"{API_BASE}/txt2img", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        job_id = data["job_id"]
        print(f"✅ Job submitted with ID: {job_id}")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{API_BASE}/queue/{job_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                if status_data["state"] == "done":
                    print(f"✅ Generation completed!")
                    print(f"📊 Progress: {status_data['progress']}%")
                    print(f"⏱️  Generation time: {status_data.get('generation_time_sec', 'N/A')}s")
                    print(f"🌱 Seed used: {status_data.get('seed', 'N/A')}")
                    print(f"💾 Output saved to: {status_data.get('output_path', 'N/A')}")
                    return status_data
                elif status_data["state"] == "error":
                    print(f"❌ Error: {status_data.get('error', 'Unknown error')}")
                    return None
                else:
                    print(f"⏳ Progress: {status_data['progress']}% - {status_data.get('current_phase', 'Processing')}")
            
            import time
            time.sleep(2)  # Wait 2 seconds before checking again
    else:
        print(f"❌ Error submitting job: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    print("🎨 DiffusionAPI - Model Label Example")
    print("=" * 50)
    
    # List available models
    list_models()
    
    print("\n" + "=" * 50)
    print("Example generations:")
    
    # Example 1: PlantMilk HempII
    print("\n1. Generating with PlantMilk (HempII)...")
    result1 = generate_image(
        "a beautiful anime girl with green hair, detailed, high quality",
        model_label="PlantMilk (HempII)"
    )
    
    # Example 2: Amanatsu
    print("\n2. Generating with Amanatsu...")
    result2 = generate_image(
        "a cute anime character, soft lighting, pastel colors",
        model_label="Amanatsu"
    )
    
    # Example 3: Ilustmix
    print("\n3. Generating with Ilustmix...")
    result3 = generate_image(
        "a fantasy landscape with mountains and trees, digital art",
        model_label="Ilustmix"
    )
    
    print("\n✅ All examples completed!")
    print("💡 You can now use these friendly model labels in your applications!") 