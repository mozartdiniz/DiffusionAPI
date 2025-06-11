import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def is_huggingface_model(model_name):
    # Check if the model name contains a forward slash, indicating it's a Hugging Face model
    return '/' in model_name

def is_sdxl_model(model_name):
    """Check if the model is an SDXL model based on its name."""
    sdxl_indicators = ["sdxl", "xl-", "-xl"]
    return any(indicator in model_name.lower() for indicator in sdxl_indicators)

def main():
    data = json.load(sys.stdin)
    print("Received data:", json.dumps(data, indent=2))  # Debug print

    prompt = data["prompt"]
    steps = data.get("steps", 30)
    cfg_scale = data.get("cfg_scale", 7.5)
    
    # Get model name from input data, with a default fallback
    model_name = data.get("model", "stable-diffusion-v1-5")
    print(f"Model name from input: {model_name}")  # Debug print
    
    # Determine if we should use a local path or Hugging Face model ID
    if is_huggingface_model(model_name):
        model_id = model_name  # Use the Hugging Face model ID directly
        print(f"Using Hugging Face model: {model_id}")  # Debug print
    else:
        # Get model path from environment variable for local models
        models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
        print(f"Using local model directory: {models_dir}")  # Debug print
        model_id = os.path.join(models_dir, model_name)
    
    output_path = data["output"]
    width = data.get("width", 512)
    height = data.get("height", 512)

    # Define o dispositivo
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    print(f"Loading model: {model_id}")  # Debug print
    print(f"Device: {device}, dtype: {dtype}")  # Debug print

    try:
        # Choose the appropriate pipeline based on the model
        if is_sdxl_model(model_name):
            print("Using SDXL pipeline")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )
        else:
            print("Using standard Stable Diffusion pipeline")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )

        pipe = pipe.to(device)

        # Geração
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height
        ).images[0]
        
        image.save(output_path)
        print(f"Image saved to: {output_path}")  # Debug print
    except Exception as e:
        print(f"Error details: {str(e)}")  # Debug print
        raise

if __name__ == "__main__":
    main()