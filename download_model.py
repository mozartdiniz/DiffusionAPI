from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def download_model(model_name="stable-diffusion-v1-5"):
    # Get model path from environment variable, with a default fallback
    models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
    model_id = os.path.join(models_dir, model_name)

    # Define o dispositivo
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Downloading model: {model_id}")
    
    # Carrega o pipeline com o dtype correto
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True
    )

    # Move o modelo para o dispositivo correto
    pipe = pipe.to(device)

    # Gera uma imagem de teste
    print("Generating test image...")
    image = pipe("a fantasy landscape with mountains").images[0]
    image.save("output.png")
    print("Test image saved as output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a Stable Diffusion model')
    parser.add_argument('--model', type=str, default="stable-diffusion-v1-5",
                      help='Name of the model to download (default: stable-diffusion-v1-5)')
    args = parser.parse_args()
    
    download_model(args.model)