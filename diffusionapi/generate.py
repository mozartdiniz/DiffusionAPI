import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub import snapshot_download
import torch

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def is_huggingface_model(model_name):
    return '/' in model_name

def is_sdxl_model(model_name):
    sdxl_indicators = ["sdxl", "xl-", "-xl"]
    return any(indicator in model_name.lower() for indicator in sdxl_indicators)

def resolve_model_path(model_name):
    if is_huggingface_model(model_name):
        return snapshot_download(model_name, local_dir_use_symlinks=False)
    models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
    return os.path.join(models_dir, model_name)

def main():
    data = json.load(sys.stdin)
    print("Received data:", json.dumps(data, indent=2))

    prompt = data["prompt"]
    steps = data.get("steps", 30)
    cfg_scale = data.get("cfg_scale", 7.5)
    model_name = data.get("model", "stable-diffusion-v1-5")
    output_path = data["output"]
    width = data.get("width", 512)
    height = data.get("height", 512)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    model_id = resolve_model_path(model_name)
    print(f"Loading model: {model_id}")
    print(f"Device: {device}, dtype: {dtype}")

    try:
        if is_sdxl_model(model_name):
            print("🧠 Using SDXL pipeline")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None,
            ).to(device)

            # Optional refiner
            refiner_name = data.get("refiner_checkpoint")
            refiner_switch = data.get("refiner_switch_at", 0.8)

            if refiner_name:
                refiner_path = resolve_model_path(refiner_name)
                print(f"🔧 Loading refiner: {refiner_path} (switch at {refiner_switch})")
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16" if dtype == torch.float16 else None,
                ).to(device)

                pipe.refiner = refiner
                pipe.refiner_switch_at = float(refiner_switch)
                print("✅ Refiner attached")

            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height
            ).images[0]

        else:
            print("🧠 Using standard Stable Diffusion pipeline")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            ).to(device)

            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height
            ).images[0]

        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error details: {str(e)}")
        raise

if __name__ == "__main__":
    main()