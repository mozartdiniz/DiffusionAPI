import sys
import json
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def is_huggingface_model(model_name):
    return "/" in model_name

def is_sdxl_model(model_name):
    sdxl_indicators = ["sdxl", "xl-", "-xl"]
    return any(indicator in model_name.lower() for indicator in sdxl_indicators)

def resolve_model_path(model_name):
    if is_huggingface_model(model_name):
        return model_name
    models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
    return os.path.join(models_dir, model_name)

def update_progress_file(job_id, content):
    os.makedirs("queue", exist_ok=True)
    with open(f"queue/{job_id}.json", "w") as f:
        json.dump(content, f)

def main():
    data = json.load(sys.stdin)

    job_id = data.get("job_id", "unknown")
    prompt = data["prompt"]
    steps = data.get("steps", 30)
    cfg_scale = data.get("cfg_scale", 7.5)
    model_name = data.get("model", "stable-diffusion-v1-5")
    output_path = data["output"]
    width = data.get("width", 512)
    height = data.get("height", 512)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    update_progress_file(job_id, {"status": "loading", "progress": 0.0})

    model_id = resolve_model_path(model_name)

    try:
        if is_sdxl_model(model_name):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )

            refiner_id = data.get("refiner_checkpoint")
            switch_at = data.get("refiner_switch_at", 0.8)

            if refiner_id:
                refiner_path = resolve_model_path(refiner_id)
                refiner = StableDiffusionXLPipeline.from_pretrained(
                    refiner_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                pipe.refiner = refiner
                pipe.refiner_inference_steps = int(steps * (1 - switch_at))

                # Só ativa offload se estiver usando CUDA
                if device == "cuda":
                    pipe.enable_sequential_cpu_offload()

        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )

        pipe = pipe.to(device)

        def callback(i, t, latents):
            progress = round((i + 1) / steps, 2)
            update_progress_file(job_id, {
                "status": "processing",
                "progress": progress
            })

        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            callback=callback,
            callback_steps=1
        ).images[0]

        image.save(output_path)

        with open(output_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        update_progress_file(job_id, {
            "status": "done",
            "progress": 1.0,
            "image": image_b64,
            "job_id": job_id,
            "output_path": output_path,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "model": model_name,
            "prompt": prompt
        })

    except Exception as e:
        update_progress_file(job_id, {
            "status": "error",
            "progress": 0,
            "detail": str(e)
        })
        raise

if __name__ == "__main__":
    main()