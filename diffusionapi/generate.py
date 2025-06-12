import sys
import json
import os
import base64
import time
import psutil
from pathlib import Path
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import logging

log_path = Path("generation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

LORAS_DIR = os.getenv("LORAS_DIR", "stable_diffusion/Lora")

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

def get_memory_usage():
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)

def main():
    logger.info("==> Starting generation")
    try:
        logger.info("==> Reading JSON from stdin")
        data = json.load(sys.stdin)
    except Exception as e:
        logger.exception("Failed to load JSON from stdin")
        update_progress_file("unknown", {
            "status": "error",
            "progress": 0,
            "detail": f"JSON read error: {str(e)}"
        })
        sys.exit(1)

    job_id = data.get("job_id", "unknown")
    prompt = data["prompt"]
    steps = data.get("steps", 30)
    cfg_scale = data.get("cfg_scale", 7.5)
    model_name = data.get("model", "stable-diffusion-v1-5")
    output_path = data["output"]
    width = data.get("width", 512)
    height = data.get("height", 512)
    logger.info(f"==> before loras")
    loras = data.get("loras", [])
    logger.info(f"==> after loras")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    update_progress_file(job_id, {"status": "loading", "progress": 0.0})

    model_id = resolve_model_path(model_name)
    
    logger.info(f"==> Job received: {job_id}")

    try:
        if is_sdxl_model(model_name):
            logger.info(f"==> Loading SDXL model: {model_id}")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )
            logger.info(f"==> SDXL model loaded: {model_id}")
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

                if device == "cuda":
                    pipe.enable_sequential_cpu_offload()
        else:
            logger.info(f"==> Loading SD model: {model_id}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )
            logger.info(f"==> SD model loaded: {model_id}")

        pipe = pipe.to(device)
        logger.info(f"==> Pipeline moved to {device}")
        
        if loras:
            logger.info(f"==> Loading {len(loras)} LoRAs")
            for lora in loras:
                lora_path = os.path.join(LORAS_DIR, lora["name"])
                if not os.path.exists(lora_path):
                    raise FileNotFoundError(f"LoRA file not found: {lora_path}")
                logger.info(f"==> Applying LoRA: {lora_path} with scale {lora['scale']}")

                try:
                    # Carrega o LoRA
                    pipe.load_lora_weights(lora_path, weight_name=None)

                    # Aplica o peso usando o nome "default"
                    if hasattr(pipe, "set_adapters"):
                        pipe.set_adapters(["default"], adapter_weights={"default": lora["scale"]})
                    else:
                        logger.warning("[!] Warning: set_adapters() not available in this pipeline")

                except Exception as e:
                    logger.exception(f"Unhandled error in job {job_id}")
                    update_progress_file(job_id, {
                        "status": "error",
                        "progress": 0,
                        "detail": str(e)
                    })

        def callback(i, t, latents):
            progress = round((i + 1) / steps, 2)
            update_progress_file(job_id, {
                "status": "processing",
                "progress": progress
            })

        memory_before = get_memory_usage()
        start_time = time.time()

        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            callback=callback,
            callback_steps=1
        ).images[0]

        end_time = time.time()
        memory_after = get_memory_usage()
        elapsed_time = round(end_time - start_time, 2)

        image.save(output_path)
        logger.info(f"==> Image saved to {output_path}")

        with open(output_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        logger.info(f"==> Image base64 encoded")
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
            "prompt": prompt,
            "loras": loras,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "generation_time_sec": elapsed_time
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