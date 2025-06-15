import sys
import json
import os
import base64
import time
import psutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

import logging

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import torch

# Use our new upscaler implementation
from .upscalers import upscale_image

# NEW: Real‑ESRGAN support for colour‑safe AnimeSharp & friends
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
# import cv2
import numpy as np

###############################################################################
# Utilities
###############################################################################

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
UPSCALERS_DIR = os.getenv("UPSCALERS_DIR", "stable_diffusion/upscalers")


def is_huggingface_model(model_name: str) -> bool:
    return "/" in model_name


def is_sdxl_model(model_name: str) -> bool:
    sdxl_indicators = ["sdxl", "xl-", "-xl"]
    return any(indicator in model_name.lower() for indicator in sdxl_indicators)


def resolve_model_path(model_name: str) -> str:
    if is_huggingface_model(model_name):
        return model_name
    models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
    return os.path.join(models_dir, model_name)


def update_progress_file(job_id: str, content: dict) -> None:
    os.makedirs("queue", exist_ok=True)
    with open(f"queue/{job_id}.json", "w") as f:
        json.dump(content, f)


def get_memory_usage() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)


def round_to_multiple_of_8(value: int) -> int:
    """Round a value to the nearest multiple of 8."""
    return round(value / 8) * 8

###############################################################################
# Main entry
###############################################################################

def main() -> None:
    logger.info("==> Starting generation")

    # ------------------------------------------------------------------
    # 1.  Read and validate stdin payload
    # ------------------------------------------------------------------
    try:
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
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    steps = int(data.get("steps", 30))
    cfg_scale = float(data.get("cfg_scale", 7.0))
    model_name = data.get("model", "stable-diffusion-v1-5")
    output_path = data.get("output")
    # Round initial dimensions to multiple of 8
    width = round_to_multiple_of_8(int(data.get("width", 512)))
    height = round_to_multiple_of_8(int(data.get("height", 512)))
    hires_cfg = data.get("hires", {}) or {}
    loras = data.get("loras", [])
    sampler_name = data.get("sampler_name", "DPM++ 2M Karras")

    logger.info(f"Initial dimensions (rounded to multiple of 8): {width}x{height}")

    # ------------------------------------------------------------------
    # 2.  Device & dtype
    # ------------------------------------------------------------------
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        "cpu"
    )
    dtype = torch.float16 if device != "cpu" else torch.float32

    update_progress_file(job_id, {"status": "loading", "progress": 0.0})

    # ------------------------------------------------------------------
    # 3.  Load diffusion pipeline (+ optional SDXL refiner)
    # ------------------------------------------------------------------
    model_id = resolve_model_path(model_name)

    logger.info("==> Loading model: %s", model_id)

    if is_sdxl_model(model_name):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None
        )

        # Optional: attach a refiner
        refiner_id = data.get("refiner_checkpoint")
        switch_at = float(data.get("refiner_switch_at", 0.8))
        if refiner_id:
            logger.info("==> Loading SDXL refiner: %s", refiner_id)
            refiner_path = resolve_model_path(refiner_id)
            refiner = StableDiffusionXLPipeline.from_pretrained(
                refiner_path,
                torch_dtype=dtype,
                use_safetensors=True,
                safety_checker=None
            )
            pipe.refiner = refiner
            pipe.refiner_inference_steps = int(steps * (1 - switch_at))
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None
        )

    # ------------------------------------------------------------------
    # 4.  Scheduler selection (quick heuristic)
    # ------------------------------------------------------------------
    scheduler_key = sampler_name.lower().split()[0]
    if "karras" in sampler_name.lower():
        scheduler_key = "karras"

    try:
        if scheduler_key == "karras":
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
        elif scheduler_key == "ddim":
            from diffusers import DDIMScheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "euler":
            from diffusers import EulerDiscreteScheduler
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "euler_a":
            from diffusers import EulerAncestralDiscreteScheduler
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "heun":
            from diffusers import HeunDiscreteScheduler
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "dpm_solver":
            from diffusers import DPMSolverSinglestepScheduler
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "dpm_solver++":
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++"
            )
        elif scheduler_key == "dpm_2":
            from diffusers import KDPM2DiscreteScheduler
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "dpm_2_a":
            from diffusers import KDPM2AncestralDiscreteScheduler
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "lms":
            from diffusers import LMSDiscreteScheduler
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_key == "dpm_fast":
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=False
            )
        elif scheduler_key == "dpm_adaptive":
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
                solver_type="adaptive"
            )
    except Exception as e:
        logger.warning("[!] Scheduler selection failed (%s). Falling back to default.", e)

    pipe = pipe.to(device)
    logger.info("==> Pipeline ready on %s with scheduler %s", device, pipe.scheduler.__class__.__name__)

    # ------------------------------------------------------------------
    # 5.  Callback to update progress
    # ------------------------------------------------------------------
    def callback(i, t, latents):
        if hires_enabled:
            # For hires, we need to track which phase we're in
            if not hasattr(callback, 'phase'):
                callback.phase = 'initial'
            
            if callback.phase == 'initial':
                # Initial generation phase (0-50%)
                progress = round((i + 1) / steps * 0.5, 2)
                update_progress_file(job_id, {
                    "status": "processing",
                    "progress": progress,
                    "phase": "initial generation"
                })
            else:
                # Upscaling phase (50-100%)
                progress = 0.5 + round((i + 1) / hires_steps * 0.5, 2)
                update_progress_file(job_id, {
                    "status": "processing",
                    "progress": progress,
                    "phase": "upscaling"
                })
        else:
            # Normal generation (0-100%)
            progress = round((i + 1) / steps, 2)
            update_progress_file(job_id, {
                "status": "processing",
                "progress": progress,
                "phase": "generation"
            })

    # ------------------------------------------------------------------
    # 6.  Load LoRAs (optional)
    # ------------------------------------------------------------------
    load_loras(pipe, loras)

    # ------------------------------------------------------------------
    # 7.  Generation (single-pass OR hires-fix)
    # ------------------------------------------------------------------
    memory_before = get_memory_usage()
    start_time = time.time()

    hires_enabled = bool(hires_cfg.get("enabled", False))

    if not hires_enabled:
        # ░░ Simple, single pass ░░
        logger.info("==> Standard generation (no hires)")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            callback=callback,
            callback_steps=1,
        ).images[0]
    else:
        # ░░ Hi-Res Fix ░░
        logger.info("==> Hi-Res fix path enabled")

        scale = float(hires_cfg.get("scale", 2.0))
        upscaler_name = hires_cfg.get("upscaler", "Latent")
        hires_steps = int(hires_cfg.get("steps", max(steps // 2, 1)))
        denoising_strength = float(hires_cfg.get("denoising_strength", 0.4))

        # 7a. Generate base image (low-res)
        callback.phase = 'initial'  # Set initial phase
        lowres_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            callback=callback,
            callback_steps=1,
        ).images[0]

        # 7b. Upscale – latent or external
        if upscaler_name.lower() == "latent":
            logger.info("   ↳ Latent upscaling %sx, denoise %.2f", scale, denoising_strength)
            # Calculate target dimensions and round to multiple of 8
            target_width = round_to_multiple_of_8(int(width * scale))
            target_height = round_to_multiple_of_8(int(height * scale))
            logger.info(f"   ↳ Target dimensions (rounded to multiple of 8): {target_width}x{target_height}")
            
            pipe.enable_vae_tiling()
            callback.phase = 'upscaling'  # Set upscaling phase
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=lowres_image,
                strength=denoising_strength,
                num_inference_steps=hires_steps,
                guidance_scale=cfg_scale,
                width=target_width,
                height=target_height,
                callback=callback,
                callback_steps=1,
            ).images[0]
        else:
            logger.info("   ↳ External upscaler '%s' (%sx)", upscaler_name, scale)
            # Debug logging for input image
            lowres_array = np.array(lowres_image)
            logger.info(f"Lowres image before upscale - shape: {lowres_array.shape}, dtype: {lowres_array.dtype}")
            logger.info(f"Lowres image range - min: {lowres_array.min()}, max: {lowres_array.max()}, mean: {lowres_array.mean():.2f}")
            logger.info(f"Lowres image mode: {lowres_image.mode}")
            
            # Check if image is grayscale
            if len(lowres_array.shape) == 2 or (len(lowres_array.shape) == 3 and lowres_array.shape[2] == 1):
                logger.warning("Input image appears to be grayscale! Converting to RGB...")
                lowres_image = lowres_image.convert('RGB')
            
            image = upscale_image(lowres_image, scale=scale, upscaler_name=upscaler_name)
            
            # Debug logging for output image
            upscaled_array = np.array(image)
            logger.info(f"Upscaled image after upscale - shape: {upscaled_array.shape}, dtype: {upscaled_array.dtype}")
            logger.info(f"Upscaled image range - min: {upscaled_array.min()}, max: {upscaled_array.max()}, mean: {upscaled_array.mean():.2f}")
            logger.info(f"Upscaled image mode: {image.mode}")

    # ------------------------------------------------------------------
    # 8.  Final bookkeeping
    # ------------------------------------------------------------------
    end_time = time.time()
    memory_after = get_memory_usage()
    elapsed_time = round(end_time - start_time, 2)

    # Save & base64
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get file type and quality settings from job
    file_type = data.get("file_type", "jpg")
    jpeg_quality = data.get("jpeg_quality", 85)
    
    # Save image with appropriate settings
    if file_type == "jpg":
        image.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)
    else:  # PNG
        image.save(output_path, "PNG", optimize=True)
    
    with open(output_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    update_progress_file(job_id, {
        "status": "done",
        "progress": 1.0,
        "image": image_b64,
        "job_id": job_id,
        "output_path": output_path,
        "file_type": file_type,
        "jpeg_quality": jpeg_quality if file_type == "jpg" else None,
        "width": image.width,
        "height": image.height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "model": model_name,
        "prompt": prompt,
        "loras": loras,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "generation_time_sec": elapsed_time
    })

    logger.info("==> Job %s finished in %.2fs (mem %.1f → %.1f MB)", job_id, elapsed_time, memory_before, memory_after)


def load_loras(pipeline, loras: List[Dict[str, Any]]) -> None:
    """Load LoRAs into the pipeline."""
    if not loras:
        return

    logger.info(f"==> Loading {len(loras)} LoRAs")
    for lora in loras:
        try:
            lora_name = lora["name"]
            lora_scale = lora["scale"]
            # Ensure we have the .safetensors extension
            lora_path = Path(lora["path"])
            if not lora_path.suffix:
                lora_path = lora_path.with_suffix(".safetensors")
            
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
            
            logger.info(f"   ↳ Loading LoRA '{lora_name}' with scale {lora_scale}")
            pipeline.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name=lora_name
            )
            pipeline.fuse_lora(adapter_name=lora_name, scale=lora_scale)
        except Exception as e:
            logger.exception(f"Failed to load LoRA {lora_name}")
            raise RuntimeError(f"LoRA error: {str(e)}")


if __name__ == "__main__":
    main()
