import sys
import json
import os
import base64
import time
import psutil
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import random
import io

import logging

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch

# Use our new upscaler implementation
from .upscalers import upscale_image

# NEW: Real‑ESRGAN support for colour‑safe AnimeSharp & friends
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
# import cv2
import numpy as np

# Import the new metadata module
from .metadata import create_infotext, save_image_with_metadata

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

SCHEDULERS = {
    "karras": "Karras",
    "automatic": "Automatic",
    "uniform": "Uniform",
    "exponential": "Exponential",
    "polyexponential": "Polyexponential",
    "sgm": "SGM Uniform",
    "kl": "KL Optimal",
    "align": "Align Your Steps",
    "simple": "Simple",
    "normal": "Normal",
    "ddim": "DDIM",
    "beta": "Beta"
}

def is_huggingface_model(model_name: str) -> bool:
    return "/" in model_name


def is_sdxl_pipeline(pipeline) -> bool:
    """Check if a pipeline is SDXL based on its type."""
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    return isinstance(pipeline, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline))


def is_sdxl_model(model_name: str) -> bool:
    """Check if a model is SDXL by examining its configuration."""
    try:
        # Resolve the model path
        model_id = resolve_model_path(model_name)
        
        # First, check for explicit SDXL indicators in the name
        sdxl_indicators = ["sdxl", "xl-", "-xl"]
        if any(indicator in model_name.lower() for indicator in sdxl_indicators):
            logger.info(f"==> Detected SDXL by name pattern: {model_name}")
            return True
        
        # Check for specific known SDXL models
        known_sdxl_models = [
            "amanatsu-illustrious-v11-sdxl",
            "ilustmix-v6-sdxl",
            "plantmilkmodelsuite",
            "plantmilk"
        ]
        if any(known_model in model_name.lower() for known_model in known_sdxl_models):
            logger.info(f"==> Detected known SDXL model: {model_name}")
            return True
        
        # Try to load the model config using transformers
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Could not load config for {model_name}: {e}")
            # If we can't load the config, be conservative and assume it's NOT SDXL
            return False
        
        # Check if it's an SDXL model by looking at the UNet configuration
        if hasattr(config, 'unet_config'):
            unet_config = config.unet_config
        else:
            # Try to load UNet config directly
            try:
                from transformers import AutoConfig
                unet_config = AutoConfig.from_pretrained(model_id, subfolder="unet")
            except Exception as e:
                logger.warning(f"Could not load UNet config for {model_name}: {e}")
                # If we can't load the UNet config, be conservative and assume it's NOT SDXL
                return False
        
        # SDXL models have specific characteristics in their UNet config
        if hasattr(unet_config, 'cross_attention_dim'):
            # SDXL has cross_attention_dim of 2048 (vs 768 for regular SD)
            is_sdxl = unet_config.cross_attention_dim == 2048
            if is_sdxl:
                logger.info(f"==> Detected SDXL by cross_attention_dim: {unet_config.cross_attention_dim}")
            return is_sdxl
        
        # Additional check for SDXL-specific attributes
        if hasattr(unet_config, 'addition_embed_type'):
            is_sdxl = unet_config.addition_embed_type == "text_time"
            if is_sdxl:
                logger.info(f"==> Detected SDXL by addition_embed_type: {unet_config.addition_embed_type}")
            return is_sdxl
        
        # If we can't determine, be conservative and assume it's NOT SDXL
        logger.info(f"==> Could not determine SDXL status for {model_name}, assuming standard SD")
        return False
        
    except Exception as e:
        logger.warning(f"Error detecting SDXL model {model_name}: {e}")
        # If there's any error, be conservative and assume it's NOT SDXL
        return False


def get_model_labels():
    """Return a dictionary mapping model names to user-friendly labels."""
    return {
        # Actual models in the folder
        "John6666__amanatsu-illustrious-v11-sdxl": "Amanatsu",
        "models--John6666--ilustmix-v6-sdxl": "Ilustmix",
        "models--misri--plantMilkModelSuite_hempII": "PlantMilk (HempII)",
        "models--misri--plantMilkModelSuite_walnut": "PlantMilk (Walnut)",
        "models--Meina--MeinaMix_V11": "MeinaMix V11",
        "models--digiplay--ChikMix_V3": "ChikMix V3",
        "models--mirroring--pastel-mix": "Pastel Mix",
    }

def get_model_name_from_label(label: str) -> str:
    """Get the actual model name from a user-friendly label."""
    labels = get_model_labels()
    
    # Normalize the input label for comparison
    normalized_input = label.lower().replace('-', ' ').replace('_', ' ').strip()
    
    for model_name, model_label in labels.items():
        # Normalize the stored label for comparison
        normalized_label = model_label.lower().replace('-', ' ').replace('_', ' ').strip()
        
        if normalized_label == normalized_input:
            return model_name
    
    return label  # Return the label if no match found

def resolve_model_path(model_name: str) -> str:
    # First check if it's a label and convert to actual model name
    actual_name = get_model_name_from_label(model_name)
    
    if is_huggingface_model(actual_name):
        return actual_name
    models_dir = os.getenv("MODELS_DIR", "stable_diffusion/models")
    return os.path.join(models_dir, actual_name)


def update_progress_file(job_id: str, content: dict) -> None:
    os.makedirs("queue", exist_ok=True)
    with open(f"queue/{job_id}.json", "w") as f:
        json.dump(content, f)


def get_memory_usage() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)


def round_to_multiple_of_8(value: int) -> int:
    """Round value to nearest multiple of 8."""
    return ((value + 4) // 8) * 8

def sanitize_payload_for_logging(data):
    """Remove sensitive data like base64 images from payload for logging."""
    sanitized = data.copy()
    
    # Remove base64 image data
    if "image" in sanitized:
        image_data = sanitized["image"]
        if isinstance(image_data, str) and (image_data.startswith("data:image/") or len(image_data) > 100):
            sanitized["image"] = f"[BASE64_IMAGE_DATA_{len(image_data)}_chars]"
    
    return sanitized

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
        logger.info("==> Received payload:")
        logger.info(json.dumps(sanitize_payload_for_logging(data), indent=2)) 
    except Exception as e:
        logger.exception("Failed to load JSON from stdin")
        update_progress_file("unknown", {
            "status": "error",
            "progress": 0,
            "detail": f"JSON read error: {str(e)}"
        })
        sys.exit(1)

    job_id = data.get("job_id", "unknown")
    job_type = data.get("type", "txt2img")  # Default to txt2img for backward compatibility
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
    scheduler_type = (data.get("scheduler_type") or "Karras").lower()
    scheduler_key = scheduler_type.split()[0]
    scheduler = SCHEDULERS.get(scheduler_key, SCHEDULERS["karras"])
    seed = data.get("seed")
    
    # Img2Img specific parameters
    input_image = None
    denoising_strength = 0.75
    resize_mode = 0  # Default: just resize
    resize_to = None
    resize_by = None
    
    if job_type == "img2img":
        # Validate required img2img fields
        if not data.get("image"):
            logger.error("Img2Img job requires 'image' field")
            update_progress_file(job_id, {
                "status": "error",
                "progress": 0,
                "detail": "Img2Img job requires 'image' field"
            })
            sys.exit(1)
        
        # Decode base64 input image
        try:
            image_data = data.get("image")
            if image_data.startswith("data:image/"):
                # Handle data URL format
                image_data = image_data.split(";")[1].split(",")[1]
            
            input_image_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
            logger.info(f"==> Loaded input image: {input_image.size}")
        except Exception as e:
            logger.error(f"Failed to decode input image: {e}")
            update_progress_file(job_id, {
                "status": "error",
                "progress": 0,
                "detail": f"Failed to decode input image: {str(e)}"
            })
            sys.exit(1)
        
        # Get img2img specific parameters
        denoising_strength = float(data.get("denoising_strength", 0.75))
        resize_mode = int(data.get("resize_mode", 0))
        resize_to = data.get("resize_to")
        resize_by = data.get("resize_by")
        
        # Handle resize parameters
        if resize_to:
            width = round_to_multiple_of_8(int(resize_to.get("width", width)))
            height = round_to_multiple_of_8(int(resize_to.get("height", height)))
            logger.info(f"==> Using resize_to: {resize_to} -> {width}x{height}")
        elif resize_by:
            original_width = input_image.width
            original_height = input_image.height
            width = round_to_multiple_of_8(int(original_width * resize_by.get("width", 1.0)))
            height = round_to_multiple_of_8(int(original_height * resize_by.get("height", 1.0)))
            logger.info(f"==> Using resize_by: {resize_by}")
            logger.info(f"==> Original size: {original_width}x{original_height}")
            logger.info(f"==> Calculated size: {width}x{height}")
        elif resize_mode == 0:  # just resize
            # Use the input image dimensions, rounded to multiple of 8
            width = round_to_multiple_of_8(input_image.width)
            height = round_to_multiple_of_8(input_image.height)
            logger.info(f"==> Using resize_mode 0 (just resize): {width}x{height}")
        elif resize_mode == 1:  # crop and resize
            # Use the input image dimensions, rounded to multiple of 8
            width = round_to_multiple_of_8(input_image.width)
            height = round_to_multiple_of_8(input_image.height)
            logger.info(f"==> Using resize_mode 1 (crop and resize): {width}x{height}")
        elif resize_mode == 2:  # resize and fill
            # Use the input image dimensions, rounded to multiple of 8
            width = round_to_multiple_of_8(input_image.width)
            height = round_to_multiple_of_8(input_image.height)
            logger.info(f"==> Using resize_mode 2 (resize and fill): {width}x{height}")
        else:
            # Fallback: use the input image dimensions, rounded to multiple of 8
            width = round_to_multiple_of_8(input_image.width)
            height = round_to_multiple_of_8(input_image.height)
            logger.info(f"==> Using fallback resize: {width}x{height}")
        
        # Ensure dimensions are never 0
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid dimensions calculated: {width}x{height}, using input image dimensions")
            width = round_to_multiple_of_8(input_image.width)
            height = round_to_multiple_of_8(input_image.height)
        
        logger.info(f"==> Img2Img parameters: denoising_strength={denoising_strength}, resize_mode={resize_mode}")
        logger.info(f"==> Final target dimensions: {width}x{height}")
    
    logger.info(f"==> Model name: {model_name}")
    logger.info(f"==> Is SDXL model: {is_sdxl_model(model_name)}")
    
    # Handle seed conversion from string to int if needed
    if seed is not None:
        try:
            seed = int(seed)  # Convert string to int if needed
        except (ValueError, TypeError):
            seed = None  # If conversion fails, treat as no seed
    
    if seed is None or seed == -1:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed provided or seed == -1, generated random seed: {seed}")
    
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
    # 3.  Pipeline loading (SD vs SDXL detection + loading)
    # ------------------------------------------------------------------
    model_id = resolve_model_path(model_name)
    logger.info(f"==> Loading model: {model_name}")
    logger.info(f"==> Model path: {model_id}")
    logger.info(f"==> Job type: {job_type}")
    
    # Try to determine if this should be an SDXL model
    should_be_sdxl = is_sdxl_model(model_name)
    pipeline_type = "unknown"
    pipe = None
    
    if should_be_sdxl:
        logger.info("==> Detected as SDXL model, attempting SDXL pipeline first")
        try:
            if job_type == "img2img":
                # Use SDXL img2img pipeline for img2img jobs
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                pipeline_type = "sdxl_img2img"
                logger.info("==> Successfully loaded SDXL img2img pipeline with safetensors")
            else:
                # Use SDXL txt2img pipeline for txt2img jobs
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                pipeline_type = "sdxl"
                logger.info("==> Successfully loaded SDXL txt2img pipeline with safetensors")
        except Exception as e:
            logger.warning(f"Failed to load SDXL pipeline with safetensors: {e}")
            try:
                if job_type == "img2img":
                    # Try SDXL img2img pipeline with bin format as fallback
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=False,
                        safety_checker=None
                    )
                    pipeline_type = "sdxl_img2img"
                    logger.info("==> Successfully loaded SDXL img2img pipeline with bin format")
                else:
                    # Try SDXL txt2img pipeline with bin format as fallback
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=False,
                        safety_checker=None
                    )
                    pipeline_type = "sdxl"
                    logger.info("==> Successfully loaded SDXL txt2img pipeline with bin format")
            except Exception as e2:
                logger.warning(f"Failed to load SDXL pipeline with bin format: {e2}")
                logger.info("==> Falling back to standard SD pipeline")
                should_be_sdxl = False
    
    if not should_be_sdxl or pipe is None:
        logger.info("==> Loading as standard SD pipeline")
        try:
            if job_type == "img2img":
                # Use standard SD img2img pipeline for img2img jobs
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                pipeline_type = "standard_img2img"
                logger.info("==> Successfully loaded standard SD img2img pipeline with safetensors")
            else:
                # Use standard SD txt2img pipeline for txt2img jobs
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                pipeline_type = "standard"
                logger.info("==> Successfully loaded standard SD txt2img pipeline with safetensors")
        except Exception as e:
            logger.error(f"Failed to load standard SD pipeline with safetensors: {e}")
            try:
                if job_type == "img2img":
                    # Try standard SD img2img pipeline with bin format as fallback
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=False,
                        safety_checker=None
                    )
                    pipeline_type = "standard_img2img"
                    logger.info("==> Successfully loaded standard SD img2img pipeline with bin format")
                else:
                    # Try standard SD txt2img pipeline with bin format as fallback
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=False,
                        safety_checker=None
                    )
                    pipeline_type = "standard"
                    logger.info("==> Successfully loaded standard SD txt2img pipeline with bin format")
            except Exception as e2:
                logger.error(f"Failed to load standard SD pipeline with bin format: {e2}")
                # If standard SD fails, try SDXL as last resort
                if pipeline_type not in ["sdxl", "sdxl_img2img"]:
                    logger.info("==> Trying SDXL pipeline as last resort")
                    try:
                        if job_type == "img2img":
                            # Try SDXL img2img pipeline with safetensors first
                            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                model_id,
                                torch_dtype=dtype,
                                use_safetensors=True,
                                safety_checker=None
                            )
                            pipeline_type = "sdxl_img2img"
                            logger.info("==> Successfully loaded SDXL img2img pipeline as fallback with safetensors")
                        else:
                            # Try SDXL txt2img pipeline with safetensors first
                            pipe = StableDiffusionXLPipeline.from_pretrained(
                                model_id,
                                torch_dtype=dtype,
                                use_safetensors=True,
                                safety_checker=None
                            )
                            pipeline_type = "sdxl"
                            logger.info("==> Successfully loaded SDXL txt2img pipeline as fallback with safetensors")
                    except Exception as e3:
                        logger.warning(f"Failed to load SDXL pipeline as fallback with safetensors: {e3}")
                        try:
                            if job_type == "img2img":
                                # Try SDXL img2img pipeline with bin format as final fallback
                                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=dtype,
                                    use_safetensors=False,
                                    safety_checker=None
                                )
                                pipeline_type = "sdxl_img2img"
                                logger.info("==> Successfully loaded SDXL img2img pipeline as fallback with bin format")
                            else:
                                # Try SDXL txt2img pipeline with bin format as final fallback
                                pipe = StableDiffusionXLPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=dtype,
                                    use_safetensors=False,
                                    safety_checker=None
                                )
                                pipeline_type = "sdxl"
                                logger.info("==> Successfully loaded SDXL txt2img pipeline as fallback with bin format")
                        except Exception as e4:
                            logger.error(f"Failed to load SDXL pipeline as fallback with bin format: {e4}")
                            raise Exception(f"Could not load model {model_name} as either standard SD or SDXL with any format: {e}")
    
    # Optional: attach a refiner (for SDXL txt2img and img2img pipelines)
    if pipeline_type in ["sdxl", "sdxl_img2img"]:
        refiner_id = data.get("refiner_checkpoint")
        switch_at = float(data.get("refiner_switch_at", 0.8))
        if refiner_id:
            logger.info("==> Loading SDXL refiner: %s", refiner_id)
            refiner_path = resolve_model_path(refiner_id)
            try:
                # Try with safetensors first
                refiner = StableDiffusionXLPipeline.from_pretrained(
                    refiner_path,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    safety_checker=None
                )
                logger.info("==> Successfully loaded SDXL refiner with safetensors")
            except Exception as e:
                logger.warning(f"Failed to load SDXL refiner with safetensors: {e}")
                # Try with bin format as fallback
                refiner = StableDiffusionXLPipeline.from_pretrained(
                    refiner_path,
                    torch_dtype=dtype,
                    use_safetensors=False,
                    safety_checker=None
                )
                logger.info("==> Successfully loaded SDXL refiner with bin format")
            pipe.refiner = refiner
            pipe.refiner_inference_steps = int(steps * (1 - switch_at))

    # ------------------------------------------------------------------
    # 4.  Scheduler selection (quick heuristic)
    # ------------------------------------------------------------------
    logger.info(f"Using scheduler: {scheduler_key} (from scheduler_type: {scheduler_type})")

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
    load_loras(pipe, loras, model_name)

    # ------------------------------------------------------------------
    # 7.  Generation (single-pass OR hires-fix)
    # ------------------------------------------------------------------
    memory_before = get_memory_usage()
    start_time = time.time()

    hires_enabled = bool(hires_cfg.get("enabled", False))
    
    # Define scale and upscaler variables for both paths
    scale = float(hires_cfg.get("scale", 2.0))
    upscaler_name = hires_cfg.get("upscaler", "Latent")
    hires_steps = int(hires_cfg.get("steps", max(steps // 2, 1)))
    hires_denoising_strength = float(hires_cfg.get("denoising_strength", 0.4))

    if not hires_enabled:
        # ░░ Simple, single pass ░░
        if job_type == "img2img":
            logger.info("==> Standard img2img generation (no hires)")
        else:
            logger.info("==> Standard txt2img generation (no hires)")
            
        generator = torch.Generator(device=device).manual_seed(seed)
        
        if job_type == "img2img" and input_image is not None:
            # For img2img, use the dedicated img2img pipeline
            logger.info(f"==> Using img2img pipeline with denoising_strength={denoising_strength}")
            
            # Resize input image to target dimensions if needed
            if input_image.size != (width, height):
                logger.info(f"==> Resizing input image from {input_image.size} to {width}x{height}")
                input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Ensure input image is RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            
            # Prepare kwargs for img2img pipeline call
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": input_image,
                "strength": denoising_strength,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
                "callback": callback,
                "callback_steps": 1,
            }
            
            # Add SDXL-specific parameters if using SDXL img2img pipeline
            if pipeline_type == "sdxl_img2img":
                image_guidance_scale = float(data.get("image_guidance_scale", 1.5))
                pipeline_kwargs["image_guidance_scale"] = image_guidance_scale
                pipeline_kwargs["added_cond_kwargs"] = {"text_embeds": None, "time_ids": None}
                logger.info(f"==> Added SDXL image_guidance_scale: {image_guidance_scale}")
            
            logger.info(f"==> Running img2img pipeline with strength={denoising_strength}, steps={steps}")
            image = pipe(**pipeline_kwargs).images[0]
            logger.info(f"==> Img2img generation complete")
        else:
            # Standard txt2img generation
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "width": width,
                "height": height,
                "generator": generator,
                "callback": callback,
                "callback_steps": 1,
            }
            
            # Add SDXL-specific parameters if using SDXL txt2img pipeline
            if pipeline_type == "sdxl":
                pipeline_kwargs["added_cond_kwargs"] = {"text_embeds": None, "time_ids": None}
                logger.info(f"==> Added SDXL kwargs for txt2img")
            
            image = pipe(**pipeline_kwargs).images[0]
    else:
        # ░░ Hi-Res Fix ░░
        logger.info("==> Hi-Res fix path enabled")

        # 7a. Generate base image (low-res)
        callback.phase = 'initial'  # Set initial phase
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Add img2img specific parameters for initial generation
        if job_type == "img2img" and input_image is not None:
            # Validate dimensions before resizing
            if width <= 0 or height <= 0:
                logger.error(f"Invalid dimensions for resizing: {width}x{height}")
                update_progress_file(job_id, {
                    "status": "error",
                    "progress": 0,
                    "detail": f"Invalid dimensions calculated: {width}x{height}"
                })
                sys.exit(1)
            
            # Resize input image if needed
            if input_image.size != (width, height):
                logger.info(f"==> Resizing input image from {input_image.size} to {width}x{height}")
                input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Ensure input image is RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            
            # For img2img, we need to create a new img2img pipeline for the initial generation
            # since we're using the txt2img pipeline for hires fix
            logger.info(f"==> Creating img2img pipeline for initial generation")
            
            try:
                if pipeline_type == "sdxl":
                    # Create SDXL img2img pipeline for initial generation
                    img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        safety_checker=None
                    )
                    img2img_pipe = img2img_pipe.to(device)
                    # Copy scheduler from main pipeline
                    img2img_pipe.scheduler = pipe.scheduler
                    logger.info(f"==> Created SDXL img2img pipeline for initial generation")
                else:
                    # Create standard SD img2img pipeline for initial generation
                    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        safety_checker=None
                    )
                    img2img_pipe = img2img_pipe.to(device)
                    # Copy scheduler from main pipeline
                    img2img_pipe.scheduler = pipe.scheduler
                    logger.info(f"==> Created standard SD img2img pipeline for initial generation")
                
                # Load LoRAs into the img2img pipeline
                load_loras(img2img_pipe, loras, model_name)
                
                # Attach refiner to img2img pipeline if available
                if hasattr(pipe, 'refiner') and pipe.refiner is not None:
                    logger.info("==> Attaching refiner to img2img pipeline")
                    img2img_pipe.refiner = pipe.refiner
                    img2img_pipe.refiner_inference_steps = pipe.refiner_inference_steps
                
                # Prepare kwargs for img2img initial generation
                initial_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": input_image,
                    "strength": denoising_strength,
                    "num_inference_steps": steps,
                    "guidance_scale": cfg_scale,
                    "generator": generator,
                    "callback": callback,
                    "callback_steps": 1,
                }
                
                # Add SDXL-specific parameters if using SDXL img2img pipeline
                if pipeline_type == "sdxl":
                    image_guidance_scale = float(data.get("image_guidance_scale", 1.5))
                    initial_kwargs["image_guidance_scale"] = image_guidance_scale
                    initial_kwargs["added_cond_kwargs"] = {"text_embeds": None, "time_ids": None}
                    logger.info(f"==> Added SDXL image_guidance_scale for initial generation: {image_guidance_scale}")
                
                logger.info(f"==> Running img2img pipeline for initial generation with strength={denoising_strength}")
                image = img2img_pipe(**initial_kwargs).images[0]
                logger.info(f"==> Initial img2img generation complete")
                
            except Exception as e:
                logger.error(f"Failed to create img2img pipeline for initial generation: {e}")
                # Fallback to using the main pipeline with img2img approach
                logger.info(f"==> Falling back to main pipeline with img2img approach")
                
                initial_kwargs.update({
                    "image": input_image,
                    "strength": denoising_strength,
                })
                
                # Add SDXL-specific parameters if using SDXL model
                if pipeline_type == "sdxl":
                    initial_kwargs["added_cond_kwargs"] = {
                        "text_embeds": None,
                        "time_ids": None
                    }
                    image_guidance_scale = float(data.get("image_guidance_scale", 1.5))
                    initial_kwargs["image_guidance_scale"] = image_guidance_scale
                    logger.info(f"==> Added SDXL image_guidance_scale for fallback: {image_guidance_scale}")
                
                image = pipe(**initial_kwargs).images[0]
        else:
            # Standard txt2img generation for initial hires fix
            # Prepare kwargs for initial pipeline call
            initial_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "width": width,
                "height": height,
                "generator": generator,
                "callback": callback,
                "callback_steps": 1,
            }
            
            # Add SDXL-specific parameters if using SDXL model
            if pipeline_type == "sdxl":
                initial_kwargs["added_cond_kwargs"] = {
                    "text_embeds": None,
                    "time_ids": None
                }
                logger.info(f"==> Added SDXL kwargs for txt2img initial generation")

            # Log all pipeline parameters for debugging
            logger.info(f"==> Pipeline parameters:")
            logger.info(f"   - Prompt: {prompt[:100]}...")
            logger.info(f"   - Negative prompt: {negative_prompt[:100]}...")
            logger.info(f"   - Steps: {steps}")
            logger.info(f"   - CFG Scale: {cfg_scale}")
            logger.info(f"   - Width: {width}")
            logger.info(f"   - Height: {height}")
            if job_type == "img2img":
                logger.info(f"   - Denoising strength: {denoising_strength}")
                logger.info(f"   - Input image size: {input_image.size if input_image else 'None'}")
            logger.info(f"   - Scheduler: {pipe.scheduler.__class__.__name__}")
            logger.info(f"   - Model: {model_name}")

            image = pipe(**initial_kwargs).images[0]

    # ------------------------------------------------------------------
    # 7.  Upscale – latent or external
    # ------------------------------------------------------------------
    if scale > 1.0 and hires_enabled:
        logger.info("   ↳ Upscaling %sx", scale)
        
        if upscaler_name.lower() == "latent":
            logger.info("   ↳ Latent upscaling %sx, denoise %.2f", scale, hires_denoising_strength)
            # Calculate target dimensions and round to multiple of 8
            target_width = round_to_multiple_of_8(int(width * scale))
            target_height = round_to_multiple_of_8(int(height * scale))
            logger.info(f"   ↳ Target dimensions (rounded to multiple of 8): {target_width}x{target_height}")
            
            callback.phase = 'upscaling'  # Set upscaling phase
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # For SDXL img2img pipelines, we need to handle large images carefully
            if pipeline_type == "sdxl_img2img" and (target_width > 1024 or target_height > 1024):
                logger.info("   ↳ Large image detected for SDXL img2img, using external upscaler instead")
                # Use external upscaler for large SDXL img2img images to avoid VAE issues
                image = upscale_image(image, scale=scale, upscaler_name="Latent")
            else:
                # Use the main pipeline for upscaling
                pipe.enable_vae_tiling()
                
                # Prepare kwargs for upscaling pipeline call
                upscale_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": image,
                    "strength": hires_denoising_strength,
                    "num_inference_steps": hires_steps,
                    "guidance_scale": cfg_scale,
                    "width": target_width,
                    "height": target_height,
                    "generator": generator,
                    "callback": callback,
                    "callback_steps": 1,
                }
                
                # Add SDXL-specific parameters if using SDXL model
                if pipeline_type in ["sdxl", "sdxl_img2img"]:
                    upscale_kwargs["added_cond_kwargs"] = {
                        "text_embeds": None,
                        "time_ids": None
                    }
                    
                    # For SDXL models, add image_guidance_scale for better fidelity
                    image_guidance_scale = float(data.get("image_guidance_scale", 1.5))
                    upscale_kwargs["image_guidance_scale"] = image_guidance_scale
                    logger.info(f"==> Added SDXL image_guidance_scale for upscaling: {image_guidance_scale}")
                
                logger.info(f"==> Hires upscaling parameters: strength={hires_denoising_strength}")
                
                # Generate the upscaled image
                image = pipe(**upscale_kwargs).images[0]
        else:
            logger.info("   ↳ External upscaler '%s' (%sx)", upscaler_name, scale)
            # Debug logging for input image
            lowres_array = np.array(image)
            logger.info(f"Lowres image before upscale - shape: {lowres_array.shape}, dtype: {lowres_array.dtype}")
            logger.info(f"Lowres image range - min: {lowres_array.min()}, max: {lowres_array.max()}, mean: {lowres_array.mean():.2f}")
            logger.info(f"Lowres image mode: {image.mode}")
            
            # Check if image is grayscale
            if len(lowres_array.shape) == 2 or (len(lowres_array.shape) == 3 and lowres_array.shape[2] == 1):
                logger.warning("Input image appears to be grayscale! Converting to RGB...")
                image = image.convert('RGB')
            
            generator = torch.Generator(device=device).manual_seed(seed)
            image = upscale_image(image, scale=scale, upscaler_name=upscaler_name)
            
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
    
    # Create metadata infotext in the same format as Stable Diffusion web UI
    extra_params = {
        "LoRAs": ", ".join([f"{lora['name']}:{lora['scale']}" for lora in loras]) if loras else None,
        "Memory before": f"{memory_before:.1f} MB",
        "Memory after": f"{memory_after:.1f} MB",
        "Generation time": f"{elapsed_time:.2f}s",
    }
    
    # Add img2img specific parameters
    if job_type == "img2img":
        extra_params.update({
            "Denoising strength": f"{denoising_strength:.2f}",
            "Resize mode": resize_mode,
        })
        if resize_to:
            extra_params["Resize to"] = f"{resize_to.get('width', '?')}x{resize_to.get('height', '?')}"
        elif resize_by:
            extra_params["Resize by"] = f"{resize_by.get('width', '?')}x{resize_by.get('height', '?')}"
    
    geninfo = create_infotext(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        sampler_name=scheduler,
        scheduler=scheduler,
        cfg_scale=cfg_scale,
        seed=seed,
        width=image.width,
        height=image.height,
        model_name=model_name,
        model_hash="",  # You can add model hash if available
        vae_name="",    # You can add VAE name if available
        vae_hash="",    # You can add VAE hash if available
        denoising_strength=denoising_strength if job_type == "img2img" else None,
        clip_skip=data.get("clip_skip"),
        tiling=data.get("tiling", False),
        restore_faces=data.get("restore_faces", False),
        extra_generation_params=extra_params,
        user=data.get("user"),
        version="DiffusionAPI v1.0"
    )
    
    # Save image with metadata using the new function
    extension = f".{file_type}"
    save_image_with_metadata(
        image=image,
        filename=output_path,
        geninfo=geninfo,
        extension=extension,
        jpeg_quality=jpeg_quality
    )
    
    with open(output_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Keep status as processing until we have the image data
    progress_data = {
        "status": "processing",  # Keep as processing until we have the image
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
        "generation_time_sec": elapsed_time,
        "seed": seed  # Only include the actual seed value used
    }
    
    # Add img2img specific data
    if job_type == "img2img":
        progress_data.update({
            "denoising_strength": denoising_strength,
            "resize_mode": resize_mode,
        })
        if resize_to:
            progress_data["resize_to"] = resize_to
        if resize_by:
            progress_data["resize_by"] = resize_by
    
    update_progress_file(job_id, progress_data)

    logger.info("==> Job %s finished in %.2fs (mem %.1f → %.1f MB)", job_id, elapsed_time, memory_before, memory_after)


def get_lora_compatibility_info(lora_name: str, model_name: str) -> str:
    """Get information about LoRA compatibility with different model types."""
    lora_lower = lora_name.lower()
    model_lower = model_name.lower()
    
    # Common SDXL LoRAs
    sdxl_indicators = [
        "sdxl", "xl", "xlarge", "1024", "detail tweaker", "detail_tweaker",
        "realistic", "photorealistic", "high detail", "ultra detail"
    ]
    
    # Common SD 1.5 LoRAs
    sd15_indicators = [
        "sd15", "sd1.5", "stable diffusion", "anime", "cartoon", "artistic"
    ]
    
    # Check if LoRA name suggests SDXL
    is_likely_sdxl = any(indicator in lora_lower for indicator in sdxl_indicators)
    is_likely_sd15 = any(indicator in lora_lower for indicator in sd15_indicators)
    
    # Check model type
    is_sdxl_model = "sdxl" in model_lower or "xl" in model_lower
    is_sd_model = "sd" in model_lower and not is_sdxl_model
    
    if is_likely_sdxl and not is_sdxl_model:
        return f"LoRA '{lora_name}' appears to be designed for SDXL models, but you're using a standard SD model. Try using an SDXL base model."
    elif is_likely_sd15 and is_sdxl_model:
        return f"LoRA '{lora_name}' appears to be designed for SD 1.5 models, but you're using an SDXL model. Try using a standard SD base model."
    else:
        return f"LoRA '{lora_name}' may not be compatible with '{model_name}'. Check if the LoRA was trained for the same model architecture."


def load_loras(pipeline, loras: List[Dict[str, Any]], model_name: str = "") -> None:
    """Load LoRAs into the pipeline."""
    if not loras:
        return

    logger.info(f"==> Loading {len(loras)} LoRAs")
    successful_loras = []
    failed_loras = []
    
    for lora in loras:
        try:
            lora_name = lora["name"]
            lora_scale = lora["scale"]
            
            # Handle different LoRA file formats
            lora_path = Path(lora["path"])
            if not lora_path.suffix:
                # Try common LoRA extensions
                possible_extensions = [".safetensors", ".bin", ".pt", ".pth"]
                lora_path_found = None
                
                for ext in possible_extensions:
                    test_path = lora_path.with_suffix(ext)
                    if test_path.exists():
                        lora_path = test_path
                        lora_path_found = True
                        break
                
                if not lora_path_found:
                    # Default to safetensors
                    lora_path = lora_path.with_suffix(".safetensors")
            
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
            
            logger.info(f"   ↳ Loading LoRA '{lora_name}' with scale {lora_scale} from {lora_path}")
            
            # Try different weight file names
            weight_names = [
                "pytorch_lora_weights.safetensors",
                "pytorch_lora_weights.bin",
                "pytorch_lora_weights.pt",
                "pytorch_lora_weights.pth",
                "lora.safetensors",
                "lora.bin",
                "lora.pt",
                "lora.pth"
            ]
            
            lora_loaded = False
            for weight_name in weight_names:
                try:
                    pipeline.load_lora_weights(
                        lora_path,
                        weight_name=weight_name,
                        adapter_name=lora_name
                    )
                    lora_loaded = True
                    logger.debug(f"      Loaded with weight file: {weight_name}")
                    break
                except Exception as weight_error:
                    if "not found" in str(weight_error).lower():
                        continue
                    else:
                        raise weight_error
            
            if not lora_loaded:
                # Try loading without specifying weight_name
                pipeline.load_lora_weights(
                    lora_path,
                    adapter_name=lora_name
                )
            
            pipeline.fuse_lora(adapter_name=lora_name, scale=lora_scale)
            successful_loras.append(lora_name)
            logger.info(f"   ✓ Successfully loaded LoRA '{lora_name}'")
            
        except Exception as e:
            lora_name = lora.get("name", "unknown")
            error_msg = str(e)
            
            # Check if it's a compatibility error
            if any(keyword in error_msg.lower() for keyword in ["size mismatch", "incompatible", "shape", "dimension"]):
                logger.warning(f"   ⚠ LoRA '{lora_name}' is incompatible with current model architecture")
                logger.warning(f"      Error: {error_msg[:200]}...")
                
                # Provide specific compatibility guidance
                if model_name:
                    compatibility_info = get_lora_compatibility_info(lora_name, model_name)
                    logger.warning(f"      {compatibility_info}")
                else:
                    logger.warning(f"      This LoRA was likely trained for a different model type (SD vs SDXL)")
                    
            elif "not found" in error_msg.lower():
                logger.error(f"   ✗ LoRA file not found: {lora_path}")
            else:
                logger.error(f"   ✗ Failed to load LoRA '{lora_name}': {error_msg}")
            
            failed_loras.append(lora_name)
    
    # Summary
    if successful_loras:
        logger.info(f"==> Successfully loaded {len(successful_loras)} LoRAs: {', '.join(successful_loras)}")
    
    if failed_loras:
        logger.warning(f"==> Failed to load {len(failed_loras)} LoRAs: {', '.join(failed_loras)}")
        logger.warning(f"==> Generation will continue with only the compatible LoRAs")
    
    if not successful_loras and failed_loras:
        logger.warning(f"==> No LoRAs were successfully loaded. Generation will proceed without LoRAs.")


if __name__ == "__main__":
    main()
