from fastapi import FastAPI, Request
import uuid
import os
import json
import subprocess
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
import logging
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import time
import psutil

from diffusionapi.upscalers import INTERNAL_UPSCALERS, UPSCALER_DIR
from diffusionapi.generate import get_model_labels, get_model_name_from_label
from diffusionapi.metadata import read_metadata_from_image, parse_infotext

# Configura logger
log_path = Path("server.log")
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
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
QUEUE_DIR = Path("queue")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("stable_diffusion/models")
LORAS_DIR = Path("stable_diffusion/loras")

# Ensure directories exist
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LORAS_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_payload_for_logging(payload):
    """Remove sensitive data like base64 images from payload for logging."""
    sanitized = payload.copy()
    
    # Remove base64 image data
    if "image" in sanitized:
        image_data = sanitized["image"]
        if isinstance(image_data, str) and (image_data.startswith("data:image/") or len(image_data) > 100):
            sanitized["image"] = f"[BASE64_IMAGE_DATA_{len(image_data)}_chars]"
    
    return sanitized

@app.get("/hello")
async def hello():
    return {"ok": True}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Get system information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # Get GPU information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2)
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = {
                "mps_available": True,
                "device_name": "Apple Silicon GPU"
            }
        else:
            gpu_info = {
                "cuda_available": False,
                "mps_available": False,
                "device_name": "CPU"
            }
        
        # Get model information
        models_count = 0
        if MODELS_DIR.exists():
            models_count = len([d for d in MODELS_DIR.iterdir() if d.is_dir()])
        
        loras_count = 0
        if LORAS_DIR.exists():
            loras_count = len(list(LORAS_DIR.glob("*.safetensors")))
        
        # Get queue status
        queue_files = list(QUEUE_DIR.glob("*.json"))
        active_jobs = len([f for f in queue_files if not f.name.endswith("_job.json")])
        
        # Check if outputs directory exists and has space
        outputs_available = OUTPUT_DIR.exists()
        outputs_space_gb = 0
        if outputs_available:
            outputs_space_gb = round(psutil.disk_usage(str(OUTPUT_DIR)).free / (1024**3), 2)
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "system": system_info,
            "gpu": gpu_info,
            "models": {
                "count": models_count,
                "directory": str(MODELS_DIR)
            },
            "loras": {
                "count": loras_count,
                "directory": str(LORAS_DIR)
            },
            "queue": {
                "active_jobs": active_jobs,
                "directory": str(QUEUE_DIR)
            },
            "outputs": {
                "available": outputs_available,
                "free_space_gb": outputs_space_gb,
                "directory": str(OUTPUT_DIR)
            }
        }
        
        # Check for potential issues
        warnings = []
        if system_info["memory_percent"] > 90:
            warnings.append("High memory usage")
        if system_info["disk_usage_percent"] > 90:
            warnings.append("High disk usage")
        if models_count == 0:
            warnings.append("No models found")
        if not outputs_available:
            warnings.append("Outputs directory not accessible")
        if outputs_space_gb < 1:
            warnings.append("Low disk space for outputs")
        
        if warnings:
            health_status["status"] = "warning"
            health_status["warnings"] = warnings
        
        return health_status
        
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/queue/{job_id}")
async def get_queue_status(job_id: str):
    """Get the status of a specific job."""
    try:
        # Read the progress file
        progress_file = QUEUE_DIR / f"{job_id}.json"
        if not progress_file.exists():
            return JSONResponse(
                status_code=404,
                content={"status": "error", "detail": f"Job {job_id} not found"}
            )
        
        with open(progress_file) as f:
            data = json.load(f)
        
        # Get the job data to check if hires is enabled and to return full payload
        job_file = QUEUE_DIR / f"{job_id}_job.json"
        hires_enabled = False
        job_payload = None
        
        # Always try to read the job file, even if we're not done
        if job_file.exists():
            try:
                with open(job_file) as f:
                    job_payload = json.load(f)
                    hires_enabled = job_payload.get("hires", {}).get("enabled", False)
            except Exception as e:
                logger.warning(f"Failed to read job file for {job_id}: {e}")
        
        # Calculate progress based on status and hires
        status = data.get("status", "unknown")
        progress = data.get("progress", 0.0)
        phase = data.get("phase", "generation")
        
        # For hires, we have two phases:
        # Phase 1 (0-50%): Initial generation
        # Phase 2 (50-100%): Upscaling
        if hires_enabled and status == "processing":
            # Use the phase from the progress file
            current_phase = phase
            # Progress is already scaled correctly in generate.py
            progress = progress * 100  # Convert to percentage
        else:
            # Normal generation or other status
            current_phase = phase if status == "processing" else status
            progress = progress * 100  # Convert to percentage
        
        # Ensure progress is between 0 and 100
        progress = max(0, min(100, progress))
        
        # Check if we have the image data
        image_data = data.get("image")
        has_image = image_data is not None and isinstance(image_data, str) and len(image_data) > 0
        
        # Only mark as done if we have the image
        is_done = has_image and (progress >= 100 or status == "done")
        
        response = {
            "status": "success",
            "job_id": job_id,
            "progress": round(progress, 2),
            "current_phase": current_phase,
            "state": "done" if is_done else status,
            "error": data.get("detail")
        }
        
        # Only include complete payload and image data if we have the image
        if has_image:
            # Always include the job payload if we have it
            if job_payload is not None:
                # Remove any sensitive or internal fields
                safe_payload = {k: v for k, v in job_payload.items() 
                              if k not in ['output', 'job_id', 'seed']}  # Remove internal fields including seed
                response["payload"] = safe_payload
            
            # Add image data and metadata from progress file
            response["image"] = image_data
            response["output_path"] = data.get("output_path")
            response["width"] = data.get("width")
            response["height"] = data.get("height")
            response["file_type"] = data.get("file_type")
            response["jpeg_quality"] = data.get("jpeg_quality")
            response["generation_time_sec"] = data.get("generation_time_sec")
            response["memory_before_mb"] = data.get("memory_before_mb")
            response["memory_after_mb"] = data.get("memory_after_mb")
            response["seed"] = data.get("seed")  # Get seed from progress file
        
        return response
        
    except Exception as e:
        logger.exception(f"Failed to get status for job {job_id}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/txt2img")
async def txt2img(request: Request):
    payload = await request.json()
    job_id = str(uuid.uuid4())

    # Print the received payload
    logger.info("Received txt2img request:")
    logger.info(f"Job ID: {job_id}")
    logger.info("Payload:")
    logger.info(json.dumps(sanitize_payload_for_logging(payload), indent=2))    
    
    # Get file type from payload, default to jpg
    file_type = payload.get("fileType", "jpg").lower()
    if file_type not in ["png", "jpg"]:
        file_type = "jpg"
    
    # Set output path with correct extension
    output_path = str(Path("outputs") / f"{job_id}.{file_type}")

    # Handle loras
    loras = payload.get("loras", [])
    if not isinstance(loras, list):
        loras = []

    normalized_loras = []
    for entry in loras:
        if isinstance(entry, dict) and "name" in entry:
            # Get the lora name and verify it exists
            lora_name = entry["name"]
            lora_path = LORAS_DIR / f"{lora_name}.safetensors"
            
            if not lora_path.exists():
                logger.warning(f"LoRA not found: {lora_name}")
                continue
                
            normalized_loras.append({
                "name": lora_name,
                "path": str(lora_path),
                "scale": float(entry.get("scale", 1.0))
            })
        elif isinstance(entry, str):
            # Handle simple string lora names
            lora_path = LORAS_DIR / f"{entry}.safetensors"
            
            if not lora_path.exists():
                logger.warning(f"LoRA not found: {entry}")
                continue
                
            normalized_loras.append({
                "name": entry,
                "path": str(lora_path),
                "scale": 1.0
            })

    payload["loras"] = normalized_loras

    job = {
        "job_id": job_id,
        "prompt": payload.get("prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "steps": payload.get("steps", 30),
        "cfg_scale": payload.get("cfg_scale", 7.0),
        "width": payload.get("width", 512),
        "height": payload.get("height", 512),
        "model": payload.get("model", "stable-diffusion-v1-5"),
        "refiner_checkpoint": payload.get("refiner_checkpoint"),
        "refiner_switch_at": payload.get("refiner_switch_at", 0.8),
        "output": output_path,
        "file_type": file_type,
        "jpeg_quality": payload.get("jpeg_quality", 85),
        "loras": normalized_loras,
        "sampler_name": payload.get("sampler_name", "DPM++ 2M Karras"),
        "scheduler_type": payload.get("scheduler_type", "karras"),
        "seed": payload.get("seed")  # This will be modified by generate.py if needed
    }
    
    hires = payload.get("hires")
    
    logger.info("Before hires")
    
    if isinstance(hires, dict) and hires.get("enabled"):
        job["hires"] = {
            "enabled": True,
            "scale": float(hires.get("scale", 2.0)),
            "upscaler": hires.get("upscaler", "Latent"),
            "steps": int(hires.get("steps", 20)),
            "denoising_strength": float(hires.get("denoising_strength", 0.4))
        }

    logger.info("After hires")

    job = {k: v for k, v in job.items() if v is not None}

    # Save both the progress file and the job file
    with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
        json.dump({"status": "queued", "progress": 0.0, "job_id": job_id}, f)
    
    # Save the full job payload (sanitized for file storage)
    with open(QUEUE_DIR / f"{job_id}_job.json", "w") as f:
        json.dump(sanitize_payload_for_logging(job), f)

    try:
        logger.info("Before subprocess")
        proc = subprocess.Popen(
            ["python3", "-m", "diffusionapi.generate"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("After subprocess")
        proc.stdin.write(json.dumps(job))  # Send original job with base64 data to subprocess
        proc.stdin.close()
        logger.info("After stdin")

        # Start a thread to read and log the output
        def log_output(pipe, prefix):
            for line in pipe:
                logger.info(f"{prefix}: {line.strip()}")

        stdout_thread = threading.Thread(target=log_output, args=(proc.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=log_output, args=(proc.stderr, "STDERR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        logger.info(f"Job {job_id} dispatched to subprocess.")
    except Exception as e:
        logger.exception(f"Failed to start subprocess for job {job_id}")
        with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
            json.dump({
                "status": "error",
                "progress": 0.0,
                "job_id": job_id,
                "detail": str(e)
            }, f)
        return {"status": "error", "detail": str(e)}

    return {
        "job_id": job_id,
        "status": "queued",
        "seed": payload.get("seed")  # This will be updated in status endpoint
    }

@app.post("/img2img")
async def img2img(request: Request):
    payload = await request.json()
    job_id = str(uuid.uuid4())

    # Print the received payload
    logger.info("Received img2img request:")
    logger.info(f"Job ID: {job_id}")
    logger.info("Payload:")
    logger.info(json.dumps(sanitize_payload_for_logging(payload), indent=2))    
    
    # Validate required fields
    if not payload.get("image"):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Image field is required"}
        )
    
    if not payload.get("prompt"):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Prompt field is required"}
        )
    
    # Get file type from payload, default to jpg
    file_type = payload.get("fileType", "jpg").lower()
    if file_type not in ["png", "jpg"]:
        file_type = "jpg"
    
    # Set output path with correct extension
    output_path = str(Path("outputs") / f"{job_id}.{file_type}")

    # Handle loras - IGNORE LoRAs for img2img to avoid compatibility issues
    logger.info("==> Ignoring LoRAs for img2img to prevent compatibility issues")
    normalized_loras = []  # Always empty for img2img

    payload["loras"] = normalized_loras

    # Handle resize parameters
    resize_mode = payload.get("resize_mode", "just resize")
    resize_to = payload.get("resize_to")
    resize_by = payload.get("resize_by") or payload.get("resized_by")  # Accept both formats
    
    # Convert resize_mode to numeric format expected by webui
    resize_mode_map = {
        "just resize": 0,
        "crop and resize": 1,
        "resize and fill": 2
    }
    resize_mode_num = resize_mode_map.get(resize_mode, 0)
    
    # Handle resize_by as simple number (scale factor) or object
    if resize_by is not None:
        if isinstance(resize_by, (int, float)):
            # Convert simple number to scale object
            scale_factor = float(resize_by)
            resize_by = {"width": scale_factor, "height": scale_factor}
            logger.info(f"Converted simple resize_by value {resize_by} to scale object: {resize_by}")
    
    logger.info(f"Resize parameters: mode='{resize_mode}' -> {resize_mode_num}, resize_to={resize_to}, resize_by={resize_by}")

    job = {
        "job_id": job_id,
        "type": "img2img",  # Mark this as img2img job
        "prompt": payload.get("prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "image": payload.get("image"),  # Base64 encoded input image
        "steps": payload.get("steps", 30),
        "cfg_scale": payload.get("cfg_scale", 7.0),
        "width": payload.get("width", 512),
        "height": payload.get("height", 512),
        "model": payload.get("model", "stable-diffusion-v1-5"),
        "refiner_checkpoint": payload.get("refiner_checkpoint"),
        "refiner_switch_at": payload.get("refiner_switch_at", 0.8),
        "output": output_path,
        "file_type": file_type,
        "jpeg_quality": payload.get("jpeg_quality", 85),
        "loras": normalized_loras,  # Empty list - LoRAs ignored for img2img
        "sampler_name": payload.get("sampler_name", "DPM++ 2M Karras"),
        "scheduler_type": payload.get("scheduler_type", "karras"),
        "seed": payload.get("seed"),  # This will be modified by generate.py if needed
        # Img2Img specific parameters
        "denoising_strength": payload.get("denoising_strength", 0.75),
        "image_guidance_scale": payload.get("image_guidance_scale", 1.8),
        "resize_mode": resize_mode_num,
        "resize_to": resize_to,
        "resize_by": resize_by
    }
    
    hires = payload.get("hires")
    
    logger.info("Before hires")
    
    if isinstance(hires, dict) and hires.get("enabled"):
        job["hires"] = {
            "enabled": True,
            "scale": float(hires.get("scale", 2.0)),
            "upscaler": hires.get("upscaler", "Latent"),
            "steps": int(hires.get("steps", 20)),
            "denoising_strength": float(hires.get("denoising_strength", 0.4))
        }

    logger.info("After hires")

    job = {k: v for k, v in job.items() if v is not None}

    # Save both the progress file and the job file
    with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
        json.dump({"status": "queued", "progress": 0.0, "job_id": job_id}, f)
    
    # Save the full job payload (sanitized for file storage)
    with open(QUEUE_DIR / f"{job_id}_job.json", "w") as f:
        json.dump(sanitize_payload_for_logging(job), f)

    try:
        logger.info("Before subprocess")
        proc = subprocess.Popen(
            ["python3", "-m", "diffusionapi.generate"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("After subprocess")
        proc.stdin.write(json.dumps(job))  # Send original job with base64 data to subprocess
        proc.stdin.close()
        logger.info("After stdin")

        # Start a thread to read and log the output
        def log_output(pipe, prefix):
            for line in pipe:
                logger.info(f"{prefix}: {line.strip()}")

        stdout_thread = threading.Thread(target=log_output, args=(proc.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=log_output, args=(proc.stderr, "STDERR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        logger.info(f"Job {job_id} dispatched to subprocess.")
    except Exception as e:
        logger.exception(f"Failed to start subprocess for job {job_id}")
        with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
            json.dump({
                "status": "error",
                "progress": 0.0,
                "job_id": job_id,
                "detail": str(e)
            }, f)
        return {"status": "error", "detail": str(e)}

    return {
        "job_id": job_id,
        "status": "queued",
        "seed": payload.get("seed")  # This will be updated in status endpoint
    }

@app.get("/models")
async def list_models():
    """List all available models with their user-friendly labels."""
    try:
        # Set to store unique model names
        model_names = set()
        
        # Look for model directories
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                # Check if this is a valid model directory by looking for unet component
                # Look for both .safetensors and .bin files
                has_unet = (
                    any(model_dir.glob("**/unet/diffusion_pytorch_model.safetensors")) or
                    any(model_dir.glob("**/unet/diffusion_pytorch_model.bin"))
                )
                if has_unet:
                    # Use the original directory name for proper label matching
                    model_name = model_dir.name
                    model_names.add(model_name)
        
        # Get labels
        labels = get_model_labels()
        
        # Convert to sorted list with labels
        models = []
        for name in sorted(model_names):
            # Get the label for this model
            label = labels.get(name, name)  # Use the name as label if no label defined
            
            model_info = {
                "name": name,
                "label": label
            }
            models.append(model_info)
        
        if not models:
            logger.warning(f"No models found in {MODELS_DIR}")
        
        return {
            "status": "success",
            "models": models
        }
    except Exception as e:
        logger.exception(f"Failed to list models from {MODELS_DIR}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/loras")
async def list_loras():
    """List all available LoRAs."""
    try:
        # Set to store unique lora names
        lora_names = set()
        
        # Look for .safetensors files in the loras directory
        for lora_path in LORAS_DIR.glob("*.safetensors"):
            # Get the lora name without extension
            lora_name = lora_path.stem
            lora_names.add(lora_name)
        
        # Convert to sorted list
        loras = [{"name": name} for name in sorted(lora_names)]
        
        if not loras:
            logger.warning(f"No loras found in {LORAS_DIR}")
        
        return {
            "status": "success",
            "loras": loras
        }
    except Exception as e:
        logger.exception(f"Failed to list loras from {LORAS_DIR}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/upscalers")
async def list_upscalers():
    """List all available upscalers (both internal and external)."""
    try:
        upscalers = []
        
        # Add internal upscalers
        for upscaler in sorted(INTERNAL_UPSCALERS):
            upscalers.append({
                "name": upscaler,
                "type": "internal",
                "description": f"Built-in {upscaler} upscaler"
            })
        
        # Add external upscalers (ESRGAN models)
        if UPSCALER_DIR.exists():
            for model_path in UPSCALER_DIR.glob("*.pth"):
                model_name = model_path.stem
                # Try to determine scale from filename
                scale = 4  # default
                if "2x" in model_name.lower():
                    scale = 2
                elif "4x" in model_name.lower():
                    scale = 4
                
                upscalers.append({
                    "name": model_name,
                    "type": "external",
                    "scale": scale,
                    "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                    "description": f"ESRGAN upscaler model ({scale}x)"
                })
        else:
            logger.warning(f"Upscaler directory not found: {UPSCALER_DIR}")
        
        if not upscalers:
            logger.warning("No upscalers found (neither internal nor external)")
        
        return {
            "status": "success",
            "upscalers": sorted(upscalers, key=lambda x: x["name"])
        }
    except Exception as e:
        logger.exception(f"Failed to list upscalers from {UPSCALER_DIR}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/sampling_methods")
async def list_sampling_methods():
    """List all available sampling methods and scheduler types."""
    try:
        sampling_methods = [
            "Euler",
            "Euler a",
            "Heun",
            "DPM2",
            "DPM++ 2S a",
            "DPM++ 2M",
            "DPM++ 2M Karras",
            "DPM++ SDE",
            "DPM++ SDE Karras",
            "DPM fast",
            "DPM adaptive",
            "LMS",
            "LMS Karras",
            "DDIM",
            "PLMS",
            "UniPC"
        ]
        
        scheduler_types = [
            "Automatic",
            "Uniform",
            "Karras",
            "Exponential",
            "Polyexponential",
            "SGM Uniform",
            "KL Optimal",
            "Align Your Steps",
            "Simple",
            "Normal",
            "DDIM",
            "Beta"
        ]
        
        return {
            "status": "success",
            "sampling_methods": sampling_methods,
            "scheduler_types": scheduler_types
        }
    except Exception as e:
        logger.exception("Failed to list sampling methods")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/models/aliases")
async def list_model_aliases():
    """List all available models with their user-friendly labels."""
    try:
        labels = get_model_labels()
        
        # Convert to simple list format
        models = []
        for model_name, label in labels.items():
            models.append({
                "name": model_name,
                "label": label
            })
        
        return {
            "status": "success",
            "models": sorted(models, key=lambda x: x["label"])
        }
        
    except Exception as e:
        logger.exception("Failed to list model labels")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/img-info")
async def img_info(request: Request):
    """
    Extract metadata from a base64 encoded image.
    
    Expected request format:
    {
        "image": "base64_encoded_image_data"
    }
    
    Returns:
    {
        "info": "raw_metadata_string",
        "items": {"other_metadata": "values"},
        "parameters": {"parsed_parameters": "values"}
    }
    """
    try:
        # Parse the request
        payload = await request.json()
        image_base64 = payload.get("image")
        
        if not image_base64:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": "No image data provided"
                }
            )
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image/'):
                # Extract the base64 part after the comma
                image_base64 = image_base64.split(',', 1)[1]
            
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": f"Invalid base64 image data: {str(e)}"
                }
            )
        
        # Load image with PIL
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": f"Failed to load image: {str(e)}"
                }
            )
        
        # Extract metadata
        geninfo, other_metadata = read_metadata_from_image(image)
        
        # Parse the metadata if available
        parameters = {}
        if geninfo:
            try:
                parameters = parse_infotext(geninfo)
            except Exception as e:
                logger.warning(f"Failed to parse infotext: {e}")
                # If parsing fails, still return the raw info
                parameters = {"raw_info": geninfo}
        
        # Prepare response
        response = {
            "status": "success",
            "info": geninfo or "",
            "items": other_metadata or {},
            "parameters": parameters
        }
        
        logger.info(f"Successfully extracted metadata from image")
        return response
        
    except Exception as e:
        logger.exception(f"Failed to extract image metadata")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Internal server error: {str(e)}"
            }
        )