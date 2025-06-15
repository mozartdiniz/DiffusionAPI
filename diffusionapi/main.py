from fastapi import FastAPI, Request
import uuid
import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import logging
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from diffusionapi.upscalers import INTERNAL_UPSCALERS, UPSCALER_DIR

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

@app.get("/hello")
async def hello():
    return {"ok": True}

@app.get("/sdapi/v1/queue/{job_id}")
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
        
        response = {
            "status": "success",
            "job_id": job_id,
            "progress": round(progress, 2),
            "current_phase": current_phase,
            "state": status,
            "error": data.get("detail")
        }
        
        # If generation is complete (progress = 100% or status = done)
        if progress >= 100 or status == "done":
            # Always include the job payload if we have it
            if job_payload is not None:
                # Remove any sensitive or internal fields
                safe_payload = {k: v for k, v in job_payload.items() 
                              if k not in ['output', 'job_id']}  # Remove internal fields
                response["payload"] = safe_payload
            
            # Add image data if available
            if "image" in data:
                response["image"] = data["image"]
                response["output_path"] = data.get("output_path")
                response["width"] = data.get("width")
                response["height"] = data.get("height")
                response["file_type"] = data.get("file_type")
                response["jpeg_quality"] = data.get("jpeg_quality")
                response["generation_time_sec"] = data.get("generation_time_sec")
                response["memory_before_mb"] = data.get("memory_before_mb")
                response["memory_after_mb"] = data.get("memory_after_mb")
            
            # Delete both JSON files after returning the response
            try:
                progress_file.unlink()
                if job_file.exists():
                    job_file.unlink()
                logger.info(f"Cleaned up JSON files for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to delete JSON files for job {job_id}: {e}")
        
        return response
        
    except Exception as e:
        logger.exception(f"Failed to get status for job {job_id}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    payload = await request.json()
    job_id = str(uuid.uuid4())
    
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
        "scheduler_type": payload.get("scheduler_type", "karras")
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
    
    # Save the full job payload
    with open(QUEUE_DIR / f"{job_id}_job.json", "w") as f:
        json.dump(job, f)

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
        proc.stdin.write(json.dumps(job))
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
        "status": "queued"
    }

@app.get("/sdapi/v1/models")
async def list_models():
    """List all available models."""
    try:
        # Set to store unique model names
        model_names = set()
        
        # Look for model directories
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                # Check if this is a valid model directory by looking for unet component
                if any(model_dir.glob("**/unet/diffusion_pytorch_model.safetensors")):
                    # Get the model name and normalize it
                    model_name = model_dir.name.replace("__", "/")  # Convert double underscore to slash
                    model_names.add(model_name)
        
        # Convert to sorted list
        models = [{"name": name} for name in sorted(model_names)]
        
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

@app.get("/sdapi/v1/loras")
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

@app.get("/sdapi/v1/upscalers")
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

@app.get("/sdapi/v1/sampling_methods")
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
            "default",
            "karras",
            "exponential",
            "ddim",
            "pndm"
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