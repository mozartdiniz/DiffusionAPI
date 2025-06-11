from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import os
import json
import subprocess
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"ok": True}

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    payload = await request.json()
    print("Received API request:", json.dumps(payload, indent=2))  # Debug print

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")

    # Ensure model name is passed correctly
    model_name = payload.get("model", "stable-diffusion-v1-5")
    print(f"Using model: {model_name}")  # Debug print

    # Prepare parameters with all supported options from the API definition
    job = {
        "prompt": payload.get("prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "styles": payload.get("styles", []),
        "seed": payload.get("seed", -1),
        "subseed": payload.get("subseed", -1),
        "subseed_strength": payload.get("subseed_strength", 0),
        "seed_resize_from_h": payload.get("seed_resize_from_h", -1),
        "seed_resize_from_w": payload.get("seed_resize_from_w", -1),
        "sampler_name": payload.get("sampler_name"),
        "scheduler": payload.get("scheduler"),
        "batch_size": payload.get("batch_size", 1),
        "n_iter": payload.get("n_iter", 1),
        "steps": payload.get("steps", 50),
        "cfg_scale": payload.get("cfg_scale", 7.0),
        "width": payload.get("width", 512),
        "height": payload.get("height", 512),
        "restore_faces": payload.get("restore_faces"),
        "tiling": payload.get("tiling"),
        "do_not_save_samples": payload.get("do_not_save_samples", False),
        "do_not_save_grid": payload.get("do_not_save_grid", False),
        "eta": payload.get("eta"),
        "denoising_strength": payload.get("denoising_strength"),
        "s_min_uncond": payload.get("s_min_uncond"),
        "s_churn": payload.get("s_churn"),
        "s_tmax": payload.get("s_tmax"),
        "s_tmin": payload.get("s_tmin"),
        "s_noise": payload.get("s_noise"),
        "override_settings": payload.get("override_settings"),
        "override_settings_restore_afterwards": payload.get("override_settings_restore_afterwards", True),
        "refiner_checkpoint": payload.get("refiner_checkpoint"),
        "refiner_switch_at": payload.get("refiner_switch_at"),
        "disable_extra_networks": payload.get("disable_extra_networks", False),
        "firstpass_image": payload.get("firstpass_image"),
        "comments": payload.get("comments"),
        "enable_hr": payload.get("enable_hr", False),
        "firstphase_width": payload.get("firstphase_width", 0),
        "firstphase_height": payload.get("firstphase_height", 0),
        "hr_scale": payload.get("hr_scale", 2.0),
        "hr_upscaler": payload.get("hr_upscaler"),
        "hr_second_pass_steps": payload.get("hr_second_pass_steps", 0),
        "hr_resize_x": payload.get("hr_resize_x", 0),
        "hr_resize_y": payload.get("hr_resize_y", 0),
        "model": model_name,
        "output": output_path
    }

    # Remove None values to keep the payload clean
    job = {k: v for k, v in job.items() if v is not None}

    print("Sending to generate.py:", json.dumps(job, indent=2))  # Debug print

    # Run the generation script as subprocess
    process = subprocess.Popen(
        ["python3", "diffusionapi/generate.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=json.dumps(job).encode())

    # Print all output for debugging
    if stdout:
        print("STDOUT:", stdout.decode())
    if stderr:
        print("STDERR:", stderr.decode())

    if process.returncode != 0:
        return JSONResponse(
            content={
                "error": "Error generating image",
                "stderr": stderr.decode(),
                "stdout": stdout.decode()
            },
            status_code=500
        )

    # Encode the generated image to base64
    with open(output_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Return response matching the API definition
    return {
        "images": [image_base64],
        "parameters": payload,
        "info": json.dumps({
            "seed": payload.get("seed", -1),
            "subseed": payload.get("subseed", -1),
            "subseed_strength": payload.get("subseed_strength", 0),
            "width": payload.get("width", 512),
            "height": payload.get("height", 512),
            "sampler_name": payload.get("sampler_name"),
            "cfg_scale": payload.get("cfg_scale", 7.0),
            "steps": payload.get("steps", 50),
            "batch_size": payload.get("batch_size", 1),
            "n_iter": payload.get("n_iter", 1),
            "prompt": payload.get("prompt", ""),
            "negative_prompt": payload.get("negative_prompt", ""),
            "styles": payload.get("styles", []),
            "model": model_name
        })
    }