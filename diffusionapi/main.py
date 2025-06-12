from fastapi import FastAPI, Request
import uuid
import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import logging

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
QUEUE_DIR = Path("queue")
QUEUE_DIR.mkdir(exist_ok=True)

@app.get("/hello")
async def hello():
    return {"ok": True}

@app.get("/queue/{job_id}")
def get_job_status(job_id: str):
    path = f"queue/{job_id}.json"

    if not os.path.exists(path):
        return {
            "status": "empty",
            "progress": 0.0,
            "job_id": job_id
        }

    with open(path, "r") as f:
        data = json.load(f)

    if data.get("status") == "done" and "image" in data:
        full_data = dict(data)
        del data["image"]
        with open(path, "w") as f:
            json.dump(data, f)
        return full_data
    else:
        return data

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    payload = await request.json()
    job_id = str(uuid.uuid4())
    output_path = str(Path("outputs") / f"{job_id}.png")

    loras = payload.get("loras", [])
    if not isinstance(loras, list):
        loras = []

    normalized_loras = []
    for entry in loras:
        if isinstance(entry, dict) and "name" in entry:
            normalized_loras.append({
                "name": entry["name"],
                "scale": float(entry.get("scale", 1.0))
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
        "loras": normalized_loras
    }

    job = {k: v for k, v in job.items() if v is not None}

    with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
        json.dump({"status": "queued", "progress": 0.0, "job_id": job_id}, f)

    try:
        proc = subprocess.Popen(
            ["python3", "diffusionapi/generate.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        proc.stdin.write(json.dumps(job))
        proc.stdin.close()
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