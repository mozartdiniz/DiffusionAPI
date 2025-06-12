from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
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

    # Se o job ainda não existir, retornamos status "empty"
    if not os.path.exists(path):
        return {
            "status": "empty",
            "progress": 0.0,
            "job_id": job_id
        }

    with open(path, "r") as f:
        data = json.load(f)

    # Se estiver finalizado e tiver uma imagem, retornamos uma cópia e limpamos
    if data.get("status") == "done" and "image" in data:
        full_data = dict(data)  # copia
        del data["image"]

        # Salva o json limpo no disco
        with open(path, "w") as f:
            json.dump(data, f)

        return full_data  # retorna a versão com image
    else:
        return data


@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    payload = await request.json()
    job_id = str(uuid.uuid4())
    output_path = str(Path("outputs") / f"{job_id}.png")

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
        "output": output_path
    }

    # Remove None values
    job = {k: v for k, v in job.items() if v is not None}

    # Inicializa o arquivo de status como "queued"
    with open(QUEUE_DIR / f"{job_id}.json", "w") as f:
        json.dump({"status": "queued", "progress": 0.0, "job_id": job_id}, f)

    # Envia para o generate.py em background
    process = subprocess.Popen(
        ["python3", "diffusionapi/generate.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    process.stdin.write(json.dumps(job).encode())
    process.stdin.close()

    return {
        "job_id": job_id,
        "status": "queued"
    }