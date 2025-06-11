from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import os
import json
import subprocess
import base64

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"ok": True}

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    payload = await request.json()

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")

    # Prepara parâmetros para o subprocesso
    job = {
        "prompt": payload["prompt"],
        "steps": payload.get("steps", 30),
        "cfg_scale": payload.get("cfg_scale", 7.5),
        "model": payload.get("model", "runwayml/stable-diffusion-v1-5"),
        "output": output_path
    }

    # Roda o script de geração
    process = subprocess.Popen(
        ["python3", "diffusionapi/generate.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=json.dumps(job).encode())

    if process.returncode != 0:
        return JSONResponse(
            content={"error": stderr.decode()},
            status_code=500
        )

    with open(output_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "images": [image_base64],
        "parameters": payload,
        "info": json.dumps({"seed": payload.get("seed", -1)})
    }