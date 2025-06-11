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

    # Prepara parâmetros para o subprocesso
    job = {
        "prompt": payload["prompt"],
        "steps": payload.get("steps", 10),  # valor mais baixo para testes rápidos
        "cfg_scale": payload.get("cfg_scale", 7.5),
        "model": model_name,  # Use the model name directly
        "output": output_path,
        "width": payload.get("width", 512),
        "height": payload.get("height", 512)
    }

    print("Sending to generate.py:", json.dumps(job, indent=2))  # Debug print

    # Roda o script de geração como subprocesso
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
                "error": "Erro ao gerar imagem",
                "stderr": stderr.decode(),
                "stdout": stdout.decode()
            },
            status_code=500
        )

    # Codifica a imagem gerada para base64
    with open(output_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "images": [image_base64],
        "parameters": payload,
        "info": json.dumps({"seed": payload.get("seed", -1)})
    }