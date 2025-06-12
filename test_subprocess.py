import subprocess
import json

payload = {
    "prompt": "a test image",
    "steps": 10,
    "cfg_scale": 7,
    "model": "John6666__amanatsu-illustrious-v11-sdxl",  # nome da pasta local do modelo
    "output": "outputs/test.png",
    "width": 512,
    "height": 512,
    "job_id": "testjob"
}

process = subprocess.Popen(
    ["python3", "diffusionapi/generate.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(input=json.dumps(payload))

print("✅ Process finished.")
if stdout:
    print("📤 STDOUT:\n", stdout)
if stderr:
    print("⚠️ STDERR:\n", stderr)

if process.returncode != 0:
    print(f"❌ Generation failed with exit code {process.returncode}")
else:
    print("🎉 Image generation succeeded.")