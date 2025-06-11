from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, model_info
import os
import argparse
import shutil
import json
import time
import errno

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)


def is_sdxl_model(model_id: str):
    try:
        info = model_info(model_id)
        for file in info.siblings:
            if file.rfilename == "model_index.json":
                from huggingface_hub import hf_hub_download
                json_path = hf_hub_download(repo_id=model_id, filename="model_index.json")
                with open(json_path, "r") as f:
                    model_config = json.load(f)
                    if "StableDiffusionXLPipeline" in model_config.get("_class_name", ""):
                        return True
        return False
    except Exception as e:
        print(f"Warning: couldn't determine model type: {e}")
        return "xl" in model_id.lower()


def safe_copytree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)

            if os.path.exists(dst_file):
                continue

            for attempt in range(3):
                try:
                    shutil.copy2(src_file, dst_file)
                    break
                except OSError as e:
                    if e.errno == errno.ETIMEDOUT:
                        print(f"⚠️ Timeout copying {file}, retrying... ({attempt+1}/3)")
                        time.sleep(5)
                    else:
                        raise


def download_and_move(model_name: str, dest_dir: Path):
    print(f"\n📦 Downloading model: {model_name}")
    snapshot_path = snapshot_download(
        repo_id=model_name,
        local_dir_use_symlinks=False,
        cache_dir=os.getenv("HF_HOME", None)  # garante que ele vá pro cache local
    )
    print(f"✅ Download complete in: {snapshot_path}")

    dest_path = dest_dir / f"models--{model_name.replace('/', '--')}"
    print(f"📂 Copying contents to: {dest_path}")
    safe_copytree(snapshot_path, dest_path)
    print("✅ Model moved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a HuggingFace Stable Diffusion model')
    parser.add_argument('--model', type=str, required=True, help='Model repo ID (e.g. runwayml/stable-diffusion-v1-5)')
    args = parser.parse_args()

    dest_dir = os.getenv("DEST_DIR")
    if not dest_dir:
        raise RuntimeError("Missing DEST_DIR environment variable in .env")

    download_and_move(args.model, Path(dest_dir))