import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer

# ✅ Set your paths here
CKPT_PATH = "/workspace/Plant_Milk_Almond.safetensors"
OUTPUT_DIR = "/workspace/stable_diffusion/models/plant_milk_almond"

# Optional: pick dtype
torch_dtype = torch.float16

# Load weights
print("Loading checkpoint...")
state_dict = load_file(CKPT_PATH)

# Load components
print("Loading components...")
pipe = StableDiffusionXLPipeline.from_single_file(
    CKPT_PATH,
    torch_dtype=torch_dtype,
    use_safetensors=True,
)

# Save
print(f"Saving to: {OUTPUT_DIR}")
pipe.save_pretrained(OUTPUT_DIR, safe_serialization=True)
print("✅ Conversion complete.")