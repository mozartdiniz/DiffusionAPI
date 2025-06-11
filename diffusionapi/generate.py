import sys
import json
from diffusers import StableDiffusionPipeline
import torch

def main():
    data = json.load(sys.stdin)

    prompt = data["prompt"]
    steps = data.get("steps", 30)
    cfg_scale = data.get("cfg_scale", 7.5)
    model_id = data.get("model", "runwayml/stable-diffusion-v1-5")
    output_path = data["output"]
    width = data.get("width", 512)
    height = data.get("height", 512)

    # Carrega o pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32  # ou float16 se usar GPU
    )
    pipe.to("cpu")  # ou "cuda"

    # Geração
    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        height=height,
        width=width
    ).images[0]
    
    image.save(output_path)

if __name__ == "__main__":
    main()