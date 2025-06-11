from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  # online test model
    torch_dtype="auto"
)
print("Pipeline loaded successfully")
