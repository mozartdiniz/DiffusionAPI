from diffusers import StableDiffusionPipeline
import torch

# Define o dispositivo
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# Carrega o pipeline com o dtype correto
pipe = StableDiffusionPipeline.from_pretrained(
    "stable_diffusion/models/stable-diffusion-v1-5",
    torch_dtype=dtype,
    use_safetensors=True
)

# Move o modelo para o dispositivo correto
pipe = pipe.to(device)

# Gera uma imagem de teste
image = pipe("a fantasy landscape with mountains").images[0]
image.save("output.png")