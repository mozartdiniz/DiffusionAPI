import os
from PIL import Image
from pathlib import Path
import logging
import torch
from diffusers.utils.torch_utils import is_compiled_module

logger = logging.getLogger(__name__)

INTERNAL_UPSCALERS = {
    "Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)",
    "Latent (nearest)", "Latent (nearest-exact)",
    "Lanczos", "Nearest", "None"
}

UPSCALER_DIR = os.getenv("UPSCALERS_DIR", "stable_diffusion/upscalers")

def is_internal_upscaler(name: str) -> bool:
    return name in INTERNAL_UPSCALERS

def apply_internal_upscale(image: Image.Image, upscaler: str, scale: float = 2.0) -> Image.Image:
    logger.info(f"Applying internal upscaler: {upscaler} with scale {scale}")
    w, h = image.size
    new_size = (int(w * scale), int(h * scale))

    if upscaler == "None":
        return image
    elif upscaler == "Lanczos":
        return image.resize(new_size, Image.LANCZOS)
    elif upscaler == "Nearest":
        return image.resize(new_size, Image.NEAREST)
    elif "bicubic" in upscaler.lower():
        return image.resize(new_size, Image.BICUBIC)
    elif "nearest" in upscaler.lower():
        return image.resize(new_size, Image.NEAREST)
    elif "antialiased" in upscaler.lower():
        return image.resize(new_size, Image.LANCZOS)
    else:
        logger.warning(f"Unknown internal upscaler fallback: {upscaler}, using BICUBIC")
        return image.resize(new_size, Image.BICUBIC)

# Placeholder for external upscalers (like Real-ESRGAN, ESRGAN, etc)
def apply_external_upscale(image: Image.Image, model_name: str, scale: float = 2.0) -> Image.Image:
    model_path = Path(UPSCALER_DIR) / model_name
    logger.info(f"Would load external upscaler from: {model_path} with scale {scale}")
    # TODO: load .pth or .onnx and apply upscale
    # Placeholder logic until we implement full support
    return image

def upscale_image(image: Image.Image, scale: float, upscaler_name: str) -> Image.Image:
    """
    Upscale an image using a specified upscaler.
    
    Args:
        image (PIL.Image.Image): Input image
        scale (float): Desired scale factor (e.g., 2.0)
        upscaler_name (str): Name of the upscaler (e.g., "Latent", "4x-AnimeSharp")

    Returns:
        PIL.Image.Image: Upscaled image
    """
    if not isinstance(upscaler_name, str):
        raise ValueError(f"upscaler_name must be a string, got {type(upscaler_name)}")

    if upscaler_name.lower() == "latent":
        logger.info("Using Latent (placeholder) upscaler")
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        return image.resize((new_width, new_height), Image.BILINEAR)

    # Try to load from stable_diffusion/upscalers
    upscaler_dir = Path("stable_diffusion/upscalers")
    upscaler_path = upscaler_dir / f"{upscaler_name}.pth"

    if not upscaler_path.exists():
        raise FileNotFoundError(f"Upscaler model not found: {upscaler_path}")

    try:
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor, to_pil_image
        from basicsr.archs.rrdbnet_arch import RRDBNet as BaseRRDBNet
        import torch.nn as nn

        # Create model with correct architecture
        class CustomRRDBNet(BaseRRDBNet):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Override conv_hr and conv_last to match state dict
                self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)  # [64, 64, 3, 3] - matches state dict
                self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)  # [3, 64, 3, 3] - matches state dict

        # Create model with matching architecture
        model = CustomRRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4  # This is the model's built-in scale factor
        )

        # Load state dict
        state_dict = torch.load(upscaler_path, map_location="cpu")
        if isinstance(state_dict, dict) and 'params' in state_dict:
            state_dict = state_dict['params']

        # Load state dict into model
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Process image
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert RGB to BGR
        image_bgr = image.copy()
        r, g, b = image_bgr.split()
        image_bgr = Image.merge("RGB", (b, g, r))
        
        input_tensor = to_tensor(image_bgr).unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        # If output is in [-1, 1], rescale to [0, 1]
        if output_tensor.min() < 0:
            output_tensor = (output_tensor + 1) / 2
        # Clamp to [0, 1]
        output_tensor = output_tensor.clamp(0, 1)

        # Swap from BGR to RGB if needed
        output_tensor = output_tensor.squeeze(0).cpu()
        if output_tensor.shape[0] == 3:
            output_tensor = output_tensor[[2, 1, 0], :, :]

        # Convert to uint8 and to PIL
        upscaled = to_pil_image(output_tensor)

        # If the model's scale factor is greater than desired scale, resize down
        model_scale = 4  # 4x-AnimeSharp is a 4x upscaler
        if model_scale > scale:
            # Calculate the final size
            final_width = int(image.width * scale)
            final_height = int(image.height * scale)
            logger.info(f"Resizing from {upscaled.size} to {final_width}x{final_height} to match desired scale {scale}x")
            return upscaled.resize((final_width, final_height), Image.LANCZOS)
        return upscaled

    except Exception as e:
        logger.exception("Failed to load or apply custom upscaler")
        raise RuntimeError(f"Upscaler error: {str(e)}")