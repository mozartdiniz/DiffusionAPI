import os
import warnings
from PIL import Image
from pathlib import Path
import logging
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from diffusers.utils.torch_utils import is_compiled_module
import tqdm
import cv2

# Suppress torchvision warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
warnings.filterwarnings("ignore", message="The torchvision.transforms.functional_tensor module is deprecated")

logger = logging.getLogger(__name__)

# Handle basicsr import with compatibility workaround
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    BASICSR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"basicsr import failed: {e}. External upscalers will not be available.")
    BASICSR_AVAILABLE = False
    RRDBNet = None

INTERNAL_UPSCALERS = {
    "Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (bicubic antialiased)",
    "Latent (nearest)", "Latent (nearest-exact)",
    "Lanczos", "Nearest", "None"
}

# Get upscaler directory from environment or use default
UPSCALER_DIR = Path(os.getenv("UPSCALERS_DIR", "stable_diffusion/upscalers")).resolve()
UPSCALER_DIR.mkdir(parents=True, exist_ok=True)

def is_internal_upscaler(name: str) -> bool:
    return name in INTERNAL_UPSCALERS

def apply_internal_upscale(image: Image.Image, upscaler: str, scale: float = 2.0) -> Image.Image:
    """Apply internal upscaler to image."""
    logger.info(f"Applying internal upscaler: {upscaler} with scale {scale}")
    w, h = image.size
    new_size = (int(w * scale), int(h * scale))

    if upscaler == "None":
        return image
    elif upscaler == "Lanczos":
        return image.resize(new_size, Image.Resampling.LANCZOS)
    elif upscaler == "Nearest":
        return image.resize(new_size, Image.Resampling.NEAREST)
    elif "bicubic" in upscaler.lower():
        return image.resize(new_size, Image.Resampling.BICUBIC)
    elif "nearest" in upscaler.lower():
        return image.resize(new_size, Image.Resampling.NEAREST)
    elif "antialiased" in upscaler.lower():
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        logger.warning(f"Unknown internal upscaler {upscaler}, falling back to BICUBIC")
        return image.resize(new_size, Image.Resampling.BICUBIC)

def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to BGR torch tensor."""
    # Convert PIL image to numpy array in RGB format
    img = np.array(img.convert("RGB"))
    logger.info(f"Input RGB image - R: {img[:,:,0].mean():.2f}, G: {img[:,:,1].mean():.2f}, B: {img[:,:,2].mean():.2f}")
    
    # Convert RGB to BGR by swapping channels
    img = img[:, :, ::-1].copy()  # flip RGB to BGR and ensure contiguous
    logger.info(f"After BGR conversion - B: {img[:,:,0].mean():.2f}, G: {img[:,:,1].mean():.2f}, R: {img[:,:,2].mean():.2f}")
    
    # Convert to CHW format and normalize
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255.0  # Rescale to [0, 1]
    tensor = torch.from_numpy(img).float()  # Ensure float32
    logger.info(f"Final tensor - B: {tensor[0].mean():.3f}, G: {tensor[1].mean():.3f}, R: {tensor[2].mean():.3f}")
    return tensor

def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert BGR torch tensor to PIL image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    
    logger.info(f"Input tensor - B: {tensor[0].mean():.3f}, G: {tensor[1].mean():.3f}, R: {tensor[2].mean():.3f}")
    
    # Convert to numpy and ensure proper range
    arr = tensor.float().cpu().numpy()
    logger.info(f"After numpy conversion - B: {arr[0].mean():.3f}, G: {arr[1].mean():.3f}, R: {arr[2].mean():.3f}")
    
    # Convert from CHW to HWC and rescale to 0-255
    arr = np.transpose(arr, (1, 2, 0))  # CHW to HWC
    arr = (arr * 255.0).round().astype(np.uint8)
    logger.info(f"After rescale - B: {arr[:,:,0].mean():.2f}, G: {arr[:,:,1].mean():.2f}, R: {arr[:,:,2].mean():.2f}")
    
    # Convert BGR to RGB by swapping channels
    arr = arr[:, :, ::-1].copy()  # flip BGR to RGB and ensure contiguous
    logger.info(f"Final RGB - R: {arr[:,:,0].mean():.2f}, G: {arr[:,:,1].mean():.2f}, B: {arr[:,:,2].mean():.2f}")
    
    return Image.fromarray(arr, "RGB")

def upscale_pil_patch(model, img: Image.Image, device: torch.device) -> Image.Image:
    """Upscale a single PIL image patch using the model."""
    # Convert to tensor and ensure proper format
    tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
    tensor = tensor.to(device=device, dtype=torch.float32)  # Explicitly set float32
    
    with torch.inference_mode():
        # Model expects BGR input and outputs BGR
        output = model(tensor)
        logger.info(f"Model output - B: {output[0,0].mean():.3f}, G: {output[0,1].mean():.3f}, R: {output[0,2].mean():.3f}")
        
        # Clamp output to valid range [0,1]
        output = torch.clamp(output, 0, 1)
        logger.info(f"After clamping - B: {output[0,0].mean():.3f}, G: {output[0,1].mean():.3f}, R: {output[0,2].mean():.3f}")
        
        # Convert BGR output back to RGB for PIL
        return torch_bgr_to_pil_image(output)

def upscale_with_model(
    model: torch.nn.Module,
    img: Image.Image,
    *,
    tile_size: int = 512,
    tile_overlap: int = 32,
    device: torch.device,
    desc="tiled upscale",
) -> Image.Image:
    """Upscale image using tiling for large images."""
    if tile_size <= 0:
        logger.debug("Upscaling without tiling")
        return upscale_pil_patch(model, img, device)

    # Calculate grid
    w, h = img.size
    tiles = []
    for y in range(0, h, tile_size - tile_overlap):
        for x in range(0, w, tile_size - tile_overlap):
            # Calculate tile coordinates
            tile_x = min(x, w - tile_size)
            tile_y = min(y, h - tile_size)
            tile = img.crop((tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))
            tiles.append((tile_x, tile_y, tile))

    # Process tiles
    upscaled_tiles = []
    with tqdm.tqdm(total=len(tiles), desc=desc) as pbar:
        for tile_x, tile_y, tile in tiles:
            upscaled = upscale_pil_patch(model, tile, device)
            logger.debug(f"Tile at ({tile_x}, {tile_y}) range: min={np.array(upscaled).min()}, max={np.array(upscaled).max()}, mean={np.array(upscaled).mean():.2f}")
            upscaled_tiles.append((tile_x, tile_y, upscaled))
            pbar.update(1)

    # Combine tiles
    scale = upscaled_tiles[0][2].width // tile_size
    result = Image.new('RGB', (w * scale, h * scale))
    
    for tile_x, tile_y, upscaled in upscaled_tiles:
        # Calculate overlap regions
        if tile_x > 0:
            overlap_x = tile_overlap * scale
            for x in range(overlap_x):
                alpha = x / overlap_x
                for y in range(upscaled.height):
                    left_pixel = result.getpixel((tile_x * scale + x, tile_y * scale + y))
                    right_pixel = upscaled.getpixel((x, y))
                    blended = tuple(int(l * (1 - alpha) + r * alpha) for l, r in zip(left_pixel, right_pixel))
                    result.putpixel((tile_x * scale + x, tile_y * scale + y), blended)
        
        if tile_y > 0:
            overlap_y = tile_overlap * scale
            for y in range(overlap_y):
                alpha = y / overlap_y
                for x in range(upscaled.width):
                    top_pixel = result.getpixel((tile_x * scale + x, tile_y * scale + y))
                    bottom_pixel = upscaled.getpixel((x, y))
                    blended = tuple(int(t * (1 - alpha) + b * alpha) for t, b in zip(top_pixel, bottom_pixel))
                    result.putpixel((tile_x * scale + x, tile_y * scale + y), blended)
        
        # Paste non-overlapping regions
        result.paste(upscaled, (tile_x * scale, tile_y * scale))

    logger.debug(f"Final result range: min={np.array(result).min()}, max={np.array(result).max()}, mean={np.array(result).mean():.2f}")
    return result

def upscale_image(image: Image.Image, scale: float, upscaler_name: str) -> Image.Image:
    """Main upscaling function that handles both internal and external upscalers."""
    if not isinstance(upscaler_name, str):
        raise ValueError(f"upscaler_name must be a string, got {type(upscaler_name)}")

    # Handle internal upscalers
    if upscaler_name.lower() == "latent" or is_internal_upscaler(upscaler_name):
        return apply_internal_upscale(image, upscaler_name, scale)

    # Handle external upscalers
    logger.info(f"Attempting to use external upscaler: {upscaler_name}")
    
    if not BASICSR_AVAILABLE:
        raise RuntimeError("External upscalers are not available because basicsr failed to import. Please check your torchvision version compatibility.")
    
    upscaler_path = UPSCALER_DIR / f"{upscaler_name}.pth"

    if not upscaler_path.exists():
        raise FileNotFoundError(f"External upscaler model not found: {upscaler_path}")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Determine model scale
        native_model_scale = 4
        if "4x" in upscaler_name.lower():
            native_model_scale = 4
        elif "2x" in upscaler_name.lower():
            native_model_scale = 2
        logger.info(f"Using native model scale: {native_model_scale}x")

        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=native_model_scale
        ).to(device=device, dtype=torch.float32)  # Explicitly set float32

        # Load weights
        loadnet = torch.load(upscaler_path, map_location=device)
        logger.debug(f"Loaded model weights range: min={min(p.min().item() for p in model.parameters()):.3f}, max={max(p.max().item() for p in model.parameters()):.3f}")
        
        if isinstance(loadnet, dict):
            if 'params_ema' in loadnet:
                model.load_state_dict(loadnet['params_ema'], strict=False)
            elif 'params' in loadnet:
                model.load_state_dict(loadnet['params'], strict=False)
            elif 'state_dict' in loadnet:
                model.load_state_dict(loadnet['state_dict'], strict=False)
            else:
                model.load_state_dict(loadnet, strict=False)
        else:
            model.load_state_dict(loadnet, strict=False)
        
        model.eval()
        logger.debug(f"Model parameters after loading range: min={min(p.min().item() for p in model.parameters()):.3f}, max={max(p.max().item() for p in model.parameters()):.3f}")

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Upscale with tiling
        upscaled = upscale_with_model(
            model,
            image,
            tile_size=512,  # Adjust based on your GPU memory
            tile_overlap=32,
            device=device,
            desc=f"Upscaling with {upscaler_name}"
        )

        # Resize to target scale if needed
        if native_model_scale != scale:
            target_w = int(image.width * scale)
            target_h = int(image.height * scale)
            upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)

        return upscaled

    except Exception as e:
        logger.exception(f"Failed to upscale image with {upscaler_name}")
        raise RuntimeError(f"Upscaler error: {str(e)}")