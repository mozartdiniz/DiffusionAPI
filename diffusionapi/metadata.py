"""
Metadata handling module for DiffusionAPI.
Implements the same metadata format as Stable Diffusion web UI for compatibility.
"""

import json
import os
import piexif
import piexif.helper
from PIL import Image, PngImagePlugin
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_infotext(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 50,
    sampler_name: str = "Euler a",
    scheduler: str = "Karras",
    cfg_scale: float = 7.0,
    seed: int = -1,
    width: int = 512,
    height: int = 512,
    model_name: str = "",
    model_hash: str = "",
    vae_name: str = "",
    vae_hash: str = "",
    denoising_strength: Optional[float] = None,
    clip_skip: Optional[int] = None,
    tiling: bool = False,
    restore_faces: bool = False,
    extra_generation_params: Optional[Dict[str, Any]] = None,
    user: Optional[str] = None,
    version: str = "DiffusionAPI v1.0"
) -> str:
    """
    Create infotext string in the same format as Stable Diffusion web UI.
    
    Args:
        prompt: The main prompt
        negative_prompt: The negative prompt
        steps: Number of inference steps
        sampler_name: Name of the sampler used
        scheduler: Name of the scheduler used
        cfg_scale: CFG scale value
        seed: Random seed used
        width: Image width
        height: Image height
        model_name: Name of the model used
        model_hash: Hash of the model
        vae_name: Name of the VAE used
        vae_hash: Hash of the VAE
        denoising_strength: Denoising strength for img2img
        clip_skip: CLIP skip value
        tiling: Whether tiling was enabled
        restore_faces: Whether face restoration was enabled
        extra_generation_params: Additional parameters
        user: User who generated the image
        version: Version string
        
    Returns:
        Formatted infotext string
    """
    
    # Build generation parameters dictionary
    generation_params = {
        "Steps": steps,
        "Sampler": sampler_name,
        "Schedule type": scheduler,
        "CFG scale": cfg_scale,
        "Seed": seed,
        "Size": f"{width}x{height}",
        "Model": model_name if model_name else None,
        "Model hash": model_hash if model_hash else None,
        "VAE": vae_name if vae_name else None,
        "VAE hash": vae_hash if vae_hash else None,
        "Denoising strength": denoising_strength,
        "Clip skip": None if clip_skip is None or clip_skip <= 1 else clip_skip,
        "Tiling": "True" if tiling else None,
        "Face restoration": "CodeFormer" if restore_faces else None,
        "User": user,
        "Version": version,
    }
    
    # Add extra parameters if provided
    if extra_generation_params:
        generation_params.update(extra_generation_params)
    
    # Filter out None values and format the parameters
    generation_params_text = ", ".join([
        k if k == v else f'{k}: {_quote_value(v)}' 
        for k, v in generation_params.items() 
        if v is not None
    ])
    
    # Build the final infotext
    negative_prompt_text = f"\nNegative prompt: {negative_prompt}" if negative_prompt else ""
    
    return f"{prompt}{negative_prompt_text}\n{generation_params_text}".strip()


def _quote_value(value: Any) -> str:
    """Quote a value if it contains spaces or special characters."""
    if isinstance(value, str):
        if " " in value or "," in value or ":" in value:
            return f'"{value}"'
        return value
    return str(value)


def save_image_with_metadata(
    image: Image.Image,
    filename: str,
    geninfo: str,
    extension: Optional[str] = None,
    existing_pnginfo: Optional[Dict[str, str]] = None,
    pnginfo_section_name: str = 'parameters',
    jpeg_quality: int = 85
) -> None:
    """
    Save image with metadata in the same format as Stable Diffusion web UI.
    
    Args:
        image: PIL Image to save
        filename: Output filename
        geninfo: Generation info string to embed
        extension: File extension (auto-detected if None)
        existing_pnginfo: Additional PNG info
        pnginfo_section_name: Name of the PNG info section
        jpeg_quality: JPEG quality (1-100)
    """
    
    if extension is None:
        extension = os.path.splitext(filename)[1]
    
    if not extension.startswith('.'):
        extension = '.' + extension
    
    image_format = Image.registered_extensions().get(extension, 'PNG')
    
    if extension.lower() == '.png':
        # Save as PNG with metadata in PNG info
        existing_pnginfo = existing_pnginfo or {}
        existing_pnginfo[pnginfo_section_name] = geninfo
        
        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in existing_pnginfo.items():
            pnginfo_data.add_text(k, str(v))
        
        image.save(filename, format=image_format, pnginfo=pnginfo_data)
        
    elif extension.lower() in (".jpg", ".jpeg", ".webp"):
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB" if extension.lower() == ".webp" else "L")
        
        # Save the image first
        if extension.lower() == ".webp":
            image.save(filename, format=image_format, quality=jpeg_quality, lossless=False)
        else:
            image.save(filename, format=image_format, quality=jpeg_quality, optimize=True)
        
        # Add EXIF metadata
        if geninfo:
            try:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo, encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, filename)
            except Exception as e:
                logger.warning(f"Failed to add EXIF metadata: {e}")
                
    elif extension.lower() == '.avif':
        # Save AVIF with EXIF metadata
        exif_bytes = None
        if geninfo:
            try:
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo, encoding="unicode")
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to create EXIF metadata: {e}")
        
        image.save(filename, format=image_format, quality=jpeg_quality, exif=exif_bytes)
        
    elif extension.lower() == ".gif":
        # Save GIF with comment
        image.save(filename, format=image_format, comment=geninfo)
        
    else:
        # Save other formats without metadata
        image.save(filename, format=image_format, quality=jpeg_quality)


def read_metadata_from_image(image: Image.Image) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Read metadata from image in the same format as Stable Diffusion web UI.
    
    Args:
        image: PIL Image to read metadata from
        
    Returns:
        Tuple of (generation_info, other_metadata)
    """
    
    items = (image.info or {}).copy()
    
    # Try to get generation info from PNG parameters
    geninfo = items.pop('parameters', None)
    
    # Try to get from EXIF data
    if "exif" in items:
        exif_data = items["exif"]
        try:
            exif = piexif.load(exif_data)
        except OSError:
            exif = None
            
        if exif:
            exif_comment = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
            try:
                exif_comment = piexif.helper.UserComment.load(exif_comment)
                if exif_comment:
                    geninfo = exif_comment
            except ValueError:
                try:
                    exif_comment = exif_comment.decode('utf8', errors="ignore")
                    if exif_comment:
                        geninfo = exif_comment
                except:
                    pass
    
    # Try to get from GIF comment
    elif "comment" in items:
        if isinstance(items["comment"], bytes):
            geninfo = items["comment"].decode('utf8', errors="ignore")
        else:
            geninfo = items["comment"]
    
    # Remove ignored keys
    ignored_keys = {
        'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
        'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
        'icc_profile', 'chromaticity', 'photoshop',
    }
    
    for field in ignored_keys:
        items.pop(field, None)
    
    return geninfo, items


def parse_infotext(infotext: str) -> Dict[str, Any]:
    """
    Parse infotext string into a dictionary of parameters.
    
    Args:
        infotext: The infotext string to parse
        
    Returns:
        Dictionary of parsed parameters
    """
    if not infotext:
        return {}
    
    lines = infotext.strip().split('\n')
    if not lines:
        return {}
    
    # First line is the prompt
    prompt = lines[0]
    
    # Parse parameters from the last line
    params = {}
    if len(lines) > 1:
        params_line = lines[-1]
        # Split by comma and parse key-value pairs
        for param in params_line.split(','):
            param = param.strip()
            if ':' in param:
                key, value = param.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                # Try to convert to appropriate type
                try:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                params[key] = value
    
    # Check for negative prompt
    negative_prompt = ""
    for line in lines[1:-1] if len(lines) > 2 else []:
        if line.startswith('Negative prompt:'):
            negative_prompt = line[16:].strip()
            break
    
    result = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        **params
    }
    
    return result 