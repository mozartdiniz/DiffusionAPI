#!/usr/bin/env python3
"""
Example usage of the metadata functionality.
This shows how to save and read metadata in your own applications.
"""

from PIL import Image
import numpy as np
from diffusionapi.metadata import (
    create_infotext, 
    save_image_with_metadata, 
    read_metadata_from_image,
    parse_infotext
)

def example_basic_usage():
    """Basic example of creating and saving metadata."""
    
    # Create a simple test image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # Create metadata
    geninfo = create_infotext(
        prompt="a beautiful sunset over mountains",
        negative_prompt="blurry, low quality",
        steps=25,
        sampler_name="Euler a",
        scheduler="Karras",
        cfg_scale=7.5,
        seed=42,
        width=512,
        height=512,
        model_name="stable-diffusion-v1-5",
        user="example_user"
    )
    
    print("Generated metadata:")
    print(geninfo)
    print()
    
    # Save image with metadata
    save_image_with_metadata(
        image=image,
        filename="example_output.png",
        geninfo=geninfo,
        extension=".png"
    )
    
    print("Image saved with metadata!")

def example_read_metadata():
    """Example of reading metadata from an image."""
    
    # Read metadata from the saved image
    with Image.open("example_output.png") as img:
        geninfo, other_metadata = read_metadata_from_image(img)
    
    if geninfo:
        print("Found metadata:")
        print(geninfo)
        print()
        
        # Parse the metadata
        params = parse_infotext(geninfo)
        print("Parsed parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        print("No metadata found!")

def example_custom_parameters():
    """Example with custom generation parameters."""
    
    # Create image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    # Create metadata with custom parameters
    geninfo = create_infotext(
        prompt="an anime girl with blue hair",
        negative_prompt="bad anatomy, blurry",
        steps=30,
        sampler_name="DPM++ 2M Karras",
        scheduler="Karras",
        cfg_scale=8.0,
        seed=12345,
        width=512,
        height=512,
        model_name="stable-diffusion-v1-5",
        extra_generation_params={
            "LoRAs": "anime_style:0.8, detail_enhancer:0.6",
            "Custom setting": "high quality",
            "Processing time": "15.2s",
            "GPU memory": "8GB"
        },
        user="anime_artist"
    )
    
    print("Custom metadata example:")
    print(geninfo)
    print()
    
    # Save as JPEG with metadata
    save_image_with_metadata(
        image=image,
        filename="custom_example.jpg",
        geninfo=geninfo,
        extension=".jpg",
        jpeg_quality=90
    )
    
    print("Custom example saved as JPEG!")

def example_webui_compatibility():
    """Example showing compatibility with Stable Diffusion web UI format."""
    
    # This is the exact format that Stable Diffusion web UI uses
    webui_infotext = """a beautiful landscape, masterpiece, best quality
Negative prompt: blurry, low quality, distorted
Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7.0, Seed: 42, Size: 512x512, Model: stable-diffusion-v1-5, Model hash: abc123def456, VAE: vae-ft-mse-840000-ema-pruned, VAE hash: def456abc123, Denoising strength: 0.75, Clip skip: 2, Face restoration: CodeFormer, User: webui_user, Version: 1.6.0"""
    
    print("Web UI format infotext:")
    print(webui_infotext)
    print()
    
    # Parse it using our function
    params = parse_infotext(webui_infotext)
    print("Parsed Web UI infotext:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create a new image with this metadata
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    save_image_with_metadata(
        image=image,
        filename="webui_compatible.png",
        geninfo=webui_infotext,
        extension=".png"
    )
    
    print("\nWeb UI compatible image saved!")

if __name__ == "__main__":
    print("=== Metadata Usage Examples ===\n")
    
    print("1. Basic usage:")
    example_basic_usage()
    print()
    
    print("2. Reading metadata:")
    example_read_metadata()
    print()
    
    print("3. Custom parameters:")
    example_custom_parameters()
    print()
    
    print("4. Web UI compatibility:")
    example_webui_compatibility()
    print()
    
    print("=== All examples completed! ===")
    print("Check the generated files:")
    print("- example_output.png")
    print("- custom_example.jpg") 
    print("- webui_compatible.png")
    print("\nYou can open these images in Stable Diffusion web UI to verify compatibility!") 