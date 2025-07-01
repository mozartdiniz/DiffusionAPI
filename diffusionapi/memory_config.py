"""
Memory optimization configuration for DiffusionAPI.

This file contains settings that can be customized for different hardware configurations,
especially for Apple Silicon Macs and other memory-constrained systems.
"""

import os
from typing import Dict, Any

# Default memory configuration
DEFAULT_MEMORY_CONFIG = {
    # Memory thresholds for different optimization levels (in GB)
    "memory_thresholds": {
        "very_low": 6,      # < 6GB - very conservative settings
        "low": 12,          # 6-12GB - conservative settings
        "medium": 18,       # 12-18GB - moderate settings
        "high": 24,         # 18-24GB - aggressive settings
        "very_high": 32     # > 24GB - very aggressive settings
    },
    
    # Maximum image sizes for different memory levels (width x height)
    "max_image_sizes": {
        "mps": {  # Apple Silicon Macs - More aggressive like WebUI
            "very_low": (768, 768),
            "low": (1024, 1024),
            "medium": (1536, 1536),
            "high": (2048, 2048),
            "very_high": (3072, 3072)
        },
        "cuda": {  # NVIDIA GPUs - Very aggressive like WebUI
            "very_low": (1024, 1024),
            "low": (1536, 1536),
            "medium": (2048, 2048),
            "high": (3072, 3072),
            "very_high": (4096, 4096)
        },
        "cpu": {  # CPU-only
            "very_low": (512, 512),
            "low": (768, 768),
            "medium": (1024, 1024),
            "high": (1536, 1536),
            "very_high": (2048, 2048)
        }
    },
    
    # Memory optimization settings - More aggressive like WebUI
    "optimizations": {
        "enable_vae_tiling": True,
        "enable_attention_slicing": True,
        "enable_model_cpu_offload": True,
        "enable_sequential_cpu_offload": False,  # Only for very low memory
        "enable_memory_efficient_attention": True,
        "reduce_vae_sample_size": False,  # Don't reduce for better quality
        "force_cleanup_after_generation": True,
        "cleanup_before_loading": True,
        "enable_gradient_checkpointing": True,  # WebUI-like optimization
        "enable_xformers_memory_efficient_attention": True,  # WebUI uses this
        "enable_vae_slicing": True,  # WebUI-like VAE optimization
        "enable_model_offload": True,  # WebUI-like model offloading
        "enable_attention_processor": True  # WebUI-like attention optimization
    },
    
    # Device-specific settings - More aggressive
    "device_settings": {
        "mps": {
            "dtype": "float16",
            "aggressive_cleanup": True,
            "max_concurrent_models": 1,
            "enable_memory_pooling": True,
            "enable_dynamic_tiling": True,  # WebUI-like dynamic tiling
            "tile_size": 512,  # WebUI default tile size
            "tile_overlap": 64,  # WebUI default overlap
            "max_tile_size": 1024  # Maximum tile size for large images
        },
        "cuda": {
            "dtype": "float16",
            "aggressive_cleanup": False,
            "max_concurrent_models": 2,
            "enable_memory_pooling": True,
            "enable_dynamic_tiling": True,
            "tile_size": 512,
            "tile_overlap": 64,
            "max_tile_size": 1024
        },
        "cpu": {
            "dtype": "float32",
            "aggressive_cleanup": True,
            "max_concurrent_models": 1,
            "enable_memory_pooling": False,
            "enable_dynamic_tiling": True,
            "tile_size": 256,
            "tile_overlap": 32,
            "max_tile_size": 512
        }
    },
    
    # LoRA memory settings
    "lora_settings": {
        "max_loras_per_generation": 5,
        "enable_lora_memory_optimization": True,
        "unload_loras_after_generation": True
    },
    
    # Hires fix settings - More aggressive like WebUI
    "hires_settings": {
        "max_upscale_factor": {
            "mps": 4.0,  # WebUI allows up to 4x
            "cuda": 8.0,  # WebUI allows up to 8x
            "cpu": 2.0
        },
        "use_external_upscaler_for_large_images": True,
        "large_image_threshold": {
            "mps": 2048,  # Higher threshold like WebUI
            "cuda": 4096,  # Higher threshold like WebUI
            "cpu": 1024
        }
    }
}

def get_memory_config() -> Dict[str, Any]:
    """
    Get memory configuration, allowing environment variable overrides.
    
    Environment variables that can be set:
    - DIFFUSIONAPI_MEMORY_CONFIG: Path to a JSON config file
    - DIFFUSIONAPI_MAX_IMAGE_SIZE: Override max image size (e.g., "1024x1024")
    - DIFFUSIONAPI_ENABLE_VAE_TILING: "true" or "false"
    - DIFFUSIONAPI_ENABLE_ATTENTION_SLICING: "true" or "false"
    - DIFFUSIONAPI_ENABLE_CPU_OFFLOAD: "true" or "false"
    """
    config = DEFAULT_MEMORY_CONFIG.copy()
    
    # Load custom config file if specified
    config_file = os.getenv("DIFFUSIONAPI_MEMORY_CONFIG")
    if config_file and os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                # Deep merge custom config with defaults
                _deep_merge(config, custom_config)
        except Exception as e:
            print(f"Warning: Failed to load custom memory config from {config_file}: {e}")
    
    # Override with environment variables
    max_size = os.getenv("DIFFUSIONAPI_MAX_IMAGE_SIZE")
    if max_size:
        try:
            width, height = map(int, max_size.split('x'))
            for device in config["max_image_sizes"]:
                for level in config["max_image_sizes"][device]:
                    config["max_image_sizes"][device][level] = (width, height)
        except ValueError:
            print(f"Warning: Invalid DIFFUSIONAPI_MAX_IMAGE_SIZE format: {max_size}")
    
    # Override optimization settings
    for opt_name in config["optimizations"]:
        env_var = f"DIFFUSIONAPI_{opt_name.upper()}"
        env_value = os.getenv(env_var)
        if env_value is not None:
            config["optimizations"][opt_name] = env_value.lower() == "true"
    
    return config

def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Deep merge two dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

def get_optimal_settings_for_memory(available_memory_gb: float, device: str) -> Dict[str, Any]:
    """
    Get optimal settings based on available memory and device.
    
    Args:
        available_memory_gb: Available system memory in GB
        device: Device type ("mps", "cuda", "cpu")
    
    Returns:
        Dictionary with optimal settings
    """
    config = get_memory_config()
    
    # Determine memory level
    thresholds = config["memory_thresholds"]
    if available_memory_gb < thresholds["very_low"]:
        memory_level = "very_low"
    elif available_memory_gb < thresholds["low"]:
        memory_level = "low"
    elif available_memory_gb < thresholds["medium"]:
        memory_level = "medium"
    elif available_memory_gb < thresholds["high"]:
        memory_level = "high"
    else:
        memory_level = "very_high"
    
    # Get device-specific settings
    device_config = config["device_settings"].get(device, config["device_settings"]["cpu"])
    
    # Get max image size for this memory level and device
    max_image_size = config["max_image_sizes"][device][memory_level]
    
    return {
        "memory_level": memory_level,
        "max_image_size": max_image_size,
        "dtype": device_config["dtype"],
        "aggressive_cleanup": device_config["aggressive_cleanup"],
        "optimizations": config["optimizations"],
        "device_settings": device_config,  # Include the full device settings
        "lora_settings": config["lora_settings"],
        "hires_settings": config["hires_settings"]
    }

# Export the main function
__all__ = ["get_memory_config", "get_optimal_settings_for_memory"] 