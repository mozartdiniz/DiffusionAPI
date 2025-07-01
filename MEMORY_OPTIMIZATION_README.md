# Memory Optimization for Apple Silicon Macs

This document explains the memory optimization features implemented in DiffusionAPI to prevent memory issues on Apple Silicon Macs (M1, M2, M3) while maintaining excellent performance on NVIDIA GPUs.

## Problem

Apple Silicon Macs, particularly M3 MacBook Pro models, can experience memory pressure when running img2img operations with large models like SDXL. This can lead to:
- System freezes
- Automatic reboots
- Poor performance
- Out-of-memory errors

## Solution

We've implemented a comprehensive memory optimization system that:

1. **Automatically detects hardware and memory constraints**
2. **Dynamically adjusts image sizes** based on available memory
3. **Applies memory-efficient pipeline optimizations**
4. **Provides configurable settings** for different hardware configurations

## Features

### 🧠 Automatic Memory Detection

The system automatically detects:
- Total system memory
- Available memory
- Device type (MPS, CUDA, CPU)
- Memory pressure levels

### 📏 Dynamic Image Size Adjustment

Image sizes are automatically adjusted based on available memory:

| Memory Level | Apple Silicon | NVIDIA GPU | CPU |
|--------------|---------------|------------|-----|
| Very Low (<8GB) | 512x512 | 768x768 | 512x512 |
| Low (8-16GB) | 768x768 | 1024x1024 | 512x512 |
| Medium (16-24GB) | 1024x1024 | 1280x1280 | 768x768 |
| High (24-32GB) | 1280x1280 | 1536x1536 | 1024x1024 |
| Very High (>32GB) | 1536x1536 | 2048x2048 | 1024x1024 |

### ⚡ Pipeline Optimizations

The following optimizations are automatically applied:

- **VAE Tiling**: Reduces memory usage for large images
- **Attention Slicing**: Processes attention in smaller chunks
- **Model CPU Offload**: Moves unused model parts to CPU (Apple Silicon)
- **Sequential CPU Offload**: For very memory-constrained systems
- **Memory Efficient Attention**: For SDXL models
- **Reduced VAE Sample Size**: For SDXL on Apple Silicon

### 🔧 Configurable Settings

You can customize memory optimization behavior using:

#### Environment Variables

```bash
# Set maximum image size
export DIFFUSIONAPI_MAX_IMAGE_SIZE="1024x1024"

# Enable/disable specific optimizations
export DIFFUSIONAPI_ENABLE_VAE_TILING="true"
export DIFFUSIONAPI_ENABLE_ATTENTION_SLICING="true"
export DIFFUSIONAPI_ENABLE_CPU_OFFLOAD="true"
export DIFFUSIONAPI_ENABLE_SEQUENTIAL_CPU_OFFLOAD="false"
export DIFFUSIONAPI_ENABLE_MEMORY_EFFICIENT_ATTENTION="true"
export DIFFUSIONAPI_REDUCE_VAE_SAMPLE_SIZE="true"
export DIFFUSIONAPI_FORCE_CLEANUP_AFTER_GENERATION="true"
export DIFFUSIONAPI_CLEANUP_BEFORE_LOADING="true"
```

#### Custom Configuration File

Create a JSON file and set the environment variable:

```bash
export DIFFUSIONAPI_MEMORY_CONFIG="/path/to/your/memory_config.json"
```

Example configuration file:

```json
{
  "memory_thresholds": {
    "very_low": 6,
    "low": 12,
    "medium": 18,
    "high": 24,
    "very_high": 32
  },
  "max_image_sizes": {
    "mps": {
      "very_low": [512, 512],
      "low": [768, 768],
      "medium": [1024, 1024],
      "high": [1280, 1280],
      "very_high": [1536, 1536]
    }
  },
  "optimizations": {
    "enable_vae_tiling": true,
    "enable_attention_slicing": true,
    "enable_model_cpu_offload": true,
    "enable_sequential_cpu_offload": false
  }
}
```

## Usage

### For M3 MacBook Pro Users

1. **Default Settings**: The system automatically applies conservative settings for Apple Silicon
2. **Monitor Memory**: Check the logs for memory usage information
3. **Adjust if Needed**: Use environment variables to fine-tune settings

### For NVIDIA GPU Users

1. **Optimal Performance**: The system automatically applies aggressive settings for CUDA
2. **No Changes Needed**: Default settings are optimized for NVIDIA GPUs

### Testing

Run the memory optimization test suite:

```bash
python test_memory_optimization.py
```

This will verify that all memory optimization features are working correctly.

## Logging

The system provides detailed logging about memory optimization:

```
==> System memory: 16.0GB total, 8.5GB available
==> Using device: mps
==> Memory level: low
==> Image size adjusted from 2048x2048 to 768x768 to prevent memory issues
==> Applying memory optimizations for device: mps (memory level: low)
==> Enabled VAE tiling
==> Enabled attention slicing
==> Enabled model CPU offload
==> Reduced VAE sample size for memory optimization
```

## Troubleshooting

### Still Experiencing Memory Issues?

1. **Reduce Image Size**: Set a smaller `DIFFUSIONAPI_MAX_IMAGE_SIZE`
2. **Enable Sequential CPU Offload**: Set `DIFFUSIONAPI_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true`
3. **Close Other Applications**: Free up system memory
4. **Use Smaller Models**: Consider using SD 1.5 instead of SDXL

### Performance Too Slow?

1. **Increase Image Size**: Set a larger `DIFFUSIONAPI_MAX_IMAGE_SIZE`
2. **Disable CPU Offload**: Set `DIFFUSIONAPI_ENABLE_CPU_OFFLOAD=false`
3. **Disable Attention Slicing**: Set `DIFFUSIONAPI_ENABLE_ATTENTION_SLICING=false`

### Custom Configuration

For advanced users, create a custom configuration file with your specific requirements:

```json
{
  "memory_thresholds": {
    "very_low": 4,
    "low": 8,
    "medium": 16,
    "high": 24,
    "very_high": 32
  },
  "max_image_sizes": {
    "mps": {
      "very_low": [384, 384],
      "low": [512, 512],
      "medium": [768, 768],
      "high": [1024, 1024],
      "very_high": [1280, 1280]
    }
  }
}
```

## Technical Details

### Memory Management

- **Garbage Collection**: Automatic cleanup after each generation
- **Cache Clearing**: MPS/CUDA cache clearing
- **Memory Pooling**: Efficient memory allocation
- **Model Unloading**: Automatic model cleanup

### Pipeline Optimizations

- **VAE Tiling**: Processes large images in tiles
- **Attention Slicing**: Reduces attention memory usage
- **CPU Offload**: Moves unused components to CPU
- **Memory Efficient Attention**: Optimized attention computation

### Device-Specific Optimizations

#### Apple Silicon (MPS)
- Conservative memory limits
- Aggressive cleanup
- CPU offload for memory-constrained systems
- Reduced VAE sample size for SDXL

#### NVIDIA (CUDA)
- Aggressive memory limits
- Standard cleanup
- Memory pooling
- Full VAE sample size

#### CPU
- Very conservative limits
- Aggressive cleanup
- No memory pooling

## Contributing

To improve memory optimization:

1. Test on your specific hardware
2. Adjust configuration values
3. Submit pull requests with improvements
4. Report issues with detailed system information

## Support

For memory optimization issues:

1. Check the logs for memory usage information
2. Run the test suite to verify functionality
3. Try adjusting environment variables
4. Create a custom configuration file
5. Report issues with system specifications 