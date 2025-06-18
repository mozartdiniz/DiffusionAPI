# Diffusion API

A FastAPI-based implementation for Stable Diffusion image generation, supporting both standard and SDXL models, with hires fix and LoRA support. This is a simplified API focused on core image generation functionality.

## Features

- **Multi-Model Support**: Standard SD 1.5 and SDXL models
- **LoRA Support**: Load and apply LoRA models with automatic compatibility detection
- **Hires Fix**: Built-in upscaling with configurable parameters
- **External Upscalers**: Support for Real-ESRGAN and other upscalers
- **Model Aliases**: User-friendly model names
- **LoRA Compatibility Testing**: Tools to test LoRA compatibility with models

## LoRA Compatibility

The API now includes robust LoRA compatibility handling:

### Automatic Compatibility Detection

- **Graceful Error Handling**: Incompatible LoRAs are skipped with clear warnings
- **Smart File Detection**: Automatically finds LoRA files with different extensions (.safetensors, .bin, .pt, .pth)
- **Multiple Weight Formats**: Supports various LoRA weight file naming conventions
- **Detailed Error Messages**: Provides specific guidance for compatibility issues

### Compatibility Testing Tool

Use the included test script to check LoRA compatibility before generation:

```bash
# Test a single LoRA
python test_lora_compatibility.py --model "your_model_name" --lora "path/to/lora.safetensors"

# Test all LoRAs in a directory
python test_lora_compatibility.py --model "your_model_name" --lora-dir "path/to/lora/directory"
```

### Comprehensive Compatibility Testing

For thorough testing of all models against all LoRAs:

```bash
# List available models and LoRAs first
python list_available_models_loras.py

# Run comprehensive compatibility test
python test_lora_compatibility.py

# Generate report with custom filename
python test_lora_compatibility.py --output my_compatibility_report.txt
```

The comprehensive test will:

- **Automatically discover** all models in the `models/` directory and model labels
- **Find all LoRAs** in the `loras/` directory
- **Test every combination** of model + LoRA
- **Generate a detailed report** with compatibility results
- **Provide statistics** on success rates and compatibility issues

**Expected Directory Structure:**

```
DiffusionAPI/
├── stable_diffusion/          # Main stable diffusion directory
│   ├── models/                # Model directories
│   │   ├── models--digiplay--ChikMix_V3/
│   │   ├── models--misri--plantMilkModelSuite_hempII/
│   │   └── your-custom-model/
│   ├── loras/                 # LoRA files
│   │   ├── Detail_Tweaker_smaller.safetensors
│   │   ├── Cyberpunk_Anime.safetensors
│   │   └── Humans.safetensors
│   └── upscalers/             # Upscaler models
├── models/                    # Alternative models directory (optional)
└── outputs/                   # Generated images
```

**Sample Report Output:**

```
================================================================================
COMPREHENSIVE LoRA COMPATIBILITY REPORT
================================================================================
Generated: 2024-01-15 14:30:25
Total Tests: 24
Compatible: 18
Incompatible: 6
Success Rate: 75.0%

SUMMARY BY MODEL:
----------------------------------------
Amanatsu: 6/8 (75.0%)
PlantMilk (HempII): 6/8 (75.0%)
stable-diffusion-v1-5: 6/8 (75.0%)

MODEL: Amanatsu
============================================================
✓ COMPATIBLE LoRAs:
  • detail_tweaker
  • anime_style
  • realistic_face

✗ INCOMPATIBLE LoRAs:
  • sd15_specific_lora
    Reason: LoRA 'sd15_specific_lora' appears to be designed for SD 1.5 models, but you're using an SDXL model.
```

### Common Compatibility Issues

1. **SD vs SDXL Mismatch**: LoRAs trained for SD 1.5 won't work with SDXL models and vice versa
2. **Architecture Differences**: Different model architectures have incompatible tensor shapes
3. **File Format Issues**: Ensure LoRA files are in the correct format (.safetensors recommended)

### Troubleshooting LoRA Issues

If you encounter LoRA loading errors:

1. **Check Model Type**: Ensure your LoRA matches your model type (SD vs SDXL)
2. **Verify File Integrity**: Make sure LoRA files are not corrupted
3. **Use Compatibility Tool**: Run the test script to identify issues
4. **Check Logs**: Look for specific error messages in the API logs

### Example: Detail_Tweaker_smaller Error

If you see errors like this:

```
size mismatch for mid_block.attentions.0.proj_in.lora_A.Detail_Tweaker_smaller.weight
```

This indicates the LoRA was trained for a different model architecture. The new system will:

1. **Skip the incompatible LoRA** with a clear warning
2. **Continue generation** with compatible LoRAs
3. **Provide guidance** on model compatibility

**Solution**: Use an SDXL model with SDXL-compatible LoRAs, or use a standard SD model with SD-compatible LoRAs.

## API Documentation

### Types

```typescript
// Common Types
type LoRA = {
  name: string; // LoRA model name
  scale: number; // LoRA scale (default: 1.0)
};

type ModelInfo = {
  name: string; // Model name
  label: string; // User-friendly label for this model
};

type UpscalerInfo = {
  name: string; // Upscaler name
  path: string; // Path to upscaler file
};

type HiresConfig = {
  enabled: boolean; // Whether hires fix is enabled
  scale: number; // Upscaling factor (default: 2.0)
  upscaler: string; // Upscaler to use ("Latent" or external upscaler name)
  steps: number; // Number of steps for upscaling (default: 20)
  denoising_strength: number; // Denoising strength for upscaling (default: 0.4)
};

type SamplingMethod =
  | "Euler"
  | "Euler a"
  | "Heun"
  | "DPM2"
  | "DPM++ 2S a"
  | "DPM++ 2M"
  | "DPM++ 2M Karras"
  | "DPM++ SDE"
  | "DPM++ SDE Karras"
  | "DPM fast"
  | "DPM adaptive"
  | "LMS"
  | "LMS Karras"
  | "DDIM"
  | "PLMS"
  | "UniPC";

// Request Types
type Txt2ImgRequest = {
  prompt: string; // Main prompt
  negative_prompt?: string; // Optional negative prompt
  steps?: number; // Number of inference steps (default: 30)
  cfg_scale?: number; // Guidance scale (default: 7.0)
  width?: number; // Image width (default: 512)
  height?: number; // Image height (default: 512)
  model?: string; // Model name (default: "stable-diffusion-v1-5")
  sampler_name?: string; // Sampler name (default: "DPM++ 2M Karras")
  scheduler_type?: string; // Scheduler type (default: "karras")
  fileType?: "jpg" | "png"; // Output file type (default: "jpg")
  jpeg_quality?: number; // JPEG quality (default: 85)
  loras?: (LoRA | string)[]; // Array of LoRAs (can be objects or just names)
  hires?: HiresConfig; // Hires fix configuration
  refiner_checkpoint?: string; // Optional SDXL refiner model
  refiner_switch_at?: number; // When to switch to refiner (default: 0.8)
  seed?: number; // Optional random seed for reproducibility
};

// Response Types
type QueueStatusResponse = {
  status: "success" | "error"; // Response status
  job_id: string; // Job identifier
  progress: number; // Progress percentage (0-100)
  current_phase: string; // Current phase ("generation", "initial generation", "upscaling", "done", etc.)
  state: string; // Current state ("queued", "processing", "done", "error")
  error?: string; // Error message if any
  payload?: Txt2ImgRequest; // Original request payload (only when complete)
  image?: string; // Base64 encoded image (only when complete)
  output_path?: string; // Path to saved image (only when complete)
  width?: number; // Final image width (only when complete)
  height?: number; // Final image height (only when complete)
  file_type?: "jpg" | "png"; // Output file type (only when complete)
  jpeg_quality?: number; // JPEG quality used (only when complete)
  generation_time_sec?: number; // Total generation time (only when complete)
  memory_before_mb?: number; // Memory usage before generation (only when complete)
  memory_after_mb?: number; // Memory usage after generation (only when complete)
  seed?: number; // The seed used for generation (always present when complete)
};

type Txt2ImgResponse = {
  job_id: string; // Job identifier
  status: "queued"; // Initial status
};

// Response Types for List Endpoints
type ModelsResponse = {
  status: "success" | "error";
  models: ModelInfo[];
  error?: string;
};

type LorasResponse = {
  status: "success" | "error";
  loras: LoRA[];
  error?: string;
};

type UpscalersResponse = {
  status: "success" | "error";
  upscalers: UpscalerInfo[];
  error?: string;
};
```

### Endpoints

#### `GET /hello`

Simple health check endpoint.

**Response:**

```typescript
type HelloResponse = {
  ok: boolean;
};
```

#### `GET /models`

List all available models with their user-friendly labels.

**Response:** `ModelsResponse`

**Example Response:**

```json
{
  "status": "success",
  "models": [
    {
      "name": "John6666__amanatsu-illustrious-v11-sdxl",
      "label": "Amanatsu"
    },
    {
      "name": "models--John6666--ilustmix-v6-sdxl",
      "label": "Ilustmix"
    },
    {
      "name": "models--Meina--MeinaMix_V11",
      "label": "MeinaMix V11"
    },
    {
      "name": "models--digiplay--ChikMix_V3",
      "label": "ChikMix V3"
    },
    {
      "name": "models--misri--plantMilkModelSuite_hempII",
      "label": "PlantMilk (HempII)"
    },
    {
      "name": "models--misri--plantMilkModelSuite_walnut",
      "label": "PlantMilk (Walnut)"
    }
  ]
}
```

#### `GET /loras`

List all available LoRA models.

**Response:** `LorasResponse`

**Example Response:**

```json
{
  "status": "success",
  "loras": [
    {
      "name": "my-lora",
      "scale": 1.0
    },
    {
      "name": "another-lora",
      "scale": 1.0
    }
  ]
}
```

#### `GET /upscalers`

List all available upscalers.

**Response:** `UpscalersResponse`

**Example Response:**

```json
{
  "status": "success",
  "upscalers": [
    {
      "name": "Latent",
      "path": "internal"
    },
    {
      "name": "ESRGAN_4x",
      "path": "stable_diffusion/upscalers/ESRGAN_4x.pth"
    }
  ]
}
```

#### `GET /sampling_methods`

List all available sampling methods and scheduler types.

**Response:**

```typescript
type SamplingMethodsResponse = {
  status: "success" | "error";
  sampling_methods: string[];
  scheduler_types: string[];
  error?: string;
};
```

**Example Response:**

```json
{
  "status": "success",
  "sampling_methods": [
    "Euler",
    "Euler a",
    "Heun",
    "DPM2",
    "DPM++ 2S a",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM fast",
    "DPM adaptive",
    "LMS",
    "LMS Karras",
    "DDIM",
    "PLMS",
    "UniPC"
  ],
  "scheduler_types": ["default", "karras", "exponential", "ddim", "pndm"]
}
```

**Usage:**

- Use the values from `sampling_methods` for the `sampler_name` field in your payload.
- Use the values from `scheduler_types` for the `scheduler_type` field in your payload.

#### `GET /models/aliases`

List all available models with their user-friendly labels.

**Response:**

```typescript
type ModelAliasesResponse = {
  status: "success" | "error";
  models: {
    name: string; // Actual model name
    label: string; // User-friendly label
  }[];
  message: string;
};
```

**Example Response:**

```json
{
  "status": "success",
  "models": [
    {
      "name": "John6666__amanatsu-illustrious-v11-sdxl",
      "label": "Amanatsu"
    },
    {
      "name": "models--digiplay--ChikMix_V3",
      "label": "ChikMix V3"
    },
    {
      "name": "models--misri--plantMilkModelSuite_hempII",
      "label": "PlantMilk (HempII)"
    }
  ]
}
```

## Model Label System

The DiffusionAPI includes a user-friendly model label system that allows you to use simple, memorable labels instead of the full model names.

### Available Model Labels

- `Amanatsu` → `John6666__amanatsu-illustrious-v11-sdxl`
- `ChikMix V3` → `models--digiplay--ChikMix_V3`
- `Ilustmix` → `models--John6666--ilustmix-v6-sdxl`
- `MeinaMix V11` → `models--Meina--MeinaMix_V11`
- `Pastel Mix` → `models--mirroring--pastel-mix`
- `PlantMilk (HempII)` → `models--misri--plantMilkModelSuite_hempII`
- `PlantMilk (Walnut)` → `models--misri--plantMilkModelSuite_walnut`

### Usage Examples

Instead of using the full model name:

```json
{
  "prompt": "a beautiful anime girl",
  "model": "John6666__amanatsu-illustrious-v11-sdxl"
}
```

You can use the friendly label:

```json
{
  "prompt": "a beautiful anime girl",
  "model": "Amanatsu"
}
```

### Listing Available Models

You can list all available models and their labels using:

**Command line:**

```bash
python list_models.py
```

**API endpoint:**

```bash
curl http://localhost:8000/models/aliases
```

**Python script:**

```

```
