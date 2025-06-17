# Diffusion API

A FastAPI-based implementation for Stable Diffusion image generation, supporting both standard and SDXL models, with hires fix and LoRA support. This is a simplified API focused on core image generation functionality.

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

```python
from diffusionapi.generate import get_model_labels
labels = get_model_labels()
for model_name, label in labels.items():
    print(f"{label} -> {model_name}")
```

#### `POST /txt2img`

Submit a new image generation job.

**Request Body:** `Txt2ImgRequest`

**Response:** `Txt2ImgResponse`

**Example Request:**

```json
{
  "prompt": "a beautiful green forest with a river and a waterfall",
  "negative_prompt": "blurry, bad quality",
  "steps": 30,
  "cfg_scale": 7.0,
  "width": 1024,
  "height": 1024,
  "model": "stable-diffusion-xl-base-1.0",
  "refiner_checkpoint": "stable-diffusion-xl-refiner-1.0",
  "refiner_switch_at": 0.8,
  "sampler_name": "DPM++ 2M Karras",
  "scheduler_type": "karras",
  "fileType": "jpg",
  "jpeg_quality": 85,
  "loras": [
    {
      "name": "my-lora",
      "scale": 0.8
    }
  ],
  "hires": {
    "enabled": true,
    "scale": 2.0,
    "upscaler": "Latent",
    "steps": 20,
    "denoising_strength": 0.4
  },
  "seed": 123456789
}
```

**Example Response:**

```json
{
  "job_id": "242d6f93-ea2d-4d5e-b820-0b61468558be",
  "status": "queued"
}
```

#### `GET /queue/{job_id}`

Check the status of a generation job.

**Path Parameters:**

- `job_id`: string (UUID of the job)

**Response:** `QueueStatusResponse`

**Example Response (In Progress):**

```json
{
  "status": "success",
  "job_id": "242d6f93-ea2d-4d5e-b820-0b61468558be",
  "progress": 45.5,
  "current_phase": "initial generation",
  "state": "processing",
  "error": null
}
```

**Example Response (Complete):**

```json
{
  "status": "success",
  "job_id": "242d6f93-ea2d-4d5e-b820-0b61468558be",
  "progress": 100,
  "current_phase": "done",
  "state": "done",
  "error": null,
  "payload": {
    "prompt": "a beautiful green forest with a river and a waterfall",
    "negative_prompt": "blurry, bad quality",
    "steps": 30,
    "cfg_scale": 7.0,
    "width": 1024,
    "height": 1024,
    "model": "stable-diffusion-xl-base-1.0",
    "refiner_checkpoint": "stable-diffusion-xl-refiner-1.0",
    "refiner_switch_at": 0.8,
    "sampler_name": "DPM++ 2M Karras",
    "scheduler_type": "karras",
    "fileType": "jpg",
    "jpeg_quality": 85,
    "loras": [
      {
        "name": "my-lora",
        "scale": 0.8
      }
    ],
    "hires": {
      "enabled": true,
      "scale": 2.0,
      "upscaler": "Latent",
      "steps": 20,
      "denoising_strength": 0.4
    },
    "seed": 123456789
  },
  "image": "base64...",
  "output_path": "outputs/242d6f93-ea2d-4d5e-b820-0b61468558be.jpg",
  "width": 1024,
  "height": 1024,
  "file_type": "jpg",
  "jpeg_quality": 85,
  "generation_time_sec": 58.36,
  "memory_before_mb": 538.34,
  "memory_after_mb": 920.11,
  "seed": 123456789
}
```

## HTTPS Setup

The DiffusionAPI supports HTTPS for secure connections. Here are the available options:

### Option 1: Self-signed Certificate (Recommended for Development)

1. **Generate SSL certificates:**

   ```bash
   ./generate_ssl_certs.sh
   ```

2. **Start the server with HTTPS:**

   ```bash
   ./init.sh
   ```

   The server will now be available at `https://localhost:7866`

### Option 2: Interactive HTTPS Setup

Use the interactive script to choose your preferred setup:

```bash
./init_https.sh
```

This script provides three options:

- **HTTP (default)**: Standard HTTP connection
- **HTTPS with self-signed certificate**: Automatically generates and uses self-signed certificates
- **HTTPS with custom certificates**: Use your own SSL certificates

### Option 3: Manual HTTPS Configuration

You can manually specify SSL certificates when starting uvicorn:

```bash
uvicorn diffusionapi.main:app --host 0.0.0.0 --port 7866 --ssl-keyfile=/path/to/key.pem --ssl-certfile=/path/to/cert.pem
```

### Important Notes for HTTPS

- **Self-signed certificates**: Your browser will show a security warning. This is normal for development. Click "Advanced" and "Proceed to localhost" to continue.
- **Production use**: For production environments, use certificates from a trusted Certificate Authority (CA).
- **Certificate files**: Keep your private key (`key.pem`) secure and never commit it to version control.

## Notes

1. **Progress Tracking:**

   - For normal generation: 0-100%
   - For hires generation:
     - Initial generation: 0-50%
     - Upscaling: 50-100%

2. **File Cleanup:**

   - Both progress and job files are automatically deleted after the final response
   - Generated images are saved in the `outputs` directory

3. **Supported Models:**

   - Standard Stable Diffusion models
   - SDXL models (with optional refiner)
   - Custom models in the models directory

4. **LoRA Support:**

   - LoRAs can be specified as objects with name and scale
   - Or as simple strings (using default scale of 1.0)
   - LoRA files must be in the `stable_diffusion/loras` directory

5. **Hires Fix:**
   - Supports both latent upscaling and external upscalers
   - External upscaler models must be in the `stable_diffusion/upscalers` directory
   - Latent upscaling is the default and recommended method
