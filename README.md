# Diffusion API

A FastAPI-based implementation for Stable Diffusion image generation, supporting both standard and SDXL models, with hires fix and LoRA support. This is a simplified API focused on core image generation functionality.

## API Documentation

### Types

```typescript
// Common Types
type LoRA = {
    name: string;      // LoRA model name
    scale: number;     // LoRA scale (default: 1.0)
}

type ModelInfo = {
    name: string;      // Model name
    type: "standard" | "sdxl";  // Model type
    path: string;      // Path to model file
}

type UpscalerInfo = {
    name: string;      // Upscaler name
    path: string;      // Path to upscaler file
}

type HiresConfig = {
    enabled: boolean;              // Whether hires fix is enabled
    scale: number;                 // Upscaling factor (default: 2.0)
    upscaler: string;             // Upscaler to use ("Latent" or external upscaler name)
    steps: number;                 // Number of steps for upscaling (default: 20)
    denoising_strength: number;    // Denoising strength for upscaling (default: 0.4)
}

// Request Types
type Txt2ImgRequest = {
    prompt: string;                // Main prompt
    negative_prompt?: string;      // Optional negative prompt
    steps?: number;                // Number of inference steps (default: 30)
    cfg_scale?: number;           // Guidance scale (default: 7.0)
    width?: number;               // Image width (default: 512)
    height?: number;              // Image height (default: 512)
    model?: string;               // Model name (default: "stable-diffusion-v1-5")
    sampler_name?: string;        // Sampler name (default: "DPM++ 2M Karras")
    scheduler_type?: string;      // Scheduler type (default: "karras")
    fileType?: "jpg" | "png";     // Output file type (default: "jpg")
    jpeg_quality?: number;        // JPEG quality (default: 85)
    loras?: (LoRA | string)[];    // Array of LoRAs (can be objects or just names)
    hires?: HiresConfig;          // Hires fix configuration
    refiner_checkpoint?: string;  // Optional SDXL refiner model
    refiner_switch_at?: number;   // When to switch to refiner (default: 0.8)
}

// Response Types
type QueueStatusResponse = {
    status: "success" | "error";   // Response status
    job_id: string;               // Job identifier
    progress: number;             // Progress percentage (0-100)
    current_phase: string;        // Current phase ("generation", "initial generation", "upscaling", "done", etc.)
    state: string;                // Current state ("queued", "processing", "done", "error")
    error?: string;               // Error message if any
    payload?: Txt2ImgRequest;     // Original request payload (only when complete)
    image?: string;               // Base64 encoded image (only when complete)
    output_path?: string;         // Path to saved image (only when complete)
    width?: number;               // Final image width (only when complete)
    height?: number;              // Final image height (only when complete)
    file_type?: "jpg" | "png";    // Output file type (only when complete)
    jpeg_quality?: number;        // JPEG quality used (only when complete)
    generation_time_sec?: number; // Total generation time (only when complete)
    memory_before_mb?: number;    // Memory usage before generation (only when complete)
    memory_after_mb?: number;     // Memory usage after generation (only when complete)
}

type Txt2ImgResponse = {
    job_id: string;               // Job identifier
    status: "queued";             // Initial status
}

// Response Types for List Endpoints
type ModelsResponse = {
    status: "success" | "error";
    models: ModelInfo[];
    error?: string;
}

type LorasResponse = {
    status: "success" | "error";
    loras: LoRA[];
    error?: string;
}

type UpscalersResponse = {
    status: "success" | "error";
    upscalers: UpscalerInfo[];
    error?: string;
}
```

### Endpoints

#### `GET /hello`
Simple health check endpoint.

**Response:**
```typescript
type HelloResponse = {
    ok: boolean;
}
```

#### `GET /models`
List all available models.

**Response:** `ModelsResponse`

**Example Response:**
```json
{
    "status": "success",
    "models": [
        {
            "name": "stable-diffusion-v1-5",
            "type": "standard",
            "path": "stable_diffusion/models/stable-diffusion-v1-5.safetensors"
        },
        {
            "name": "stable-diffusion-xl-base-1.0",
            "type": "sdxl",
            "path": "stable_diffusion/models/stable-diffusion-xl-base-1.0.safetensors"
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
    "model": "stable-diffusion-v1-5",
    "sampler_name": "DPM++ 2M Karras",
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
    }
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
        "model": "stable-diffusion-v1-5",
        "sampler_name": "DPM++ 2M Karras",
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
        }
    },
    "image": "base64...",
    "output_path": "outputs/242d6f93-ea2d-4d5e-b820-0b61468558be.jpg",
    "width": 1024,
    "height": 1024,
    "file_type": "jpg",
    "jpeg_quality": 85,
    "generation_time_sec": 58.36,
    "memory_before_mb": 538.34,
    "memory_after_mb": 920.11
}
```

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
