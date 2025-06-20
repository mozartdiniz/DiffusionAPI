// API Types for DiffusionAPI

export interface LoRA {
    name: string;
    scale: number;
}

export interface HiresConfig {
    enabled: boolean;
    scale?: number;
    upscaler?: string;
    steps?: number;
    denoising_strength?: number;
}

export interface Img2ImgRequest {
    prompt: string; // Main prompt
    negative_prompt?: string; // Optional negative prompt
    image: string; // Base64 encoded input image (required)
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
    // Img2Img specific parameters
    denoising_strength?: number; // How much to change the image (0.0-1.0, default: 0.75)
    image_guidance_scale?: number; // How much to push the generated image towards the initial image. Only used in SDXL models. (default: 1.8)
    resize_mode?: "just resize" | "crop and resize" | "resize and fill"; // How to handle input image resizing (default: "just resize")
    resize_to?: { width: number; height: number }; // Target dimensions for resizing
    resize_by?: { width: number; height: number } | number; // Scale factors for resizing (can be object or simple number)
    resized_by?: { width: number; height: number } | number; // Alternative name for resize_by (both are accepted)
}

export interface Img2ImgResponse {
    job_id: string; // Job identifier
    status: "queued"; // Initial status
    seed?: number; // The seed used for generation
}

export interface Txt2ImgRequest {
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
}

export interface Txt2ImgResponse {
    job_id: string; // Job identifier
    status: "queued"; // Initial status
    seed?: number; // The seed used for generation
}

export interface JobStatus {
    status: "success" | "error";
    job_id: string;
    progress: number;
    current_phase: string;
    state: "queued" | "loading" | "processing" | "done" | "error";
    error?: string;
    payload?: any; // Job payload (when done)
    image?: string; // Base64 encoded image (when done)
    output_path?: string;
    width?: number;
    height?: number;
    file_type?: string;
    jpeg_quality?: number;
    generation_time_sec?: number;
    memory_before_mb?: number;
    memory_after_mb?: number;
    seed?: number;
    // Img2Img specific fields
    denoising_strength?: number;
    resize_mode?: number;
    resize_to?: { width: number; height: number };
    resize_by?: { width: number; height: number };
}

export interface ModelInfo {
    name: string;
    label: string;
}

export interface ModelsResponse {
    status: "success" | "error";
    models: ModelInfo[];
    detail?: string;
}

export interface LoRAInfo {
    name: string;
}

export interface LoRAsResponse {
    status: "success" | "error";
    loras: LoRAInfo[];
    detail?: string;
}

export interface UpscalerInfo {
    name: string;
    type: "internal" | "external";
    description: string;
}

export interface UpscalersResponse {
    status: "success" | "error";
    upscalers: UpscalerInfo[];
    detail?: string;
}

export interface SamplingMethodInfo {
    name: string;
    aliases: string[];
    options: Record<string, string>;
}

export interface SamplingMethodsResponse {
    status: "success" | "error";
    samplers: SamplingMethodInfo[];
    detail?: string;
}

// Client class example
export class DiffusionAPIClient {
    private baseUrl: string;

    constructor(baseUrl: string = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }

    async img2img(params: Img2ImgRequest): Promise<Img2ImgResponse> {
        const response = await fetch(`${this.baseUrl}/img2img`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async txt2img(params: Txt2ImgRequest): Promise<Txt2ImgResponse> {
        const response = await fetch(`${this.baseUrl}/txt2img`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getJobStatus(jobId: string): Promise<JobStatus> {
        const response = await fetch(`${this.baseUrl}/queue/${jobId}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getModels(): Promise<ModelsResponse> {
        const response = await fetch(`${this.baseUrl}/models`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getLoRAs(): Promise<LoRAsResponse> {
        const response = await fetch(`${this.baseUrl}/loras`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getUpscalers(): Promise<UpscalersResponse> {
        const response = await fetch(`${this.baseUrl}/upscalers`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getSamplingMethods(): Promise<SamplingMethodsResponse> {
        const response = await fetch(`${this.baseUrl}/sampling_methods`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }
} 