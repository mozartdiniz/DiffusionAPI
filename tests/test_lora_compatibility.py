#!/usr/bin/env python3
"""
Comprehensive LoRA Compatibility Test Script

This script automatically tests all models against all LoRAs and generates a detailed compatibility report.
It will test every possible combination and save the results to a file.

Usage:
    python test_lora_compatibility.py
    python test_lora_compatibility.py --output compatibility_report.txt
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

# Add the current directory to the path so we can import from diffusionapi
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import after adding to path
try:
    from diffusionapi.generate import load_loras, get_lora_compatibility_info, resolve_model_path, get_model_labels
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the DiffusionAPI root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_models() -> List[Dict[str, str]]:
    """Find all available models."""
    models = []
    
    # Get models from the model labels system
    try:
        model_labels = get_model_labels()
        for model_info in model_labels:
            models.append({
                "name": model_info["name"],
                "label": model_info["label"],
                "path": resolve_model_path(model_info["name"])
            })
    except Exception as e:
        logger.warning(f"Could not get model labels: {e}")
    
    # Check the stable_diffusion/models directory
    models_dir = Path("stable_diffusion/models")
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Check if it contains model files
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                if model_files:
                    models.append({
                        "name": model_dir.name,
                        "label": model_dir.name,
                        "path": str(model_dir)
                    })
    
    # Also check the models directory in root
    root_models_dir = Path("models")
    if root_models_dir.exists():
        for model_dir in root_models_dir.iterdir():
            if model_dir.is_dir():
                # Check if it contains model files
                model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                if model_files:
                    models.append({
                        "name": model_dir.name,
                        "label": model_dir.name,
                        "path": str(model_dir)
                    })
    
    return models

def find_lora_files() -> List[Dict[str, str]]:
    """Find all LoRA files in the loras directory."""
    loras = []
    
    # Check the stable_diffusion/loras directory first
    loras_dir = Path("stable_diffusion/loras")
    if loras_dir.exists():
        # Common LoRA file extensions
        extensions = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        
        for ext in extensions:
            for lora_file in loras_dir.glob(ext):
                loras.append({
                    "name": lora_file.stem,
                    "path": str(lora_file),
                    "extension": lora_file.suffix
                })
    
    # Also check the root loras directory if it exists
    root_loras_dir = Path("loras")
    if root_loras_dir.exists():
        extensions = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
        
        for ext in extensions:
            for lora_file in root_loras_dir.glob(ext):
                loras.append({
                    "name": lora_file.stem,
                    "path": str(lora_file),
                    "extension": lora_file.suffix
                })
    
    if not loras:
        logger.warning(f"No LoRA files found in stable_diffusion/loras or loras directories")
    
    return sorted(loras, key=lambda x: x["name"])

def load_pipeline(model_info: Dict[str, str], device: str = "cuda") -> Any:
    """Load the appropriate pipeline based on model type."""
    model_path = model_info["path"]
    model_name = model_info["name"]
    
    # Check if it's an SDXL model
    if "sdxl" in model_name.lower() or "xl" in model_name.lower():
        logger.info(f"Loading SDXL pipeline for model: {model_name}")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    else:
        logger.info(f"Loading standard SD pipeline for model: {model_name}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    
    pipeline = pipeline.to(device)
    return pipeline

def test_lora_compatibility(model_info: Dict[str, str], lora_info: Dict[str, str], device: str = "cuda") -> Dict[str, Any]:
    """Test if a specific LoRA is compatible with a model."""
    result = {
        "model_name": model_info["name"],
        "model_label": model_info["label"],
        "lora_name": lora_info["name"],
        "lora_path": lora_info["path"],
        "compatible": False,
        "error": None,
        "compatibility_info": None,
        "test_time": None
    }
    
    start_time = time.time()
    
    try:
        # Load the pipeline
        pipeline = load_pipeline(model_info, device)
        
        # Create a test LoRA configuration
        test_lora = {
            "name": lora_info["name"],
            "path": lora_info["path"],
            "scale": 1.0
        }
        
        # Try to load the LoRA
        load_loras(pipeline, [test_lora], model_info["name"])
        
        # If we get here, the LoRA loaded successfully
        result["compatible"] = True
        result["compatibility_info"] = "LoRA loaded successfully"
        
    except Exception as e:
        result["error"] = str(e)
        result["compatibility_info"] = get_lora_compatibility_info(lora_info["name"], model_info["name"])
    
    result["test_time"] = time.time() - start_time
    
    return result

def generate_compatibility_report(results: List[Dict[str, Any]], output_file: str) -> None:
    """Generate a comprehensive compatibility report."""
    # Group results by model
    model_results = {}
    for result in results:
        model_name = result["model_name"]
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    # Calculate statistics
    total_tests = len(results)
    compatible_tests = sum(1 for r in results if r["compatible"])
    incompatible_tests = total_tests - compatible_tests
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE LoRA COMPATIBILITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Tests: {total_tests}")
    report_lines.append(f"Compatible: {compatible_tests}")
    report_lines.append(f"Incompatible: {incompatible_tests}")
    report_lines.append(f"Success Rate: {(compatible_tests/total_tests*100):.1f}%")
    report_lines.append("")
    
    # Summary by model
    report_lines.append("SUMMARY BY MODEL:")
    report_lines.append("-" * 40)
    for model_name, model_tests in model_results.items():
        model_compatible = sum(1 for r in model_tests if r["compatible"])
        model_total = len(model_tests)
        success_rate = (model_compatible / model_total * 100) if model_total > 0 else 0
        report_lines.append(f"{model_name}: {model_compatible}/{model_total} ({success_rate:.1f}%)")
    report_lines.append("")
    
    # Detailed results by model
    for model_name, model_tests in sorted(model_results.items()):
        report_lines.append(f"MODEL: {model_name}")
        report_lines.append("=" * 60)
        
        # Compatible LoRAs
        compatible_loras = [r for r in model_tests if r["compatible"]]
        if compatible_loras:
            report_lines.append("✓ COMPATIBLE LoRAs:")
            for result in compatible_loras:
                report_lines.append(f"  • {result['lora_name']}")
        else:
            report_lines.append("✓ No compatible LoRAs found")
        
        report_lines.append("")
        
        # Incompatible LoRAs
        incompatible_loras = [r for r in model_tests if not r["compatible"]]
        if incompatible_loras:
            report_lines.append("✗ INCOMPATIBLE LoRAs:")
            for result in incompatible_loras:
                report_lines.append(f"  • {result['lora_name']}")
                if result["compatibility_info"]:
                    report_lines.append(f"    Reason: {result['compatibility_info']}")
                if result["error"]:
                    error_preview = result["error"][:100] + "..." if len(result["error"]) > 100 else result["error"]
                    report_lines.append(f"    Error: {error_preview}")
        else:
            report_lines.append("✗ No incompatible LoRAs found")
        
        report_lines.append("")
        report_lines.append("")
    
    # Write report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Compatibility report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test all LoRAs against all models for compatibility")
    parser.add_argument("--output", default="lora_compatibility_report.txt", 
                       help="Output file for the compatibility report")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--skip-existing", action="store_true", 
                       help="Skip models that already have results")
    
    args = parser.parse_args()
    
    logger.info("Starting comprehensive LoRA compatibility testing...")
    
    # Find all models and LoRAs
    models = find_models()
    loras = find_lora_files()
    
    if not models:
        logger.error("No models found!")
        sys.exit(1)
    
    if not loras:
        logger.error("No LoRAs found!")
        sys.exit(1)
    
    logger.info(f"Found {len(models)} models and {len(loras)} LoRAs")
    logger.info(f"Total combinations to test: {len(models) * len(loras)}")
    
    # Test all combinations
    results = []
    total_combinations = len(models) * len(loras)
    current_combination = 0
    
    for model_info in models:
        logger.info(f"Testing model: {model_info['name']} ({model_info['label']})")
        
        for lora_info in loras:
            current_combination += 1
            logger.info(f"  [{current_combination}/{total_combinations}] Testing LoRA: {lora_info['name']}")
            
            try:
                result = test_lora_compatibility(model_info, lora_info, args.device)
                results.append(result)
                
                if result["compatible"]:
                    logger.info(f"    ✓ COMPATIBLE")
                else:
                    logger.warning(f"    ✗ INCOMPATIBLE")
                    logger.warning(f"       {result['compatibility_info']}")
                
            except Exception as e:
                logger.error(f"    ✗ ERROR: {e}")
                results.append({
                    "model_name": model_info["name"],
                    "model_label": model_info["label"],
                    "lora_name": lora_info["name"],
                    "lora_path": lora_info["path"],
                    "compatible": False,
                    "error": str(e),
                    "compatibility_info": "Test failed due to error",
                    "test_time": 0
                })
        
        logger.info(f"Completed testing for model: {model_info['name']}")
        logger.info("-" * 60)
    
    # Generate report
    generate_compatibility_report(results, args.output)
    
    # Print summary
    compatible_count = sum(1 for r in results if r["compatible"])
    total_tests = len(results)
    
    logger.info("TESTING COMPLETED!")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Compatible combinations: {compatible_count}")
    logger.info(f"Success rate: {(compatible_count/total_tests*100):.1f}%")
    logger.info(f"Report saved to: {args.output}")

if __name__ == "__main__":
    main() 