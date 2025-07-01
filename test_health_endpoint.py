#!/usr/bin/env python3
"""
Test script to verify the health endpoint functionality.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
API_BASE = "http://localhost:7866"

def test_health_endpoint():
    """Test the health endpoint."""
    print("=== Testing Health Endpoint ===\n")
    
    try:
        # Test the health endpoint
        print("Sending GET request to /health...")
        response = requests.get(f"{API_BASE}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check successful!")
            print(f"Status: {data.get('status')}")
            print(f"Version: {data.get('version')}")
            print(f"Timestamp: {datetime.fromtimestamp(data.get('timestamp', 0))}")
            
            # System information
            system = data.get('system', {})
            print(f"\n📊 System Information:")
            print(f"  CPU Count: {system.get('cpu_count')}")
            print(f"  Memory Total: {system.get('memory_total_gb')} GB")
            print(f"  Memory Available: {system.get('memory_available_gb')} GB")
            print(f"  Memory Usage: {system.get('memory_percent')}%")
            print(f"  Disk Usage: {system.get('disk_usage_percent')}%")
            
            # GPU information
            gpu = data.get('gpu', {})
            print(f"\n🎮 GPU Information:")
            print(f"  Device: {gpu.get('device_name', 'Unknown')}")
            if gpu.get('cuda_available'):
                print(f"  CUDA Version: {gpu.get('cuda_version')}")
                print(f"  GPU Count: {gpu.get('gpu_count')}")
                print(f"  Memory Allocated: {gpu.get('memory_allocated_gb')} GB")
                print(f"  Memory Reserved: {gpu.get('memory_reserved_gb')} GB")
            elif gpu.get('mps_available'):
                print(f"  Apple Silicon GPU available")
            
            # Models and LoRAs
            models = data.get('models', {})
            loras = data.get('loras', {})
            print(f"\n🤖 Models & LoRAs:")
            print(f"  Models: {models.get('count')} found")
            print(f"  LoRAs: {loras.get('count')} found")
            
            # Queue status
            queue = data.get('queue', {})
            print(f"\n📋 Queue Status:")
            print(f"  Active Jobs: {queue.get('active_jobs')}")
            
            # Outputs
            outputs = data.get('outputs', {})
            print(f"\n💾 Outputs:")
            print(f"  Available: {outputs.get('available')}")
            print(f"  Free Space: {outputs.get('free_space_gb')} GB")
            
            # Warnings
            warnings = data.get('warnings', [])
            if warnings:
                print(f"\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print(f"\n✅ No warnings detected")
            
            return True
            
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {API_BASE}")
        print("Make sure the DiffusionAPI server is running with:")
        print("  python -m uvicorn diffusionapi.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_hello_endpoint():
    """Test the simple hello endpoint."""
    print("\n=== Testing Hello Endpoint ===\n")
    
    try:
        response = requests.get(f"{API_BASE}/hello")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Hello endpoint working!")
            print(f"Response: {data}")
            return True
        else:
            print(f"❌ Hello endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing hello endpoint: {e}")
        return False

if __name__ == "__main__":
    print("DiffusionAPI Health Check Test")
    print("=" * 50)
    print("This test verifies the health monitoring endpoints.")
    print()
    
    # Test hello endpoint
    hello_success = test_hello_endpoint()
    
    # Test health endpoint
    health_success = test_health_endpoint()
    
    print("\n" + "=" * 50)
    if hello_success and health_success:
        print("✅ All health checks passed!")
    else:
        print("❌ Some health checks failed!")
        sys.exit(1) 