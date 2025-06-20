#!/usr/bin/env python3
"""
Quick comparison between original and preservation test images.
"""

import cv2
import numpy as np
from PIL import Image

def quick_compare():
    """Quick comparison of images."""
    print("=== Quick Image Comparison ===")
    
    try:
        # Load images
        original = cv2.imread("original.png")
        result = cv2.imread("preservation_test_0.05.png")
        
        if original is None or result is None:
            print("❌ Could not load images")
            return
        
        # Resize to same size if needed
        if original.shape != result.shape:
            result = cv2.resize(result, (original.shape[1], original.shape[0]))
        
        print(f"✅ Original shape: {original.shape}")
        print(f"✅ Result shape: {result.shape}")
        
        # Calculate basic metrics
        diff = cv2.absdiff(original, result)
        mse = np.mean((original.astype(float) - result.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Calculate SSIM manually (simplified)
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(float)
        
        mu1 = np.mean(gray_orig)
        mu2 = np.mean(gray_result)
        sigma1_sq = np.var(gray_orig)
        sigma2_sq = np.var(gray_result)
        sigma12 = np.mean((gray_orig - mu1) * (gray_result - mu2))
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        print(f"\n=== Metrics ===")
        print(f"MSE: {mse:.2f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        
        # Check unchanged pixels
        unchanged_pixels = np.sum(diff == 0)
        total_pixels = diff.size
        unchanged_percentage = (unchanged_pixels / total_pixels) * 100
        
        print(f"Unchanged pixels: {unchanged_pixels:,} / {total_pixels:,} ({unchanged_percentage:.2f}%)")
        print(f"Average pixel difference: {np.mean(diff):.2f}")
        print(f"Max pixel difference: {np.max(diff)}")
        
        # Interpretation
        print(f"\n=== Interpretation ===")
        if ssim > 0.9:
            print("✅ EXCELLENT preservation (SSIM > 0.9)")
        elif ssim > 0.7:
            print("✅ GOOD preservation (SSIM > 0.7)")
        elif ssim > 0.5:
            print("⚠️  MODERATE preservation (SSIM > 0.5)")
        else:
            print("❌ POOR preservation (SSIM < 0.5)")
        
        if unchanged_percentage > 50:
            print("✅ HIGH percentage of unchanged pixels")
        elif unchanged_percentage > 20:
            print("⚠️  MODERATE percentage of unchanged pixels")
        else:
            print("❌ LOW percentage of unchanged pixels")
        
        # Save difference image
        cv2.imwrite("quick_diff.png", diff)
        print(f"\n📊 Difference image saved as: quick_diff.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_compare() 