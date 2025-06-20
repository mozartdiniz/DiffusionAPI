#!/usr/bin/env python3
"""
Simple script to compare original.png and img2img_cfg3.0_result.png
to analyze why img2img is not preserving the original structure.
"""

import cv2
import numpy as np
from PIL import Image

def load_and_resize_image(path, target_size=(512, 512)):
    """Load and resize image to target size."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

def calculate_ssim(img1, img2):
    """Calculate SSIM manually."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(float)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(float)
    
    # Calculate means
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    
    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

def calculate_differences(img1, img2):
    """Calculate various difference metrics between two images."""
    # Convert to grayscale for some metrics
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Calculate differences
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.absdiff(gray1, gray2)
    
    # Calculate metrics
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    ssim_val = calculate_ssim(img1, img2)
    
    # Calculate histogram differences
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_val,
        'hist_correlation': hist_diff,
        'diff': diff,
        'diff_gray': diff_gray
    }

def analyze_images():
    """Main analysis function."""
    print("=== Image Comparison Analysis ===")
    print("Comparing original.png with img2img_cfg3.0_result.png")
    print("Expected: With denoising_strength=0.3, images should be very similar")
    print()
    
    # Load images
    try:
        original = load_and_resize_image("original.png")
        result = load_and_resize_image("img2img_cfg3.0_result.png")
        print(f"✅ Loaded original.png: {original.shape}")
        print(f"✅ Loaded img2img_cfg3.0_result.png: {result.shape}")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return
    
    # Calculate differences
    metrics = calculate_differences(original, result)
    
    print("\n=== Difference Metrics ===")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.2f}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB")
    print(f"SSIM (Structural Similarity): {metrics['ssim']:.4f}")
    print(f"Histogram Correlation: {metrics['hist_correlation']:.4f}")
    
    # Interpret results
    print("\n=== Interpretation ===")
    if metrics['ssim'] > 0.9:
        print("✅ SSIM > 0.9: Images are very similar (good preservation)")
    elif metrics['ssim'] > 0.7:
        print("⚠️  SSIM 0.7-0.9: Images are moderately similar")
    else:
        print("❌ SSIM < 0.7: Images are quite different (poor preservation)")
    
    if metrics['psnr'] > 30:
        print("✅ PSNR > 30 dB: Good image quality preservation")
    elif metrics['psnr'] > 20:
        print("⚠️  PSNR 20-30 dB: Moderate image quality preservation")
    else:
        print("❌ PSNR < 20 dB: Poor image quality preservation")
    
    # Additional analysis
    print("\n=== Additional Analysis ===")
    
    # Check if images are the same size
    if original.shape == result.shape:
        print("✅ Images have the same dimensions")
    else:
        print(f"⚠️  Images have different dimensions: {original.shape} vs {result.shape}")
    
    # Check brightness differences
    orig_brightness = np.mean(original)
    result_brightness = np.mean(result)
    brightness_diff = abs(orig_brightness - result_brightness)
    print(f"Original brightness: {orig_brightness:.2f}")
    print(f"Generated brightness: {result_brightness:.2f}")
    print(f"Brightness difference: {brightness_diff:.2f}")
    
    # Check contrast differences
    orig_contrast = np.std(original)
    result_contrast = np.std(result)
    contrast_diff = abs(orig_contrast - result_contrast)
    print(f"Original contrast: {orig_contrast:.2f}")
    print(f"Generated contrast: {result_contrast:.2f}")
    print(f"Contrast difference: {contrast_diff:.2f}")
    
    # Save difference image
    diff_img = Image.fromarray(metrics['diff'])
    diff_img.save('difference_image.png')
    print(f"\n📊 Difference image saved as: difference_image.png")
    
    # Check pixel value ranges
    print(f"\n=== Pixel Value Analysis ===")
    print(f"Original - Min: {original.min()}, Max: {original.max()}, Mean: {original.mean():.2f}")
    print(f"Generated - Min: {result.min()}, Max: {result.max()}, Mean: {result.mean():.2f}")
    
    # Check if there are any completely unchanged pixels
    unchanged_pixels = np.sum(metrics['diff'] == 0)
    total_pixels = metrics['diff'].size
    unchanged_percentage = (unchanged_pixels / total_pixels) * 100
    print(f"Unchanged pixels: {unchanged_pixels:,} / {total_pixels:,} ({unchanged_percentage:.2f}%)")
    
    # Check maximum difference
    max_diff = np.max(metrics['diff'])
    print(f"Maximum pixel difference: {max_diff}")
    
    # Check average difference
    avg_diff = np.mean(metrics['diff'])
    print(f"Average pixel difference: {avg_diff:.2f}")

if __name__ == "__main__":
    analyze_images() 