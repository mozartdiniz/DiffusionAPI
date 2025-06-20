#!/usr/bin/env python3
"""
Script to compare original.png and img2img_cfg3.0_result.png
to analyze why img2img is not preserving the original structure.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_and_resize_image(path, target_size=(512, 512)):
    """Load and resize image to target size."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

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
    psnr_val = psnr(img1, img2)
    ssim_val = ssim(gray1, gray2)
    
    # Calculate histogram differences
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return {
        'mse': mse,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'hist_correlation': hist_diff,
        'diff': diff,
        'diff_gray': diff_gray
    }

def analyze_images():
    """Main analysis function."""
    print("=== Image Comparison Analysis ===")
    
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
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Result image
    axes[0, 1].imshow(result)
    axes[0, 1].set_title('Generated Image')
    axes[0, 1].axis('off')
    
    # Difference image
    axes[0, 2].imshow(metrics['diff'])
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    
    # Grayscale difference
    axes[1, 0].imshow(metrics['diff_gray'], cmap='hot')
    axes[1, 0].set_title('Grayscale Difference (Hot)')
    axes[1, 0].axis('off')
    
    # Histogram comparison
    for i, color in enumerate(['red', 'green', 'blue']):
        hist1 = cv2.calcHist([original], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([result], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist1, color=color, alpha=0.7, label=f'Original {color}')
        axes[1, 1].plot(hist2, color=color, alpha=0.3, linestyle='--', label=f'Generated {color}')
    
    axes[1, 1].set_title('Color Histograms')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Metrics summary
    axes[1, 2].text(0.1, 0.8, f"SSIM: {metrics['ssim']:.4f}", fontsize=12)
    axes[1, 2].text(0.1, 0.6, f"PSNR: {metrics['psnr']:.2f} dB", fontsize=12)
    axes[1, 2].text(0.1, 0.4, f"MSE: {metrics['mse']:.2f}", fontsize=12)
    axes[1, 2].text(0.1, 0.2, f"Hist Corr: {metrics['hist_correlation']:.4f}", fontsize=12)
    axes[1, 2].set_title('Metrics Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_comparison_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved as: image_comparison_analysis.png")
    
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

if __name__ == "__main__":
    analyze_images() 