#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-frequency noise demonstration script
Adds different levels of noise to a randomly selected test image
"""

import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def add_high_frequency_noise(image, intensity=0.1, noise_type='gaussian'):
    """
    Add high-frequency noise to an image
    
    Args:
        image: Input image
        intensity: Noise intensity (0-1)
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
    """
    noisy = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # Gaussian high-frequency noise
        noise = np.random.randn(*image.shape) * 255 * intensity
        noisy = noisy + noise
        
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        prob = intensity
        rnd = np.random.random(image.shape[:2])
        # Salt
        noisy[rnd < prob/2] = 255
        # Pepper
        noisy[rnd > 1 - prob/2] = 0
        
    elif noise_type == 'speckle':
        # Speckle noise (multiplicative)
        noise = np.random.randn(*image.shape) * intensity
        noisy = noisy + noisy * noise
        
    elif noise_type == 'periodic':
        # High-frequency periodic noise
        rows, cols = image.shape[:2]
        for i in range(rows):
            for j in range(cols):
                noisy[i, j] += intensity * 255 * np.sin(2 * np.pi * i * 10 / rows) * np.sin(2 * np.pi * j * 10 / cols)
    
    # Clip values to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def demonstrate_noise_effects(image_path, output_dir):
    """
    Demonstrate different noise levels and types on an image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define noise configurations
    noise_configs = [
        ('Original', image_rgb, None, 0),
        ('Gaussian 5%', 'gaussian', 0.05),
        ('Gaussian 10%', 'gaussian', 0.10),
        ('Gaussian 20%', 'gaussian', 0.20),
        ('Gaussian 30%', 'gaussian', 0.30),
        ('Salt & Pepper 5%', 'salt_pepper', 0.05),
        ('Salt & Pepper 10%', 'salt_pepper', 0.10),
        ('Speckle 10%', 'speckle', 0.10),
        ('Speckle 20%', 'speckle', 0.20),
        ('Periodic', 'periodic', 0.15),
        ('Mixed (Gauss+S&P)', 'mixed', 0.10),
        ('Heavy Mixed', 'mixed_heavy', 0.20),
    ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, config in enumerate(noise_configs):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        if idx == 0:
            # Original image
            title, img, _, _ = config
            ax.imshow(img)
        else:
            title, noise_type, intensity = config
            
            if noise_type == 'mixed':
                # Apply multiple noise types
                noisy = add_high_frequency_noise(image_rgb, intensity/2, 'gaussian')
                noisy = add_high_frequency_noise(noisy, intensity/2, 'salt_pepper')
            elif noise_type == 'mixed_heavy':
                # Heavy mixed noise
                noisy = add_high_frequency_noise(image_rgb, intensity, 'gaussian')
                noisy = add_high_frequency_noise(noisy, intensity/3, 'salt_pepper')
                noisy = add_high_frequency_noise(noisy, intensity/2, 'speckle')
            else:
                noisy = add_high_frequency_noise(image_rgb, intensity, noise_type)
            
            ax.imshow(noisy)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add main title
    fig.suptitle(f'High-Frequency Noise Effects Demonstration\nImage: {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold')
    
    # Save the figure
    output_path = os.path.join(output_dir, 'noise_effects_demonstration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved noise demonstration to: {output_path}")
    
    # Also create a detailed comparison of just Gaussian noise
    create_gaussian_detail(image_rgb, image_path, output_dir)
    
    plt.show()


def create_gaussian_detail(image_rgb, image_path, output_dir):
    """
    Create detailed comparison of Gaussian noise levels
    """
    intensities = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, intensity in enumerate(intensities):
        if intensity == 0:
            noisy = image_rgb
            title = 'Original'
        else:
            noisy = add_high_frequency_noise(image_rgb, intensity, 'gaussian')
            title = f'Gaussian {int(intensity*100)}%'
        
        axes[idx].imshow(noisy)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')
        
        # Add PSNR calculation for quality measurement
        if intensity > 0:
            mse = np.mean((image_rgb.astype(float) - noisy.astype(float)) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                axes[idx].text(0.5, -0.1, f'PSNR: {psnr:.1f}dB', 
                             transform=axes[idx].transAxes,
                             ha='center', fontsize=8, color='red')
    
    fig.suptitle('Gaussian Noise Intensity Comparison (with PSNR values)', 
                 fontsize=14, fontweight='bold')
    
    output_path = os.path.join(output_dir, 'gaussian_noise_detail.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Gaussian noise detail to: {output_path}")


def analyze_frequency_spectrum(image_path, output_dir):
    """
    Analyze and visualize frequency spectrum before and after adding noise
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    
    # Add noise
    noisy = add_high_frequency_noise(image, 0.15, 'gaussian')
    
    # Compute FFT
    f_original = np.fft.fft2(image)
    f_noisy = np.fft.fft2(noisy)
    
    # Shift zero frequency to center
    fshift_original = np.fft.fftshift(f_original)
    fshift_noisy = np.fft.fftshift(f_noisy)
    
    # Get magnitude spectrum (log scale for better visualization)
    magnitude_original = np.log(np.abs(fshift_original) + 1)
    magnitude_noisy = np.log(np.abs(fshift_noisy) + 1)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(magnitude_original, cmap='gray')
    axes[0, 1].set_title('Original Frequency Spectrum')
    axes[0, 1].axis('off')
    
    # Difference in frequency domain
    diff = magnitude_noisy - magnitude_original
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Frequency Difference (Noise)')
    axes[0, 2].axis('off')
    
    # Noisy
    axes[1, 0].imshow(noisy, cmap='gray')
    axes[1, 0].set_title('Noisy Image (15% Gaussian)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(magnitude_noisy, cmap='gray')
    axes[1, 1].set_title('Noisy Frequency Spectrum')
    axes[1, 1].axis('off')
    
    # Radial profile comparison
    ax = axes[1, 2]
    center = (image.shape[0]//2, image.shape[1]//2)
    
    # Calculate radial average
    def radial_profile(data, center):
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile
    
    profile_original = radial_profile(magnitude_original, center)
    profile_noisy = radial_profile(magnitude_noisy, center)
    
    ax.plot(profile_original[:100], label='Original', linewidth=2)
    ax.plot(profile_noisy[:100], label='Noisy', linewidth=2, alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')
    ax.set_title('Radial Frequency Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Frequency Domain Analysis of High-Frequency Noise', 
                 fontsize=14, fontweight='bold')
    
    output_path = os.path.join(output_dir, 'frequency_spectrum_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved frequency spectrum analysis to: {output_path}")


def main():
    """
    Main function to run the demonstration
    """
    # Create output directory
    output_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\tests\analyze\analyze_result\high_frequency_noise"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to test images directory
    test_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\annotated\test"
    
    # Get list of all PNG files
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    if not image_files:
        print("Error: No PNG files found in test directory")
        return
    
    # Randomly select an image
    selected_image = random.choice(image_files)
    image_path = os.path.join(test_dir, selected_image)
    
    print("="*60)
    print("High-Frequency Noise Demonstration")
    print("="*60)
    print(f"Selected image: {selected_image}")
    print(f"Full path: {image_path}")
    print(f"Output directory: {output_dir}")
    print("-"*60)
    
    # Run demonstrations
    print("\n1. Creating noise effects demonstration...")
    demonstrate_noise_effects(image_path, output_dir)
    
    print("\n2. Analyzing frequency spectrum...")
    analyze_frequency_spectrum(image_path, output_dir)
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print(f"Generated files in: {output_dir}")
    print("  - noise_effects_demonstration.png")
    print("  - gaussian_noise_detail.png")
    print("  - frequency_spectrum_analysis.png")
    print("="*60)


if __name__ == "__main__":
    main()