#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image histogram analysis
Counts pixel quantity for each grayscale value (0-255) and plots distribution
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_histogram(image_path):
    """
    Analyze histogram of an image
    
    Args:
        image_path: Path to image file
    """
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Flatten to 1D array
    hist = hist.flatten()
    
    return hist, img


def plot_comparison(hist1, hist2, name1, name2):
    """
    Plot histogram comparison
    
    Args:
        hist1: First histogram data
        hist2: Second histogram data
        name1: Name of first image
        name2: Name of second image
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot first histogram
    ax1.bar(range(256), hist1, width=1.0, color='blue', edgecolor='none')
    ax1.set_xlabel('Grayscale Value')
    ax1.set_ylabel('Pixel Count')
    ax1.set_title(f'Histogram - {name1}')
    ax1.set_xlim([0, 255])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot second histogram
    ax2.bar(range(256), hist2, width=1.0, color='green', edgecolor='none')
    ax2.set_xlabel('Grayscale Value')
    ax2.set_ylabel('Pixel Count')
    ax2.set_title(f'Histogram - {name2}')
    ax2.set_xlim([0, 255])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = r"D:\Hacker\Sider_CAPTCHA_Solver\tests\analyze\analyze_result\histogram"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'histogram_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    plt.show()


def main():
    """
    Main function
    """
    # Analyze two images
    image1_path = "tests/analyze/data/site1.png"
    image2_path = "tests/analyze/data/synthesis.png"
    
    print("Comparing histograms:")
    print(f"Image 1: site1.png")
    print(f"Image 2: synthesis.png")
    
    # Analyze first image
    result1 = analyze_histogram(image1_path)
    # Analyze second image
    result2 = analyze_histogram(image2_path)
    
    if result1 is not None and result2 is not None:
        hist1, img1 = result1
        hist2, img2 = result2
        
        # Plot comparison
        plot_comparison(hist1, hist2, "site1.png", "synthesis.png")
        
        # Print statistics for both images
        print(f"\nHistogram Statistics - site1.png:")
        print(f"Min pixel value: {np.min(np.where(hist1 > 0))}")
        print(f"Max pixel value: {np.max(np.where(hist1 > 0))}")
        print(f"Most frequent value: {np.argmax(hist1)} (count: {int(hist1[np.argmax(hist1)])})")
        print(f"Total pixels: {int(np.sum(hist1))}")
        
        print(f"\nHistogram Statistics - synthesis.png:")
        print(f"Min pixel value: {np.min(np.where(hist2 > 0))}")
        print(f"Max pixel value: {np.max(np.where(hist2 > 0))}")
        print(f"Most frequent value: {np.argmax(hist2)} (count: {int(hist2[np.argmax(hist2)])})")
        print(f"Total pixels: {int(np.sum(hist2))}")


if __name__ == "__main__":
    main()