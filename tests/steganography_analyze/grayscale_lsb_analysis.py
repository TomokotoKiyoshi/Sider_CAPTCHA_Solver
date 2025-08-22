"""
LSB (Least Significant Bits) analysis - 4-bit version
Shows high 4 bits retention and low 4 bits visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_4bit_analysis(image_path):
    """
    Process image: keep high 4 bits and visualize low 4 bits
    """
    # Read image and convert to grayscale
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Keep only high 4 bits (mask with 0b11110000 = 240)
    high_4bits = gray & 0b11110000
    
    # Extract low 4 bits and shift to high 4 bits
    low_4bits = gray & 0b00001111  # Extract low 4 bits (values 0-15)
    low_to_high = low_4bits << 4   # Shift left by 4 bits (multiply by 16)
    
    return gray, high_4bits, low_to_high

def create_comparison_plot(image_path, gray, high_4bits, low_to_high, output_dir):
    """
    Create comparison plot showing original, high 4 bits, and low 4 bits visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original grayscale
    axes[0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Grayscale', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. High 4 bits only
    axes[1].imshow(high_4bits, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('High 4 Bits Only\n(Low 4 bits removed)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Low 4 bits shifted to high
    axes[2].imshow(low_to_high, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Low 4 Bits → High 4 Bits\n(Hidden detail visualization)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add main title
    fig.suptitle(f'4-Bit Analysis: {image_path.name}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save comparison plot
    plot_path = output_dir / f"{image_path.stem}_4bit_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_path}")
    plt.close()

def main():
    # Setup paths
    data_dir = Path("tests/analyze/data")
    output_dir = Path("tests/analyze/analyze_result/grayscale_4bit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process both images
    images = ["site1.png", "synthesis.png"]
    
    for img_name in images:
        img_path = data_dir / img_name
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping...")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {img_name}")
        print('='*50)
        
        # Process image
        gray, high_4bits, low_to_high = process_4bit_analysis(img_path)
        
        if gray is not None:
            # Create comparison plot
            create_comparison_plot(img_path, gray, high_4bits, low_to_high, output_dir)
    
    print(f"\n✅ All processing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()