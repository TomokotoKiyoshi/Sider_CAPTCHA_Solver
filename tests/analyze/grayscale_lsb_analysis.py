"""
Grayscale conversion and LSB (Least Significant Bits) analysis
Converts images to grayscale and visualizes the lowest 3 bits as highest 3 bits
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_grayscale_and_lsb(image_path, output_dir):
    """
    Process image: convert to grayscale and extract LSB information
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return None, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save grayscale image
    gray_output_path = output_dir / f"{image_path.stem}_grayscale.png"
    cv2.imwrite(str(gray_output_path), gray)
    print(f"Saved grayscale: {gray_output_path}")
    
    # Extract lowest 3 bits and shift to highest 3 bits
    # Lowest 3 bits: binary mask 0b00000111 = 7
    lsb_3bits = gray & 0b00000111  # Extract lowest 3 bits (values 0-7)
    
    # Shift to highest 3 bits (multiply by 32 to shift left by 5 positions)
    # This maps 0-7 to 0, 32, 64, 96, 128, 160, 192, 224
    lsb_to_msb = lsb_3bits << 5  # Shift left by 5 bits (equivalent to * 32)
    
    # Save LSB visualization
    lsb_output_path = output_dir / f"{image_path.stem}_lsb3_to_msb.png"
    cv2.imwrite(str(lsb_output_path), lsb_to_msb)
    print(f"Saved LSB visualization: {lsb_output_path}")
    
    # Remove lowest 3 bits (set them to 0)
    # This is done by masking with 0b11111000 = 248
    no_lsb = gray & 0b11111000  # Keep only the highest 5 bits
    
    # Save image without LSB
    no_lsb_output_path = output_dir / f"{image_path.stem}_no_lsb.png"
    cv2.imwrite(str(no_lsb_output_path), no_lsb)
    print(f"Saved no-LSB image: {no_lsb_output_path}")
    
    return gray, lsb_to_msb, no_lsb

def create_comparison_plot(image_path, gray, lsb_to_msb, no_lsb, output_dir):
    """
    Create comparison plot showing original, grayscale, and LSB visualization
    """
    # Read original image for display
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots (now 2x3 for 6 images)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Grayscale image
    axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Grayscale Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Image with LSB removed
    axes[0, 2].imshow(no_lsb, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('LSB Removed (Lowest 3 Bits = 0)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. LSB (lowest 3 bits) shown as MSB
    axes[1, 0].imshow(lsb_to_msb, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Lowest 3 Bits → Highest 3 Bits\n(Hidden Information Visualization)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Difference between original and no-LSB
    diff = np.abs(gray.astype(int) - no_lsb.astype(int)).astype(np.uint8)
    diff_enhanced = diff * 32  # Enhance visibility
    axes[1, 1].imshow(diff_enhanced, cmap='viridis', vmin=0, vmax=255)
    axes[1, 1].set_title('Difference Map\n(Original - No LSB)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. LSB Activity heatmap
    lsb_original = gray & 0b00000111
    lsb_heatmap = np.zeros_like(gray, dtype=np.uint8)
    lsb_heatmap[lsb_original > 0] = 255
    
    axes[1, 2].imshow(lsb_heatmap, cmap='hot', vmin=0, vmax=255)
    axes[1, 2].set_title('LSB Activity Map\n(Red = Contains LSB Data)', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add main title
    fig.suptitle(f'LSB Analysis: {image_path.name}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save comparison plot
    plot_path = output_dir / f"{image_path.stem}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_path}")
    plt.close()

def analyze_lsb_statistics(gray):
    """
    Analyze statistics of LSB values
    """
    lsb_3bits = gray & 0b00000111
    
    stats = {
        'lsb_0': np.sum(lsb_3bits == 0),
        'lsb_1': np.sum(lsb_3bits == 1),
        'lsb_2': np.sum(lsb_3bits == 2),
        'lsb_3': np.sum(lsb_3bits == 3),
        'lsb_4': np.sum(lsb_3bits == 4),
        'lsb_5': np.sum(lsb_3bits == 5),
        'lsb_6': np.sum(lsb_3bits == 6),
        'lsb_7': np.sum(lsb_3bits == 7),
        'total_pixels': gray.size,
        'non_zero_lsb': np.sum(lsb_3bits > 0)
    }
    
    print("\nLSB Statistics:")
    print("-" * 40)
    for i in range(8):
        percentage = (stats[f'lsb_{i}'] / stats['total_pixels']) * 100
        print(f"LSB value {i}: {stats[f'lsb_{i}']:8d} pixels ({percentage:5.2f}%)")
    print("-" * 40)
    print(f"Total pixels: {stats['total_pixels']}")
    print(f"Non-zero LSB: {stats['non_zero_lsb']} ({stats['non_zero_lsb']/stats['total_pixels']*100:.2f}%)")
    
    return stats

def main():
    # Setup paths
    data_dir = Path("tests/analyze/data")
    output_dir = Path("tests/analyze/analyze_result/grayscale_lsb")
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
        gray, lsb_to_msb, no_lsb = process_grayscale_and_lsb(img_path, output_dir)
        
        if gray is not None:
            # Analyze LSB statistics
            stats = analyze_lsb_statistics(gray)
            
            # Create comparison plot
            create_comparison_plot(img_path, gray, lsb_to_msb, no_lsb, output_dir)
    
    print(f"\n✅ All processing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()