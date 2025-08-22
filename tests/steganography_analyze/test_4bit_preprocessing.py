"""
Test script to verify 4-bit preprocessing
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.preprocessor import TrainingPreprocessor
from src.preprocessing.config_loader import get_full_config

def test_preprocessing():
    """Test the preprocessing with 4-bit quantization"""
    
    # Load configuration
    config = get_full_config()
    
    # Initialize preprocessor with config
    preprocessor = TrainingPreprocessor(config)
    
    # Test image path
    test_image = "tests/analyze/data/site1.png"
    
    # Load original image for comparison
    original = cv2.imread(test_image)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Dummy labels for testing
    gap_center = (100, 100)
    slider_center = (200, 100)
    gap_angle = 0.0
    
    # Process with the modified preprocessor
    result = preprocessor.preprocess(
        image=test_image,
        gap_center=gap_center,
        slider_center=slider_center,
        gap_angle=gap_angle
    )
    
    # Extract the processed image
    processed_tensor = result['input']
    processed_gray = processed_tensor[0]  # First channel is the grayscale image
    
    # Denormalize for visualization
    processed_gray = (processed_gray * 255).astype(np.uint8)
    
    # Create comparison plot with dark background
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#2b2b2b')
    
    # Set dark background for all subplots
    for ax in axes:
        ax.set_facecolor('#404040')
    
    # Original
    axes[0].imshow(original_gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Grayscale', fontsize=14, fontweight='bold', color='white')
    axes[0].axis('off')
    
    # After 4-bit quantization and preprocessing
    axes[1].imshow(processed_gray, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('After Preprocessing\n(4-bit quantized + letterbox)', fontsize=14, fontweight='bold', color='white')
    axes[1].axis('off')
    
    # Difference
    # Show histogram of quantized values
    axes[2].hist(processed_gray.flatten(), bins=16, range=(0, 255), color='#4a90e2', alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[2].set_title('Histogram of Processed Image\n(Should show 16 discrete levels)', fontsize=14, fontweight='bold', color='white')
    axes[2].set_xlabel('Pixel Value', color='white', fontsize=12)
    axes[2].set_ylabel('Count', color='white', fontsize=12)
    axes[2].grid(True, alpha=0.2, color='white')
    axes[2].tick_params(colors='white')
    
    # Set white color for tick labels
    for label in axes[2].get_xticklabels() + axes[2].get_yticklabels():
        label.set_color('white')
    
    plt.suptitle('4-Bit Preprocessing Verification', fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    # Save the plot with dark background
    output_dir = Path("test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "4bit_preprocessing_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='#2b2b2b', edgecolor='none')
    print(f"Saved test result to: {output_file}")
    
    # Verify quantization levels
    unique_values = np.unique(processed_gray)
    print(f"\nUnique pixel values in processed image: {len(unique_values)}")
    print(f"Values: {unique_values[:20]}...")  # Show first 20 values
    
    # Check if all values are multiples of 16 (except padding)
    non_padding_mask = processed_gray != 1.0  # In normalized space, padding is 1.0 (255/255)
    non_padding_values = processed_gray[non_padding_mask] * 255  # Convert back to 0-255 range
    if len(non_padding_values) > 0:
        # Convert to int for modulo operation
        non_padding_values_int = non_padding_values.astype(np.uint8)
        is_quantized = np.all(non_padding_values_int % 16 == 0)
        print(f"All non-padding values are multiples of 16: {is_quantized}")
    
    # Print tensor shape and quantization info
    print(f"\nProcessed tensor shape: {processed_tensor.shape}")
    print(f"Expected shape: [2, 256, 512]")
    print(f"\nQuantization settings from config:")
    print(f"  Enabled: {preprocessor.quantization_enabled}")
    print(f"  Bits to keep: {preprocessor.bits_to_keep}")
    print(f"  Quantization mask: {preprocessor.quantization_mask:#010b} ({preprocessor.quantization_mask})")
    
    plt.show()

if __name__ == "__main__":
    test_preprocessing()