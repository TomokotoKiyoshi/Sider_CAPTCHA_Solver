#!/usr/bin/env python3
"""
Slider CAPTCHA Recognition API Example Code
Demonstrates how to use CaptchaPredictor for inference and visualization

This example script provides comprehensive demonstrations of:
- Basic prediction usage
- Visualization capabilities
- Threshold tuning
- Numpy array input support
"""
import sys
from pathlib import Path

# Add project root to sys.path to enable imports from the project structure
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.inference.predict import CaptchaPredictor
import argparse
import cv2


def example_basic_prediction():
    """
    Basic prediction example demonstrating simple inference workflow.
    
    This example shows:
    - How to initialize the predictor with a trained model
    - How to perform inference on a single CAPTCHA image
    - How to extract and interpret prediction results
    - How to calculate the sliding distance from predictions
    """
    print("\n=== Basic Prediction Example ===")
    
    # Initialize predictor using the best performing model (epoch 11)
    # Model selection based on evaluation results showing optimal performance at epoch 11
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(
        model_path=str(model_path),
        device='auto',  # Automatically selects GPU if available, falls back to CPU
        hm_threshold=0.0  # Heatmap threshold set to 0 for maximum sensitivity
    )
    
    # Path to test image - using a real CAPTCHA from the test dataset
    # Image naming convention: Pic{ID}_Bgx{gap_x}Bgy{gap_y}_Sdx{slider_x}Sdy{slider_y}_{hash}.png
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    if test_image.exists():
        # Perform prediction - returns dictionary with detection results
        result = predictor.predict(str(test_image))
        
        # Display prediction results with formatting
        print(f"\nPrediction Results:")
        
        # X-coordinates for slider and gap (primary output for solving CAPTCHA)
        print(f"Slider X coordinate: {result['slider_x']:.2f}" if result['slider_x'] else "Slider not detected")
        print(f"Gap X coordinate: {result['gap_x']:.2f}" if result['gap_x'] else "Gap not detected")
        
        # Confidence scores indicate model's certainty (0.0 to 1.0)
        print(f"Slider confidence: {result['slider_confidence']:.4f}")
        print(f"Gap confidence: {result['gap_confidence']:.4f}")
        
        # Calculate sliding distance if both components detected
        # This is the key value needed to solve the CAPTCHA
        if result['slider_x'] and result['gap_x']:
            distance = result['gap_x'] - result['slider_x']
            print(f"Sliding distance: {distance:.2f} pixels")
    else:
        print(f"Test image not found: {test_image}")




def example_visualization():
    """
    Visualization example demonstrating how to generate visual outputs.
    
    This example shows:
    - How to generate prediction visualizations with detection markers
    - How to visualize internal model heatmaps
    - Options for displaying or saving visualization results
    """
    print("\n=== Visualization Example ===")
    
    # Initialize predictor with the best performing model
    # Visualization helps verify model predictions visually
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(
        model_path=str(model_path),
        device='auto',      # Auto-detect available compute device
        hm_threshold=0.0   # Use most sensitive threshold for visualization
    )
    
    # Select test image for visualization demonstration
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    if test_image.exists():
        # 1. Generate prediction visualization
        # This creates an annotated image showing detected positions
        print("\nGenerating prediction visualization...")
        predictor.visualize_prediction(
            str(test_image),
            save_path=None,  # Set to file path to save the visualization
            show=True        # Display visualization in a window
        )
        print("Prediction visualization generated (not saved)")
        # Visualization includes:
        # - Red circle: Detected gap position
        # - Blue circle: Detected slider position  
        # - Green dashed line: Sliding path
        # - Text annotations: Coordinates and confidence scores
        
        # 2. Generate heatmap visualization
        # This shows the internal model activations for debugging and analysis
        print("\nGenerating heatmap visualization...")
        predictor.visualize_heatmaps(
            str(test_image),
            save_path=None,  # Set to file path to save the heatmaps
            show=True        # Display heatmaps in a window
        )
        print("Heatmap visualization generated (not saved)")
        # Heatmap visualization includes 4 subplots:
        # - Original image
        # - Gap detection heatmap overlay
        # - Slider detection heatmap overlay
        # - Combined heatmap visualization
    else:
        print("Test image not found")


def example_custom_threshold():
    """
    Custom threshold example demonstrating the impact of heatmap thresholds.
    
    This example shows:
    - How different thresholds affect detection sensitivity
    - Trade-offs between detection rate and false positives
    - How to tune thresholds for specific use cases
    
    Threshold guidelines:
    - 0.0-0.05: Maximum sensitivity, may have false positives
    - 0.1 (default): Balanced performance for most cases
    - 0.2-0.3: Higher precision, may miss weak detections
    - 0.5+: Very conservative, only strongest detections
    """
    print("\n=== Custom Threshold Example ===")
    
    # Use the same test image for consistent comparison
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    if test_image.exists():
        # Test a range of thresholds to demonstrate their effects
        thresholds = [0.0, 0.1, 0.3, 0.5]
        
        for threshold in thresholds:
            print(f"\nHeatmap threshold = {threshold}")
            
            # Initialize predictor with different thresholds using the best model
            # Each threshold creates a new predictor instance for fair comparison
            model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
            predictor = CaptchaPredictor(
                model_path=str(model_path),
                device='auto',
                hm_threshold=threshold  # Key parameter being tested
            )
            
            # Perform prediction with current threshold
            result = predictor.predict(str(test_image))
            
            # Report detection results for this threshold
            if result['slider_x'] and result['gap_x']:
                print(f"  Detection successful - Slider: {result['slider_x']:.2f}, Gap: {result['gap_x']:.2f}")
                print(f"  Confidence scores - Slider: {result['slider_confidence']:.4f}, Gap: {result['gap_confidence']:.4f}")
            else:
                # Show which component failed detection
                print(f"  Detection failed - Slider: {'Not detected' if not result['slider_x'] else result['slider_x']}, "
                      f"Gap: {'Not detected' if not result['gap_x'] else result['gap_x']}")
    else:
        print("Test image not found")


def example_numpy_array_input():
    """
    Numpy array input example for integration with computer vision pipelines.
    
    This example shows:
    - How to use numpy arrays as input (useful for OpenCV integration)
    - Direct memory-based processing without file I/O
    - Compatibility with various image processing libraries
    
    Use cases:
    - Real-time video processing
    - Integration with existing CV pipelines
    - Batch processing from memory
    - Custom preprocessing workflows
    """
    print("\n=== Numpy Array Input Example ===")
    
    # Initialize predictor with the best performing model
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(
        model_path=str(model_path),
        device='auto',     # Auto-detect compute device
        hm_threshold=0.0  # Maximum sensitivity for demonstration
    )
    
    # Path to test image for loading into numpy array
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    if test_image.exists():
        # Use OpenCV to read image as numpy array
        # OpenCV loads images in BGR format by default
        image_array = cv2.imread(str(test_image))
        print(f"Loaded image as numpy array with shape: {image_array.shape}")
        print(f"Data type: {image_array.dtype}")
        
        # Directly use numpy array for prediction
        # The predictor automatically handles BGR to RGB conversion if needed
        result = predictor.predict(image_array)
        
        # Display results - identical to file-based prediction
        print("\nPrediction results using numpy array:")
        print(f"Slider X coordinate: {result['slider_x']:.2f}" if result['slider_x'] else "Slider not detected")
        print(f"Gap X coordinate: {result['gap_x']:.2f}" if result['gap_x'] else "Gap not detected")
        
        # Visualization also works with numpy arrays
        # This is useful for debugging preprocessing steps
        predictor.visualize_prediction(
            image_array,
            save_path=None,  # Set to file path to save
            show=True        # Display the visualization
        )
        print("\nVisualization from numpy input generated (not saved)")
        
        # Note: You can apply custom preprocessing before prediction
        # Example: image_array = your_preprocessing_function(image_array)
    else:
        print("Test image not found")


def main():
    """
    Main function orchestrating the example demonstrations.
    
    Provides command-line interface for running specific examples:
    - basic: Simple prediction workflow
    - visualization: Visual output generation
    - threshold: Threshold tuning demonstration
    - numpy: Numpy array input example
    - all: Run all examples in sequence
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='CAPTCHA Solver API Examples')
    parser.add_argument('--example', type=str, default='all',
                        choices=['basic', 'visualization', 'threshold', 'numpy', 'all'],
                        help='Which example to run')
    args = parser.parse_args()
    
    # Verify model file exists before running examples
    # Using checkpoint from epoch 11 based on evaluation results
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please run the training script first to generate the model file")
        print("Training command: python scripts/training/train.py")
        return
    print(f"Using model: {model_path}")
    
    # Execute requested examples based on command-line argument
    if args.example == 'basic' or args.example == 'all':
        example_basic_prediction()
    
    if args.example == 'visualization' or args.example == 'all':
        example_visualization()
    
    if args.example == 'threshold' or args.example == 'all':
        example_custom_threshold()
    
    if args.example == 'numpy' or args.example == 'all':
        example_numpy_array_input()
    
    # Display completion message and usage instructions
    print("\n=== All examples completed ===")
    print("\nUsage instructions:")
    print("1. Basic prediction: python api_example.py --example basic")
    print("2. Visualization: python api_example.py --example visualization")
    print("3. Threshold test: python api_example.py --example threshold")
    print("4. Numpy input: python api_example.py --example numpy")
    print("5. Run all: python api_example.py --example all")


if __name__ == '__main__':
    # Entry point when script is run directly
    main()