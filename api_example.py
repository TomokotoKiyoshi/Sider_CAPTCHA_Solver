#!/usr/bin/env python3
"""
Slider CAPTCHA Solver API Example - Spyder Edition
Demonstrates all core features of CaptchaPredictor
"""
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import predictor
from src.models.predictor import CaptchaPredictor


def demo_basic_prediction():
    """Basic prediction example"""
    print("\n=== Basic Prediction ===")
    
    # Initialize predictor
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(
        model_path=str(model_path),
        device='auto'  # Auto-select GPU/CPU
    )
    
    # Test image
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    # Predict
    result = predictor.predict(str(test_image))
    
    # Display results
    print(f"Slider position: ({result['slider_x']:.2f}, {result['slider_y']:.2f})")
    print(f"Gap position: ({result['gap_x']:.2f}, {result['gap_y']:.2f})")
    print(f"Sliding distance: {result['gap_x'] - result['slider_x']:.2f} pixels")
    print(f"Confidence - Slider: {result['slider_confidence']:.3f}, Gap: {result['gap_confidence']:.3f}")


def demo_visualization():
    """Visualization example"""
    print("\n=== Visualization ===")
    
    # Initialize predictor
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(model_path=str(model_path), device='auto')
    
    # Test image
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    
    # 1. Prediction visualization
    print("Generating prediction visualization...")
    predictor.visualize_prediction(
        str(test_image),
        save_path='prediction_result.png',  # Save to file
        show=True  # Display window
    )
    
    # 2. Heatmap visualization
    print("Generating heatmap visualization...")
    predictor.visualize_heatmaps(
        str(test_image),
        save_path='heatmap_result.png',  # Save to file
        show=True  # Display window
    )


def demo_threshold_effect():
    """Threshold effect demonstration"""
    print("\n=== Threshold Effect ===")
    
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    
    # Test different thresholds
    thresholds = [0.0, 0.1, 0.3, 0.5]
    
    for threshold in thresholds:
        predictor = CaptchaPredictor(
            model_path=str(model_path),
            device='auto',
            hm_threshold=threshold
        )
        
        result = predictor.predict(str(test_image))
        
        print(f"\nThreshold = {threshold}:")
        if result['slider_x'] and result['gap_x']:
            distance = result['gap_x'] - result['slider_x']
            print(f"  Detection successful - Distance: {distance:.2f}px")
            print(f"  Confidence - Slider: {result['slider_confidence']:.3f}, Gap: {result['gap_confidence']:.3f}")
        else:
            print(f"  Detection failed - Slider: {'Not detected' if not result['slider_x'] else 'Detected'}, "
                  f"Gap: {'Not detected' if not result['gap_x'] else 'Detected'}")


def demo_numpy_input():
    """NumPy array input example"""
    print("\n=== NumPy Array Input ===")
    
    # Initialize predictor
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    predictor = CaptchaPredictor(model_path=str(model_path), device='auto')
    
    # Read image as numpy array
    test_image = project_root / 'data' / 'test' / 'Pic0001_Bgx100Bgy90_Sdx31Sdy90_841ddd70.png'
    image_array = cv2.imread(str(test_image))
    
    print(f"Image array shape: {image_array.shape}")
    print(f"Data type: {image_array.dtype}")
    
    # Predict with numpy array
    result = predictor.predict(image_array)
    
    print(f"\nPrediction results with NumPy array:")
    print(f"Sliding distance: {result['gap_x'] - result['slider_x']:.2f} pixels")
    
    # Visualization also supports numpy arrays
    predictor.visualize_prediction(
        image_array,
        save_path='numpy_input_result.png',
        show=True
    )


def run_all_demos():
    """Run all demonstrations"""
    print("=" * 60)
    print("Slider CAPTCHA Solver API Demonstration")
    print("=" * 60)
    
    # Check model file
    model_path = project_root / 'src' / 'checkpoints' / 'checkpoint_epoch_0011.pth'
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please run training script first")
        return
    
    # Run all demos
    demo_basic_prediction()
    demo_visualization()
    demo_threshold_effect()
    demo_numpy_input()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)


# ============================================================================
# Spyder Usage Guide
# ============================================================================
# Run specific demos in Spyder:
#   demo_basic_prediction()    # Basic prediction
#   demo_visualization()       # Visualization
#   demo_threshold_effect()    # Threshold effects
#   demo_numpy_input()        # NumPy input
#   run_all_demos()           # Run all demos
# ============================================================================

if __name__ == '__main__':
    # Default: run all demos
    run_all_demos()