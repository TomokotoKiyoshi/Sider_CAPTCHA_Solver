# Industrial-Grade Slider CAPTCHA Recognition System

<div align="center">

[English](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README.md) | [简体中文](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/README_zh.md)

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-precision slider CAPTCHA recognition solution based on deep learning, utilizing an improved CenterNet architecture to achieve 80% accuracy on real CAPTCHA datasets.

</div>

## 📋 Project Overview

This project is an industrial-grade slider CAPTCHA recognition system that overcomes the accuracy bottleneck of traditional template matching algorithms through deep learning methods. The system is trained on **over 300,000** synthetic CAPTCHA images, employing a lightweight CNN architecture that ensures high precision while maintaining real-time inference capabilities.

### 🎯 Core Features

- **High-Precision Recognition**: 80% accuracy with 7px error tolerance, 73% with 5px on real CAPTCHAs
- **Real-Time Inference**: GPU inference 1.30ms (RTX 5090), CPU inference 5.21ms (AMD Ryzen 9 9950X), supporting real-time applications
- **Lightweight Architecture**: Only 3.5M parameters, model file approximately 36MB
- **Industrial-Grade Design**: Complete data generation, training, and evaluation pipeline
- **Sub-pixel Precision**: Achieves sub-pixel level localization using CenterNet offset mechanism

### 🖼️ Recognition Performance Demo

#### Real CAPTCHA Dataset Recognition Results

![Real Dataset Recognition Results](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/best_model_evaluation/real_captchas/visualizations/sample_0031.png?raw=true)

*Figure: Recognition results on real website CAPTCHAs, with red circles marking gap positions and blue circles marking slider positions*

#### Test Dataset Recognition Results

![Test Dataset Recognition Results](https://github.com/TomokotoKiyoshi/Sider_CAPTCHA_Solver/blob/main/results/best_model_evaluation/test_dataset/visualizations/sample_0014.png?raw=true)

*Figure: Recognition results on synthetic test set, demonstrating the model's adaptability to different shapes and lighting conditions*

## 🚀 Quick Start

### Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Installation

#### Install via pip

```bash
pip install Sider_CAPTCHA_Solver
```

### Basic Usage

After pip installation, you can directly import and use:

#### 1. Basic Prediction - Get Sliding Distance

```python
from sider_captcha_solver import CaptchaPredictor

# Initialize predictor
predictor = CaptchaPredictor(
    model_path='best',  # Use built-in best model, or specify custom model path
    device='auto'       # Auto-select GPU/CPU
)

# Predict single image
result = predictor.predict('path/to/captcha.png')

# Get sliding distance
if result['slider_x'] and result['gap_x']:
    sliding_distance = result['gap_x'] - result['slider_x']
    print(f"Sliding distance: {sliding_distance:.2f} px")
    print(f"Gap position: ({result['gap_x']:.2f}, {result['gap_y']:.2f})")
    print(f"Slider position: ({result['slider_x']:.2f}, {result['slider_y']:.2f})")
else:
    print("Detection failed")
```

#### 2. Batch Processing - Process Multiple Images

```python
from sider_captcha_solver import CaptchaPredictor
import glob
import os

# Initialize predictor
predictor = CaptchaPredictor(model_path='best', device='auto')

# Batch process CAPTCHAs
captcha_folder = 'path/to/captchas'

for img_path in glob.glob(os.path.join(captcha_folder, '*.png')):
    result = predictor.predict(img_path)

    if result['slider_x'] and result['gap_x']:
        distance = result['gap_x'] - result['slider_x']
        confidence = (result['slider_confidence'] + result['gap_confidence']) / 2
        print(f"{os.path.basename(img_path)}: Slide {distance:.1f} px (Confidence: {confidence:.3f})")
    else:
        print(f"{os.path.basename(img_path)}: Detection failed")
```

#### 3. Visualization and Debugging

```python
from sider_captcha_solver import CaptchaPredictor
import matplotlib.pyplot as plt

# Initialize predictor
predictor = CaptchaPredictor(model_path='best', device='auto')

# Test image path
test_image = 'path/to/captcha.png'

# Generate and save prediction visualization
predictor.visualize_prediction(
    test_image,
    save_path='prediction_result.png',  # Save path
    show=True                           # Display window
)

# Generate heatmap visualization (view model internal activations)
predictor.visualize_heatmaps(
    test_image,
    save_path='heatmap_result.png',    # Save 4-panel heatmap
    show=True
)

# Compare different threshold effects
thresholds = [0.0, 0.1, 0.3, 0.5]
fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4))

for idx, threshold in enumerate(thresholds):
    # Create predictor with different thresholds
    pred = CaptchaPredictor(model_path='best', hm_threshold=threshold)
    result = pred.predict(test_image)

    # Visualize to subplot
    ax = axes[idx]
    img = plt.imread(test_image)
    ax.imshow(img)
    ax.set_title(f'Threshold={threshold}')

    if result['slider_x'] and result['gap_x']:
        ax.plot(result['slider_x'], result['slider_y'], 'bo', markersize=10)
        ax.plot(result['gap_x'], result['gap_y'], 'ro', markersize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('threshold_comparison.png')
plt.show()
```

#### 4. Complete Production Environment Example

```python
from sider_captcha_solver import CaptchaPredictor
import logging
import time
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptchaSolver:
    """Production environment CAPTCHA solver wrapper"""

    def __init__(self, model_path: str = 'best', device: str = 'auto'):
        self.predictor = CaptchaPredictor(
            model_path=model_path,
            device=device,
            hm_threshold=0.1  # Balance accuracy and recall
        )
        logger.info(f"CAPTCHA solver initialized, device: {device}")

    def solve(self, image_path: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Solve CAPTCHA with retry mechanism

        Args:
            image_path: CAPTCHA image path
            max_retries: Maximum retry attempts

        Returns:
            Dictionary containing sliding distance and confidence, None on failure
        """
        for attempt in range(max_retries):
            try:
                # Record start time
                start_time = time.time()

                # Execute prediction
                result = self.predictor.predict(image_path)

                # Calculate elapsed time
                elapsed_time = (time.time() - start_time) * 1000

                # Check result validity
                if result['slider_x'] and result['gap_x']:
                    sliding_distance = result['gap_x'] - result['slider_x']
                    confidence = (result['slider_confidence'] + result['gap_confidence']) / 2

                    logger.info(f"Solve success: distance={sliding_distance:.1f}px, "
                              f"confidence={confidence:.3f}, time={elapsed_time:.1f}ms")

                    return {
                        'success': True,
                        'sliding_distance': sliding_distance,
                        'confidence': confidence,
                        'elapsed_ms': elapsed_time,
                        'details': result
                    }
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: no valid result detected")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {str(e)}")

            # Brief delay if not last attempt
            if attempt < max_retries - 1:
                time.sleep(0.1)

        logger.error(f"Solve failed: reached maximum retries {max_retries}")
        return None

# Usage example
if __name__ == "__main__":
    solver = CaptchaSolver()

    # Solve single CAPTCHA
    result = solver.solve('path/to/captcha.png')

    if result and result['success']:
        print(f"Sliding distance: {result['sliding_distance']:.1f} px")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing time: {result['elapsed_ms']:.1f} ms")
    else:
        print("CAPTCHA solving failed")
```

### Advanced Features

#### 1. Custom Model and Configuration

```python
from sider_captcha_solver import CaptchaPredictor
import torch

# Use your own trained model
custom_predictor = CaptchaPredictor(
    model_path='path/to/your_trained_model.pth',
    device='cuda:0',    # Specify GPU
    hm_threshold=0.15   # Adjust based on model characteristics
)

# Check model info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM usage: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Predict
result = custom_predictor.predict('captcha.png')
```

#### 2. Performance Benchmarking

```python
from sider_captcha_solver import CaptchaPredictor
import time
import numpy as np

# Initialize predictor
predictor = CaptchaPredictor(model_path='best', device='auto')

# Test image list
test_images = ['captcha1.png', 'captcha2.png', 'captcha3.png']

# Warm up (first inference is slower)
_ = predictor.predict(test_images[0])

# Performance test
times = []
for _ in range(10):  # Test each image 10 times
    for img_path in test_images:
        start = time.time()
        result = predictor.predict(img_path)
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        times.append(elapsed)

# Statistics
print(f"Average inference time: {np.mean(times):.2f} ms")
print(f"Fastest: {np.min(times):.2f} ms")
print(f"Slowest: {np.max(times):.2f} ms")
print(f"Std deviation: {np.std(times):.2f} ms")
print(f"FPS: {1000 / np.mean(times):.1f}")
```

## 📊 Data Generation Process

### 1. Data Collection

Downloaded high-quality images from Pixabay across 10 categories as backgrounds: Minecraft, Pixel Food, Block Public Square, Block Illustration, Backgrounds, Buildings, Nature, Anime Cityscape, Abstract Geometric Art, etc. Up to 200 images per category, totaling approximately 2,000 raw images.

### 2. CAPTCHA Generation Logic

```
Raw Images (2000+) → Resize(320×160) → Puzzle Generation
                                        ↓
                              11 shapes × 3 sizes × 4 positions
                                        ↓
                              132 CAPTCHAs per original image
                                        ↓
                              Total: 354,024 training images generated
```

**Puzzle Shape Design**:

- 5 regular puzzle shapes (combinations of convex, concave, and flat edges)
- 6 special shapes (circle, square, triangle, hexagon, pentagon, star)

**Random Parameters**:

- Puzzle size: 40-70 pixels (3 random sizes)
- Position distribution: x-axis beyond slider width + 10px to avoid overlap
- Lighting effects: Randomly added lighting variations for robustness

### 3. Dataset Split

- Training set: 90% (split by original images to avoid data leakage)
- Test set: 10% (Test Set 1)
- Real CAPTCHA test set: 100 NetEase Yidun CAPTCHAs (Test Set 2)

## 🏗️ Network Architecture

### Model Structure

```
Input (3×160×320)
    │
    ├─ Stem Conv (3×3, stride=2) ──────→ 32×80×160
    │
    ├─ ResBlock Stage-1 (×2, stride=2) ─→ 64×40×80
    │
    ├─ ResBlock Stage-2 (×2, stride=2) ─→ 128×20×40
    │
    ├─ ResBlock Stage-3 (×2, stride=2) ─→ 256×10×20
    │
    ├─ Neck (1×1 Conv) ─────────────────→ 128×10×20
    │
    ├─ UpConv-1 (3×3, stride=2) ────────→ 64×20×40
    │
    ├─ UpConv-2 (3×3, stride=2) ────────→ 64×40×80
    │
    └─┬─ Gap Detection Head ────┐
        │   ├─ Heatmap (1×40×80)   │
        │   └─ Offset (2×40×80)    │
        │                              │
        └─ Piece Detection Head ───┤
             ├─ Heatmap (1×40×80)   │
             └─ Offset (2×40×80)    │
```

### Key Design Elements

- **Backbone**: ResNet18-Lite, removed global pooling and fully connected layers
- **Detection Heads**: Dual-branch CenterNet design, detecting gap and slider centers separately
- **Loss Function**: Focal Loss (heatmap) + L1 Loss (offset regression)
- **Downsampling Rate**: 4x, output resolution 80×40
- **Activation**: ReLU (except output layers)
- **Normalization**: BatchNorm

### Model Parameters

| Component       | Parameters | Description      |
| -------------- | ---------- | ---------------- |
| Backbone       | ~3.0M      | ResNet18-Lite    |
| Neck + UpConv  | ~0.3M      | Feature fusion   |
| Detection Heads| ~0.2M      | Dual-branch heads|
| **Total**      | **~3.5M**  | FP32 ~36MB       |

## 📈 Performance Metrics

### Accuracy (Based on Sliding Distance Error)

| Dataset       | 5px Threshold | 7px Threshold | Best Epoch |
| ------------- | ------------- | ------------- | ---------- |
| Test Set (Synthetic) | 99.4%  | 99.4%        | 16         |
| Real CAPTCHAs | **73%**       | **80%**       | 15/16      |

### Inference Performance

| Hardware          | Inference Time | FPS | Batch (×32) |
| ----------------- | -------------- | --- | ----------- |
| RTX 5090          | 1.30ms         | 771 | 11.31ms     |
| AMD Ryzen 9 9950X | 5.21ms         | 192 | 144.89ms    |

### Mean Absolute Error (MAE)

- Test set: Slider 0.30px, Gap 1.14px
- Real CAPTCHAs: Slider 2.84px, Gap 9.98px

## 🛠️ Main Features

### 1. Data Generation

- Auto-download Pixabay images
- Batch generate slider CAPTCHAs
- Support multiple puzzle shapes

### 2. Model Training

- Automatic learning rate scheduling
- Training process visualization

### 3. Inference Deployment

- Support batch prediction
- REST API interface
- Heatmap visualization support

### 4. Evaluation Analysis

- Training curve analysis

## ⚠️ Disclaimer

**This project is for learning and research purposes only. Commercial or illegal use is prohibited.**

1. This project aims to promote academic research in computer vision and deep learning
2. Users must comply with relevant laws and regulations, and must not use this project to bypass website security mechanisms
3. Any legal liability arising from the use of this project shall be borne by the user
4. Please do not use this project for any behavior that may harm others' interests

## 📁 Project Structure

```
Sider_CAPTCHA_Solver/
│
├── configs/                       # Configuration files
│   └── config.yaml               # Project configuration
│
├── data/                          # Data directory
│   ├── captchas/                  # Generated CAPTCHAs (354,024 images)
│   │   └── Pic*.png              # Format: Pic{XXXX}_Bgx{X}Bgy{Y}_Sdx{X}Sdy{Y}_{hash}.png
│   ├── raw_images/                # Raw images (2000 images)
│   ├── real_captchas/             # Real CAPTCHA test set
│   │   └── annotated/             # Annotated data (100 images)
│   ├── annotations.json           # Training set annotations
│   ├── test_annotations.json      # Test set annotations
│   ├── generation_stats.json      # Generation statistics
│   └── dataset_split_stats.json   # Dataset split statistics
│
├── logs/                          # Log files
│   ├── training_accuracy_curves_all.png    # Training accuracy curves
│   ├── accuracy_comparison.png             # Test set vs real data comparison
│   ├── training_analysis_report.txt        # Training analysis report
│   ├── training_accuracy_results.csv       # Accuracy CSV data
│   ├── training_accuracy_results.json      # Accuracy JSON data
│   ├── evaluation_*.log                    # Evaluation logs
│   ├── training_log.txt                    # Training log
│   └── benchmark_results_*.json            # Performance benchmark results
│
├── results/                       # Evaluation results
│   └── best_model_evaluation/     # Best model evaluation
│       ├── test_dataset/          # Test set results
│       │   ├── evaluation_results.json     # Evaluation metrics
│       │   └── visualizations/             # Visualizations (100 images)
│       ├── real_captchas/         # Real CAPTCHA results
│       │   ├── evaluation_results.json     # Evaluation metrics
│       │   └── visualizations/             # Visualizations (50 images)
│       └── summary_report.json    # Summary report
│
├── scripts/                       # Core scripts
│   ├── annotation/                # Annotation tools
│   │   ├── annotate_captchas_matplotlib.py  # Matplotlib annotation UI
│   │   └── annotate_captchas_web.py         # Web annotation UI
│   │
│   ├── data_generation/           # Data generation scripts
│   │   ├── geometry_generator.py  # Geometry shape generator
│   │   └── puzzle_background_generator.py   # Puzzle background generator
│   │
│   ├── training/                  # Training related
│   │   ├── train.py              # Main training script
│   │   ├── dataset.py            # PyTorch dataset class
│   │   └── analyze_training.py   # Training analysis tool
│   │
│   ├── inference/                 # Inference related
│   │   └── predict.py            # Prediction interface (CaptchaPredictor class)
│   │
│   ├── evaluation/                # Evaluation scripts
│   │   └── evaluate_model.py      # Comprehensive evaluation tool (multi-mode support)
│   │
│   ├── download_images.py         # Pixabay image downloader
│   ├── generate_captchas.py       # Batch CAPTCHA generator
│   └── split_dataset.py           # Dataset splitting script
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── checkpoints/               # Model weights
│   │   ├── best_model.pth         # Best model (highest accuracy)
│   │   ├── checkpoint_epoch_0001.pth ~ checkpoint_epoch_0020.pth  # Epoch checkpoints
│   │   ├── latest_checkpoint.pth  # Latest checkpoint
│   │   ├── training_log_*.txt     # Training logs
│   │   └── logs/                  # TensorBoard logs
│   │       └── events.out.tfevents.*
│   │
│   ├── captcha_generator/         # CAPTCHA generation module
│   │   ├── __init__.py
│   │   ├── batch_generator.py    # Batch generator
│   │   ├── lighting_effects.py   # Lighting effects
│   │   ├── simple_puzzle_generator.py  # Puzzle generator
│   │   └── slider_effects.py     # Slider effects
│   │
│   ├── data_collection/           # Data collection module
│   │   ├── __init__.py
│   │   └── pixabay_downloader.py # Pixabay downloader
│   │
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── captcha_solver.py     # CaptchaSolver main model
│   │   ├── centernet_heads.py    # CenterNet detection heads
│   │   ├── losses.py             # Loss functions (Focal Loss + L1)
│   │   └── resnet18_lite.py      # ResNet18-Lite backbone
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── logger.py             # Logging utilities
│
├── tests/                        # Test scripts
│   ├── benchmark_inference.py     # Inference performance benchmark
│   ├── merge_real_captchas.py     # Real CAPTCHA merge tool
│   ├── test_all_puzzle_shapes.py  # All puzzle shapes test
│   ├── test_captcha_generation.py # CAPTCHA generation test
│   ├── test_darkness_levels.py    # Brightness level test
│   ├── test_distance_error_visualization.py  # Distance error visualization
│   ├── test_generate_captchas.py  # Generation function test
│   ├── test_model.py             # Model unit test
│   ├── test_model_architecture.py # Model architecture test
│   ├── test_real_captchas.py     # Real CAPTCHA test
│   └── test_slider_effects.py    # Slider effects test
│
├── outputs/                      # Test output files
│   └── *.png                     # Various test result images
│
├── api_example.py                # API usage examples
├── requirements.txt              # Dependencies
├── README.md                     # English documentation
├── README_zh.md                  # Chinese documentation
└── CLAUDE.md                     # Project technical specifications
```

## 🔧 Tech Stack

- **Deep Learning Framework**: PyTorch 2.0+
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: FastAPI
- **Others**: tqdm, requests, psutil

## 📞 Contact

For questions or suggestions, please submit an Issue or Pull Request.

---

<div align="center">
<i>This project is licensed under MIT License, for learning and research purposes only</i>
</div>