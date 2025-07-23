# Test Scripts User Guide

This directory contains various testing and validation scripts for testing model performance, visualization effects, and data generation functionality.

## 📑 Table of Contents

1. [Performance Testing](#performance-testing)
   - [benchmark_inference.py](#benchmark_inferencepy)

2. [Visualization Testing](#visualization-testing)
   - [test_distance_error_visualization.py](#test_distance_error_visualizationpy)
   - [test_darkness_levels.py](#test_darkness_levelspy)
   - [test_slider_effects.py](#test_slider_effectspy)

3. [Data Generation Testing](#data-generation-testing)
   - [test_captcha_generation.py](#test_captcha_generationpy)
   - [test_generate_captchas.py](#test_generate_captchaspy)
   - [test_all_puzzle_shapes.py](#test_all_puzzle_shapespy)

4. [Model Testing](#model-testing)
   - [test_model_architecture.py](#test_model_architecturepy)
   - [test_real_captchas.py](#test_real_captchaspy)

5. [Data Processing Tools](#data-processing-tools)
   - [merge_real_captchas.py](#merge_real_captchaspy)

---

## Performance Testing

### benchmark_inference.py
**Function**: Comprehensively test model inference performance, including model loading, image preprocessing, inference speed, post-processing decoding, and verify prediction accuracy.

**Core Features**:
- Automatically detect and display hardware information (CPU/GPU model)
- Stage-by-stage performance testing (loading → preprocessing → inference → post-processing)
- Warmup mechanism ensures test accuracy (GPU cache warmup)
- Automatically calculate prediction accuracy (extract true coordinates from filename)
- Save results in JSON format with detailed metric explanations

**Usage Examples**:
```bash
# Basic usage - GPU test 100 times
python tests/benchmark_inference.py

# CPU test 50 times
python tests/benchmark_inference.py --device cpu --num_runs 50

# GPU test 1000 times, warmup 20 times (more accurate benchmark)
python tests/benchmark_inference.py --device cuda --num_runs 1000 --warmup_runs 20

# Quick test (10 times)
python tests/benchmark_inference.py --num_runs 10 --warmup_runs 5
```

**Output Description**:
- **Model Loading Time**: Model weight loading time
- **Mean Inference Time**: Average inference time (most important metric)
- **FPS**: Number of CAPTCHAs processable per second
- **Sliding Distance Error**: Sliding distance prediction error (pixels)

**Result Files**:
- `logs/benchmark_results_cpu.json` - CPU test results
- `logs/benchmark_results_cuda.json` - GPU test results

The JSON files contain complete performance data and detailed explanations for each metric.

---

## Visualization Testing

### test_distance_error_visualization.py
**Function**: Visualize the impact of different pixel errors (0px, 3px, 4px, 5px, 6px, 7px) on slider CAPTCHA recognition, and compare with actual model predictions.

**Core Features**:
- Generate 6 subplots showing visual effects of different error levels (including perfect prediction)
- Green circles and solid lines indicate true positions, red dashed circles indicate predicted positions
- Error magnification diagram at bottom with scale visualization and arrow annotations
- Automatic model loading and real prediction comparison (if best_model.pth exists)
- Color-coded result boxes: green for perfect prediction, yellow for errors
- Fixed test image for reproducible results and consistent benchmarking

**Usage Example**:
```bash
# Run directly - generates both error visualization and actual model prediction
python tests/test_distance_error_visualization.py

# Note: No command-line arguments needed - all parameters are fixed for consistency
```

**Output Files**:
- `outputs/distance_error_visualization.png` - 2x3 grid visualization of 6 error levels
- `outputs/actual_model_prediction.png` - Single image showing actual model prediction vs ground truth

**Console Output**:
- Selected image filename and coordinates
- True sliding distance calculation
- Model prediction results (if available):
  - Predicted gap and slider coordinates
  - Predicted vs true distance comparison
  - Absolute distance error in pixels

**Visualization Details**:
- **Green elements**: Ground truth positions and connections
- **Red elements**: Predicted/simulated positions (dashed lines)
- **Error indicators**: Scaled error bars with pixel annotations
- **Info boxes**: Distance calculations and error values

**Note**: Script uses fixed test image `Pic0001_Bgx116Bgy80_Sdx32Sdy80_9b469398.png` to ensure reproducibility.

### test_darkness_levels.py
**Function**: Test different darkness rendering effects at CAPTCHA gaps, including base darkness, edge darkness, directional lighting, and outer edge shadow parameter debugging.

**Core Features**:
- Display 4 preset darkness configurations (light, default, dark, with outer edge)
- Visualize effects of different darkness parameter combinations
- Support custom parameter testing for fine-tuning gap rendering
- Use real puzzle shapes to test lighting effects
- Generate comparison charts showing parameter impacts

**Usage Example**:
```bash
# Run complete test (generate 4 configuration comparison + custom parameter demo)
python tests/test_darkness_levels.py
```

**Output Files**:
- `outputs/darkness_levels_comparison.png` - Comparison of 4 preset configurations

**Parameter Description**:
- **base_darkness**: Base darkness inside the gap (0-50)
- **edge_darkness**: Extra darkness at gap edges for depth effect
- **directional_darkness**: Directional lighting intensity, simulating light source direction
- **outer_edge_darkness**: Soft shadow width around the gap

**Note**: Need to run `scripts/download_images.py` first to download test images.

### test_slider_effects.py
**Function**: Test slider 3D lighting rendering effects, including edge highlights, directional lighting, decay factors, and slider frame composition effects.

**Core Features**:
- Display 6 different scenarios of slider rendering effects
- Compare 4 different intensity lighting parameter configurations
- Demonstrate complete slider frame composition process
- Support custom lighting parameter debugging
- Use grayscale puzzles to show lighting effects more clearly

**Usage Example**:
```bash
# Run complete test (generate 6-grid effect chart + 4 parameter comparisons)
python tests/test_slider_effects.py
```

**Output Files**:
- `outputs/slider_lighting_test.png` - Display of 6 slider effect scenarios
- `outputs/slider_lighting_comparison.png` - Comparison of 4 lighting intensities

**Lighting Parameter Description**:
- **edge_highlight**: Edge highlight intensity (0-100)
- **directional_highlight**: Directional lighting intensity, simulating 3D effect
- **edge_width**: Highlight edge width (pixels)
- **decay_factor**: Lighting decay factor, controlling transition smoothness

**Note**: Need to run `scripts/download_images.py` first to download test images.

---

## Data Generation Testing

### test_captcha_generation.py
**Function**: Test the complete process of generating slider CAPTCHAs from real images, including puzzle extraction, gap rendering, lighting effects, and comparison of multiple puzzle shapes.

**Core Features**:
- Randomly extract puzzle pieces from real images
- Apply gap depression lighting effects (preserve content but darken)
- Add shadows to puzzle pieces for enhanced 3D effect
- Test 6 different puzzle edge combinations
- Automatically ensure reasonable puzzle position (avoid overlap with slider)

**Usage Example**:
```bash
# Run complete test (single random + 6 shape comparisons)
python tests/test_captcha_generation.py
```

**Output Files**:
- `outputs/captcha_generation_test.png` - CAPTCHA generation effect from single image
- `outputs/multiple_shapes_test.png` - 6 puzzle shape comparisons on same image

**Puzzle Shape Types**:
- **flat**: Straight edge
- **convex**: Protruding edge
- **concave**: Indented edge

**Note**: Need to run `scripts/download_images.py` first to download test images.

### test_generate_captchas.py
**Function**: Test batch CAPTCHA generation script functionality, using small-scale data to verify generation process correctness.

**Core Features**:
- Select 2 images from original images for testing
- Generate 120 CAPTCHA variants per image (10 shapes × 3 sizes × 4 positions)
- Test annotation file generation and format correctness
- Use single-process mode for easy debugging
- Automatically clean up temporary test files

**Usage Example**:
```bash
# Run small batch test (2 images, total 240 CAPTCHAs)
python tests/test_generate_captchas.py
```

**Output Directory**:
- `data/test_captchas/` - Test generated CAPTCHAs and annotation files

**Generation Rules**:
- 10 puzzle shapes: 5 regular shapes + 5 special shapes
- 3 puzzle sizes: 48px, 60px, 72px
- 4 random positions: x-axis must be greater than (slider width + 10px)

**Note**: Need to run `scripts/download_images.py` first to download test images.

### test_all_puzzle_shapes.py
**Function**: Generate and display all 87 puzzle shapes (81 regular combinations + 6 special shapes) to verify puzzle generation algorithm completeness.

**Core Features**:
- Display 81 four-edge combination shapes (3³ = 81 types)
- Display 6 special shapes (circle, square, triangle, hexagon, pentagon, star)
- Use blue puzzles with light gray background for easy observation
- Generate additional example displays for better understanding
- All shapes labeled with numbers or names

**Usage Example**:
```bash
# Generate all shape displays
python tests/test_all_puzzle_shapes.py
```

**Output Files**:
- `outputs/all_puzzle_shapes_2d.png` - Complete display of 87 shapes (10×9 grid)
- `outputs/example_puzzle_pieces.png` - Detailed examples of 6 typical shapes

**Shape Combination Rules**:
- Each edge can be: flat, convex, or concave
- Order: top, right, bottom, left
- Total: 3×3×3×3 = 81 combinations

**Special Shapes**: Used to increase CAPTCHA diversity and complexity.

---

## Model Testing

### test_model_architecture.py
**Function**: Comprehensively verify CaptchaSolver model architecture (ResNet18 Lite + CenterNet) correctness, including input/output shapes, parameter count, decoding functionality, and gradient flow.

**Core Features**:
- Verify input/output tensor shapes conform to design specifications
- Count model parameters (target: ~3.5M parameters)
- Test coordinate decoding functionality correctness
- Check gradient backpropagation normality
- Output detailed module parameter statistics

**Usage Example**:
```bash
# Run complete architecture test
python tests/test_model_architecture.py
```

**Test Items**:
1. **Shape Test**: Verify input (3×160×320) → output (4 feature maps of 40×80)
2. **Parameter Test**: Check total parameters around 3.5M (±0.5M tolerance)
3. **Decode Test**: Verify coordinate decoding within reasonable range (0-320, 0-160)
4. **Gradient Test**: Ensure all parameters can receive gradients

**Output Information**:
- Module parameter statistics (Backbone, Neck, Heads)
- Model size (MB)
- Pass/fail status for each test

### test_real_captchas.py
**Function**: Test detection performance of legacy CaptchaDetector model on real CAPTCHA datasets and generate visualization results.

**Core Features**:
- Randomly sample 100 images from real CAPTCHA dataset for testing
- Calculate Gap and Piece detection rates
- Generate 5 visualization result charts (4×5=20 samples each)
- Use red boxes to mark Gap positions, green boxes for Piece positions
- Output detection statistics in JSON format

**Usage Example**:
```bash
# Test real CAPTCHAs (need to prepare merged data first)
python tests/test_real_captchas.py
```

**Output Files**:
- `results/real_captcha_results/site1_detection_results_*.png` - Visualization results (5 charts)
- `results/real_captcha_results/site1_detection_stats.json` - Detection statistics

**Detection Markers**:
- Red dot + red box: Detected Gap position
- Green dot + green box: Detected Piece position
- Title shows: ✓ (detection success) or ✗ (detection failure)

**Note**: Need to run `merge_real_captchas.py` first to prepare data.

---

## Data Processing Tools

### merge_real_captchas.py
**Function**: Merge real CAPTCHA background (bg) and slider images into single images for model testing and visualization.

**Core Features**:
- Automatically match *_bg.png and *_slider.png file pairs
- Correctly composite slider onto background (considering alpha channel)
- Randomly place slider in left 0-10 pixel range
- Ensure uniform output size of 320×160 pixels
- Generate JSON format annotation file recording processing information

**Usage Example**:
```bash
# Merge all CAPTCHAs in site1 folder
python tests/merge_real_captchas.py
```

**Input Directory Structure**:
```
data/real_captchas/site1/
  ├── captcha_001_bg.png
  ├── captcha_001_slider.png
  ├── captcha_002_bg.png
  └── captcha_002_slider.png
```

**Output Files**:
- `data/real_captchas/merged/site1/*_merged.png` - Merged CAPTCHA images
- `data/real_captchas/merged/site1/site1_annotations.json` - Processing records

**Composition Rules**:
- Background image adjusted to 320×160
- Slider maintains original size
- Slider x coordinate: 0-10px random
- Slider y coordinate: vertically centered
- Use alpha channel blending for transparency effect

---

## 🔧 General Instructions

1. All scripts should be run from the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Most test scripts output results to the `outputs/` directory
4. Performance test results are saved to the `logs/` directory