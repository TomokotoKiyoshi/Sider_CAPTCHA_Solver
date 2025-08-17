"""
Sider CAPTCHA Solver API Demo
============================

This script demonstrates the basic usage of the Sider CAPTCHA Solver API.

Features demonstrated:
1. Get detailed prediction results (distance, positions, confidence)
2. Visualize the detection results with marked positions

Requirements:
- sider_captcha_solver package installed
- Sample CAPTCHA image in data/captchas/ directory

Usage:
    python api_demo.py

Output:
    - Console: Sliding distance and detection details
    - File: result.png with visualization
"""

from sider_captcha_solver import solve, visualize

pic_path = r"path/to/your/image.png"  # Replace with your CAPTCHA image path

# Module 1: Get detailed information
result = solve(pic_path, detailed=True)
print(f"Sliding distance: {result['distance']:.2f} px")
print(f"Gap position: {result['gap']}")
print(f"Slider position: {result['slider']}")
print(f"Confidence: {result['confidence']:.2f}")

# Module 2: Visualize results
visualize(pic_path, save_path="result.png", show=True)

