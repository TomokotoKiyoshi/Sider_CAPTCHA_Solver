#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate coordinate generation error with concrete example"""

print("="*70)
print("Coordinate Generation Error Demonstration")
print("="*70)

# Real example from error log
print("\nActual Error Case:")
print("-"*70)

# Example 1: Pic0002.png
print("\nExample 1: Pic0002.png")
print("  Original image size: 640x427 (downloaded from pixabay)")
print("  v")
print("  Resized to: 320x160 (by SizeVariation.apply_size_variation)")
print()

# Coordinate generation (wrong way)
print("Coordinate generation process (using wrong target_size):")
print("  Line 256: img_width, img_height = size_config.target_size")
print("  Used size: 512x256 [WRONG] (this is final NN input size, not current image size)")
print()

# Specific calculation
puzzle_size = 58
half_size = puzzle_size // 2  # 29

print(f"Puzzle parameters:")
print(f"  puzzle_size = {puzzle_size}")
print(f"  half_size = {half_size}")
print()

print("Gap position calculation:")
print(f"  gap_x_min = slider_x + 2*half_size + 10")
print(f"  gap_x_min = 29 + 2*29 + 10 = 97")
print(f"  gap_x_max = img_width - half_size")
print(f"  gap_x_max = 512 - 29 = 483 [WRONG] (based on wrong 512 width)")
print(f"  Generated gap_x = 295 (random in range 97-483)")
print()

print("Problem during validation:")
print(f"  Actual image width: 320")
print(f"  Gap right edge = gap_x + half_size = 295 + 29 = 324")
print(f"  324 > 320 --> Overflow error!")
print()

print("="*70)
print("\nCorrect approach should be:")
print("-"*70)

print("\nGeneration phase:")
print("  1. Original image: 640x427")
print("  2. Resize to: 320x160 (actual generated size)")
print("  3. Generate coordinates based on 320x160:")
print(f"     gap_x_max = 320 - 29 = 291 [CORRECT]")
print(f"     Generated gap_x should be in range 97-291")
print()

print("Training phase:")
print("  1. Load generated image (320x160) and coordinates")
print("  2. Calculate padding parameters:")
print("     - Target ratio: 512/256 = 2:1")
print("     - Current ratio: 320/160 = 2:1 (already matches)")
print("     - Scale factor: min(512/320, 256/160) = 1.6")
print("  3. Scale to 512x256")
print("  4. Transform coordinates:")
print("     - Original gap_x = 200 (in 320x160)")
print("     - Transformed gap_x = 200 * 1.6 = 320 (in 512x256)")
print()

print("="*70)
print("\nComplete size confusion workflow:")
print("-"*70)
print()
print("1. Download phase: Various sizes (640x427, 1478x831, ...)")
print("     v")
print("2. Generation phase: Resize to 280-480 x 140-240 range")
print("     v (generate coordinates based on actual size)")
print("3. Training phase: Padding + resize to 512x256")
print("     v (transform coordinates accordingly)")
print("4. Model input: Uniform 512x256 size")
print()

print("="*70)
print("\nSpecific numerical examples:")
print("-"*70)

# Example: Processing different sizes
sizes = [
    (320, 160),  # 2:1 perfect ratio
    (400, 200),  # 2:1 perfect ratio
    (280, 140),  # 2:1 perfect ratio
    (360, 200),  # 1.8:1 needs padding
    (480, 200),  # 2.4:1 needs padding
]

for w, h in sizes:
    print(f"\nOriginal size: {w}x{h}")
    
    # Calculate available gap_x range
    puzzle_size = 60  # Maximum puzzle size
    half_size = puzzle_size // 2
    
    # Based on actual size
    gap_x_min = 97  # slider_x + 2*half_size + 10
    gap_x_max_correct = w - half_size
    gap_x_max_wrong = 512 - half_size  # Wrong way
    
    print(f"  Correct gap_x range: {gap_x_min} - {gap_x_max_correct}")
    print(f"  Wrong gap_x range: {gap_x_min} - {gap_x_max_wrong}")
    
    if gap_x_max_wrong > w:
        print(f"  WARNING: May generate overflow! Max possible {gap_x_max_wrong} > image width {w}")
    
    # Scale to 512x256
    scale = min(512/w, 256/h)
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    print(f"  After scaling: {scaled_w}x{scaled_h} (scale={scale:.3f})")
    
print("\n" + "="*70)
print("\nSummary:")
print("-"*70)
print("ERROR: Line 256 in generate_captchas_with_components.py uses")
print("       size_config.target_size (512x256) for coordinate generation")
print("FIX:   Should use actual resized image size (e.g., 320x160)")
print("       which is stored in size_info['target_size'] after apply_size_variation()")
print("="*70)