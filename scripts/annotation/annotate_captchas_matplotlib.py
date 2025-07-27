#!/usr/bin/env python3
"""
CAPTCHA Annotation Tool using Matplotlib
Works in Spyder and other Python environments
"""
import os
import json
import shutil
from pathlib import Path
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import cv2

class MatplotlibAnnotator:
    def __init__(self, input_dir="data/real_captchas/merged/site1", 
                 output_dir="data/real_captchas/annotated", max_images=100):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_images = max_images
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load images
        self.images = list(self.input_dir.glob("*.png"))[:max_images]
        self.current_index = 0
        self.annotations = []
        
        # Current annotation state
        self.current_image = None
        self.current_path = None
        self.clicks = []
        self.markers = []
        
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add buttons
        self.setup_buttons()
        
    def setup_buttons(self):
        """Setup control buttons"""
        # Button positions
        ax_save = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_skip = plt.axes([0.45, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.6, 0.02, 0.1, 0.04])
        
        self.btn_save = Button(ax_save, 'Save')
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        self.btn_save.on_clicked(self.save_annotation)
        self.btn_skip.on_clicked(self.skip_image)
        self.btn_reset.on_clicked(self.reset_annotation)
    
    def load_image(self):
        """Load current image"""
        if self.current_index >= len(self.images):
            self.finish()
            return
        
        self.current_path = self.images[self.current_index]
        self.current_image = cv2.imread(str(self.current_path))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Reset state
        self.clicks = []
        self.clear_markers()
        
        # Display image
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"Image {self.current_index + 1}/{len(self.images)}: {self.current_path.name}")
        self.ax.axis('off')
        
        # Add instructions
        self.update_instructions()
        plt.draw()
    
    def update_instructions(self):
        """Update instruction text"""
        if len(self.clicks) == 0:
            instruction = "Click SLIDER center (will be marked in RED)"
        elif len(self.clicks) == 1:
            instruction = "Click GAP center (will be marked in BLUE)"
        else:
            instruction = f"Slider: {self.clicks[0]}, Gap: {self.clicks[1]} - Click Save to continue"
        
        self.fig.suptitle(instruction, fontsize=14)
        plt.draw()
    
    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax:
            return
        
        if len(self.clicks) >= 2:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        self.clicks.append((x, y))
        
        # Draw marker
        if len(self.clicks) == 1:
            # Slider - Red
            circle = patches.Circle((x, y), 8, color='red', fill=True, alpha=0.7)
            self.ax.add_patch(circle)
            self.ax.text(x, y-15, 'S', color='red', fontsize=12, ha='center', weight='bold')
            self.markers.extend([circle, self.ax.texts[-1]])
        else:
            # Gap - Blue
            circle = patches.Circle((x, y), 8, color='blue', fill=True, alpha=0.7)
            self.ax.add_patch(circle)
            self.ax.text(x, y-15, 'G', color='blue', fontsize=12, ha='center', weight='bold')
            self.markers.extend([circle, self.ax.texts[-1]])
        
        self.update_instructions()
        plt.draw()
    
    def clear_markers(self):
        """Clear all markers"""
        for marker in self.markers:
            if hasattr(marker, 'remove'):
                marker.remove()
        self.markers = []
    
    def save_annotation(self, event):
        """Save current annotation"""
        if len(self.clicks) != 2:
            print("Please mark both slider and gap positions!")
            return
        
        slider_x, slider_y = self.clicks[0]
        gap_x, gap_y = self.clicks[1]
        
        # Generate filename following README.md format
        # Format: Pic{XXXX}_Bgx{gap_x}Bgy{gap_y}_Sdx{slider_x}Sdy{slider_y}_{hash}.png
        annotated_count = len(self.annotations) + 1
        hash_str = hashlib.md5(str(self.current_path).encode()).hexdigest()[:8]
        filename = f"Pic{annotated_count:04d}_Bgx{gap_x}Bgy{gap_y}_Sdx{slider_x}Sdy{slider_y}_{hash_str}.png"
        
        # Save image
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
        
        # Save annotation - 使用与test数据集一致的格式
        annotation = {
            'filename': filename,
            'bg_center': [gap_x, gap_y],      # gap对应bg_center
            'sd_center': [slider_x, slider_y]  # slider对应sd_center
        }
        self.annotations.append(annotation)
        
        print(f"Saved: {filename} (Progress: {len(self.annotations)}/{self.max_images})")
        
        # Save JSON
        self.save_json()
        
        # Next image
        self.current_index += 1
        self.load_image()
    
    def skip_image(self, event):
        """Skip current image"""
        print(f"Skipped: {self.current_path.name}")
        self.current_index += 1
        self.load_image()
    
    def reset_annotation(self, event):
        """Reset current annotation"""
        self.clicks = []
        self.clear_markers()
        
        # Redraw image
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"Image {self.current_index + 1}/{len(self.images)}: {self.current_path.name}")
        self.ax.axis('off')
        self.update_instructions()
        plt.draw()
    
    def save_json(self):
        """Save annotations to JSON"""
        json_path = self.output_dir / 'annotations.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
    
    def finish(self):
        """Finish annotation"""
        print(f"\nAnnotation complete! Annotated {len(self.annotations)} images.")
        print(f"Saved to: {self.output_dir}")
        plt.close('all')
    
    def run(self):
        """Start annotation tool"""
        print(f"Starting annotation tool...")
        print(f"Found {len(self.images)} images")
        print(f"Output directory: {self.output_dir}")
        print("\nInstructions:")
        print("1. Click to mark SLIDER center (Red)")
        print("2. Click to mark GAP center (Blue)")
        print("3. Click 'Save' button to save and move to next")
        print("4. Click 'Skip' to skip current image")
        print("5. Click 'Reset' to clear current marks")
        print("6. Close window to quit\n")
        
        # Load first image
        self.load_image()
        
        # Show plot
        plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Annotation Tool (Matplotlib)')
    parser.add_argument('--input', type=str, 
                        default='../../data/real_captchas/merged/site1',
                        help='Input directory')
    parser.add_argument('--output', type=str,
                        default='../../data/real_captchas/annotated',
                        help='Output directory')
    parser.add_argument('--max', type=int, default=100,
                        help='Maximum number of images to annotate')
    
    args = parser.parse_args()
    
    annotator = MatplotlibAnnotator(args.input, args.output, args.max)
    annotator.run()

if __name__ == '__main__':
    main()