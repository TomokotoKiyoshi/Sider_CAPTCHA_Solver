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
    def __init__(self, input_dir="data/real_captchas/merged/site2", 
                 output_dir="data/real_captchas/annotated/site2", max_images=100):
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
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
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
    
    def load_current_image_only(self):
        """只重新显示当前图像（用于重置）"""
        if self.current_image is not None:
            self.ax.clear()
            self.ax.imshow(self.current_image)
            self.ax.set_title(f"Image {self.current_index + 1}/{len(self.images)}: {self.current_path.name}")
            self.ax.axis('off')
    
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
        
        # 先清除标记，再清除画布
        self.clear_markers()
        
        # Display image（clear会清除所有内容）
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"Image {self.current_index + 1}/{len(self.images)}: {self.current_path.name}")
        self.ax.axis('off')
        
        # Add instructions
        self.update_instructions()
        
        # 强制刷新画布
        self.fig.canvas.draw_idle()
        plt.draw()
    
    def update_instructions(self):
        """Update instruction text"""
        if len(self.clicks) == 0:
            instruction = "点击滑块中心 (红色) | 快捷键: 空格=下一张, R=重置"
        elif len(self.clicks) == 1:
            instruction = f"点击缺口X位置 (Y自动对齐到{self.clicks[0][1]}) | 快捷键: 空格=下一张, R=重置"
        else:
            instruction = f"滑块: {self.clicks[0]}, 缺口: {self.clicks[1]} - 按空格保存并下一张"
        
        self.fig.suptitle(instruction, fontsize=14)
        plt.draw()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == ' ':  # 空格键
            # 如果已经标记了两个点，保存并进入下一张
            if len(self.clicks) == 2:
                self.save_annotation(None)
            # 否则跳过当前图片
            else:
                self.skip_image(None)
        elif event.key == 'r':  # R键重置
            self.reset_annotation(None)
        elif event.key == 'escape':  # ESC键退出
            plt.close('all')
    
    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax:
            return
        
        if len(self.clicks) >= 2:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # 第二次点击（缺口）时，强制使用滑块的Y坐标
        if len(self.clicks) == 1:
            # 缺口的Y必须与滑块的Y相同
            slider_y = self.clicks[0][1]
            self.clicks.append((x, slider_y))  # 使用滑块的Y坐标
            
            # Gap - Blue，使用滑块的Y坐标
            circle = patches.Circle((x, slider_y), 8, color='blue', fill=True, alpha=0.7)
            self.ax.add_patch(circle)
            text = self.ax.text(x, slider_y-15, 'G', color='blue', fontsize=12, ha='center', weight='bold')
            self.markers.append(circle)
            self.markers.append(text)
            
            # 画一条水平线显示Y坐标对齐
            line = self.ax.axhline(y=slider_y, color='green', linestyle='--', alpha=0.3)
            self.markers.append(line)
        else:
            # 第一次点击 - Slider
            self.clicks.append((x, y))
            
            # Slider - Red
            circle = patches.Circle((x, y), 8, color='red', fill=True, alpha=0.7)
            self.ax.add_patch(circle)
            text = self.ax.text(x, y-15, 'S', color='red', fontsize=12, ha='center', weight='bold')
            self.markers.append(circle)
            self.markers.append(text)
        
        self.update_instructions()
        plt.draw()
    
    def clear_markers(self):
        """Clear all markers"""
        # 清除所有标记
        for marker in self.markers:
            if hasattr(marker, 'remove'):
                marker.remove()
        
        # 清除所有文本（确保文本也被清除）
        for text in self.ax.texts[:]:  # 使用切片创建副本避免修改列表时的问题
            text.remove()
        
        # 清除所有patches（圆圈等）
        for patch in self.ax.patches[:]:
            patch.remove()
        
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
        
        # 在加载下一张图片前，先执行reset功能彻底清除
        self.reset_annotation(None)
        
        # Next image
        self.current_index += 1
        self.load_image()
    
    def skip_image(self, event):
        """Skip current image"""
        print(f"Skipped: {self.current_path.name}")
        
        # 执行reset功能彻底清除
        self.reset_annotation(None)
        
        self.current_index += 1
        self.load_image()
    
    def reset_annotation(self, event):
        """Reset current annotation - 彻底清除所有标记"""
        # 重置点击记录
        self.clicks = []
        
        # 清除所有标记
        self.clear_markers()
        
        # 清除所有axes上的内容并重新显示图像
        self.ax.clear()
        
        # 清除所有残留的artists
        for artist in self.ax.get_children():
            if hasattr(artist, 'remove'):
                try:
                    artist.remove()
                except:
                    pass
        
        # 重新显示当前图像
        if self.current_image is not None:
            self.ax.imshow(self.current_image)
            self.ax.set_title(f"Image {self.current_index + 1}/{len(self.images)}: {self.current_path.name}")
            self.ax.axis('off')
        
        # 更新指令
        self.update_instructions()
        
        # 强制刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
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