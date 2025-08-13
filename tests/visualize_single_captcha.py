#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单张CAPTCHA图片可视化工具
显示一张图片上所有的滑块、缺口和混淆缺口位置
"""
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re


class SingleCaptchaVisualizer:
    def __init__(self, root):
        """初始化GUI"""
        self.root = root
        self.root.title("Single CAPTCHA Visualizer")
        self.root.geometry("1200x800")
        
        # 数据路径
        self.captcha_dir = Path("D:/Hacker/Sider_CAPTCHA_Solver/data/captchas")
        self.labels_file = Path("D:/Hacker/Sider_CAPTCHA_Solver/data/labels/labels_by_pic.json")
        
        # 加载标签数据
        self.load_labels()
        
        # 获取所有图片文件
        self.image_files = sorted(self.captcha_dir.glob("*.png"))
        self.current_index = 0
        
        # 创建GUI
        self.setup_gui()
        
        # 显示第一张图片
        if self.image_files:
            self.update_display()
    
    def load_labels(self):
        """加载标签数据"""
        print("Loading labels...")
        with open(self.labels_file, 'r') as f:
            self.labels_by_pic = json.load(f)
        print(f"Loaded labels for {len(self.labels_by_pic)} pictures")
    
    def setup_gui(self):
        """设置GUI界面"""
        # 顶部控制栏
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 导航按钮
        ttk.Button(control_frame, text="<<< -100", command=lambda: self.navigate(-100)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="<< -10", command=lambda: self.navigate(-10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="< Previous", command=lambda: self.navigate(-1)).pack(side=tk.LEFT, padx=5)
        
        # 当前位置标签
        self.position_label = ttk.Label(control_frame, text="", font=("Arial", 12, "bold"))
        self.position_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(control_frame, text="Next >", command=lambda: self.navigate(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+10 >>", command=lambda: self.navigate(10)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+100 >>>", command=lambda: self.navigate(100)).pack(side=tk.LEFT, padx=5)
        
        # 信息面板
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.info_text = tk.Text(info_frame, height=6, width=150, font=("Courier", 10))
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 图像显示区域
        self.figure = plt.Figure(figsize=(12, 7), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 键盘绑定
        self.root.bind('<Left>', lambda e: self.navigate(-1))
        self.root.bind('<Right>', lambda e: self.navigate(1))
        self.root.bind('<Prior>', lambda e: self.navigate(-10))  # Page Up
        self.root.bind('<Next>', lambda e: self.navigate(10))    # Page Down
        self.root.bind('<Control-Prior>', lambda e: self.navigate(-100))
        self.root.bind('<Control-Next>', lambda e: self.navigate(100))
        self.root.bind('<Escape>', lambda e: self.root.quit())
    
    def navigate(self, delta):
        """导航到其他图片"""
        new_index = self.current_index + delta
        new_index = max(0, min(new_index, len(self.image_files) - 1))
        
        if new_index != self.current_index:
            self.current_index = new_index
            self.update_display()
    
    def update_display(self):
        """更新显示内容"""
        # 清空图形
        self.figure.clear()
        
        # 获取当前图片文件
        image_path = self.image_files[self.current_index]
        
        # 提取sample_id（完整文件名不带扩展名）
        sample_id = image_path.stem
        
        # 提取pic_id（前面的PicXXXX部分）
        pic_match = re.match(r'(Pic\d+)', sample_id)
        if not pic_match:
            self.show_error(f"Cannot extract pic_id from: {sample_id}")
            return
        
        pic_id = pic_match.group(1)
        
        # 获取该图片的标签信息
        if pic_id not in self.labels_by_pic:
            self.show_error(f"No labels found for: {pic_id}")
            return
        
        # 找到对应的样本
        sample_data = None
        for sample in self.labels_by_pic[pic_id]:
            if sample['sample_id'] == sample_id:
                sample_data = sample
                break
        
        if not sample_data:
            self.show_error(f"Sample not found: {sample_id}")
            return
        
        # 更新位置标签
        self.position_label.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
        
        # 更新信息文本
        self.update_info(sample_id, sample_data)
        
        # 显示图片和标记
        self.display_image(image_path, sample_data)
        
        # 刷新画布
        self.canvas.draw()
    
    def update_info(self, sample_id, sample_data):
        """更新信息面板"""
        self.info_text.delete(1.0, tk.END)
        
        # 提取坐标
        slider_x, slider_y = sample_data['labels']['comp_piece_center']
        gap_x, gap_y = sample_data['labels']['bg_gap_center']
        
        info = f"Sample ID: {sample_id}\n"
        info += f"Picture ID: {sample_data['pic_id']}\n"
        info += f"Image Size: {sample_data['image_size']['width']}×{sample_data['image_size']['height']}\n"
        info += f"Slider Position: ({slider_x}, {slider_y})\n"
        info += f"Gap Position: ({gap_x}, {gap_y})\n"
        info += f"Puzzle Size: {sample_data['labels']['puzzle_size']}px\n"
        
        # 显示gap_pose信息
        if 'gap_pose' in sample_data['labels']:
            gap_pose = sample_data['labels']['gap_pose']
            info += f"Gap Rotation: {gap_pose.get('delta_theta_deg', 0)}°\n"
        
        # 检查混淆缺口（fake_gaps）
        if 'augmented_labels' in sample_data and 'fake_gaps' in sample_data['augmented_labels']:
            fake_gaps = sample_data['augmented_labels']['fake_gaps']
            info += f"Fake Gaps: {len(fake_gaps)} positions - "
            info += ", ".join([f"({fg['center'][0]}, {fg['center'][1]})" for fg in fake_gaps])
        else:
            info += "Fake Gaps: None"
        
        self.info_text.insert(1.0, info)
    
    def display_image(self, image_path, sample_data):
        """显示图片和标记"""
        # 创建单个轴
        ax = self.figure.add_subplot(111)
        
        # 读取图片
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 显示图片
        ax.imshow(image)
        
        # 获取坐标
        slider_x, slider_y = sample_data['labels']['comp_piece_center']
        gap_x, gap_y = sample_data['labels']['bg_gap_center']
        
        # 标记滑块位置（绿色大十字）
        ax.plot(slider_x, slider_y, 'g+', markersize=25, markeredgewidth=3, 
               label=f'Slider ({slider_x}, {slider_y})', zorder=5)
        
        # 画滑块边界框（绿色虚线框）
        puzzle_size = sample_data['labels']['puzzle_size']
        slider_rect = plt.Rectangle((slider_x - puzzle_size/2, slider_y - puzzle_size/2),
                                   puzzle_size, puzzle_size,
                                   fill=False, edgecolor='green', linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(slider_rect)
        
        # 标记缺口位置（红色大十字）
        ax.plot(gap_x, gap_y, 'r+', markersize=25, markeredgewidth=3, 
               label=f'Gap ({gap_x}, {gap_y})', zorder=5)
        
        # 画缺口边界框（红色虚线框）
        gap_rect = plt.Rectangle((gap_x - puzzle_size/2, gap_y - puzzle_size/2),
                                puzzle_size, puzzle_size,
                                fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(gap_rect)
        
        # 如果有旋转角度，显示旋转信息
        if 'gap_pose' in sample_data['labels']:
            rotation = sample_data['labels']['gap_pose'].get('delta_theta_deg', 0)
            if abs(rotation) > 0.01:
                # 在缺口位置显示旋转角度文本
                ax.text(gap_x, gap_y - puzzle_size/2 - 10, f'Rotation: {rotation:.1f}°',
                       color='red', fontsize=10, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # 标记混淆缺口（蓝色X）- fake_gaps
        if 'augmented_labels' in sample_data and 'fake_gaps' in sample_data['augmented_labels']:
            for i, fake_gap in enumerate(sample_data['augmented_labels']['fake_gaps']):
                fake_x, fake_y = fake_gap['center']
                ax.plot(fake_x, fake_y, 'bx', markersize=20, markeredgewidth=3,
                       alpha=0.8, label=f'Fake Gap {i+1} ({fake_x}, {fake_y})', zorder=4)
                
                # 画混淆缺口边界框（蓝色虚线框）
                # 检查是否有scale信息
                fake_scale = fake_gap.get('scale', 1.0)
                fake_size = puzzle_size * fake_scale
                
                fake_rect = plt.Rectangle((fake_x - fake_size/2, fake_y - fake_size/2),
                                        fake_size, fake_size,
                                        fill=False, edgecolor='blue', linewidth=2, 
                                        linestyle='--', alpha=0.5)
                ax.add_patch(fake_rect)
                
                # 如果有旋转角度，显示
                fake_rotation = fake_gap.get('delta_theta_deg', 0)
                if abs(fake_rotation) > 0.01:
                    ax.text(fake_x, fake_y + fake_size/2 + 5, f'Rot: {fake_rotation:.1f}°',
                           color='blue', fontsize=8, ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
        
        # 画连接线（从滑块到缺口）
        ax.plot([slider_x, gap_x], [slider_y, gap_y], 'y--', linewidth=1, alpha=0.5,
               label=f'Distance: {int(gap_x - slider_x)}px')
        
        # 设置标题和图例
        ax.set_title(f"{sample_data['sample_id']} - {sample_data['image_size']['width']}×{sample_data['image_size']['height']}", 
                    fontsize=14)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 图例
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
        
        # 设置坐标轴范围
        ax.set_xlim(0, sample_data['image_size']['width'])
        ax.set_ylim(sample_data['image_size']['height'], 0)  # Y轴反向
    
    def show_error(self, message):
        """显示错误信息"""
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, color='red')
        ax.set_title("Error")
        ax.axis('off')


def main():
    """主函数"""
    root = tk.Tk()
    app = SingleCaptchaVisualizer(root)
    
    print("=" * 60)
    print("Single CAPTCHA Visualizer")
    print("=" * 60)
    print("Controls:")
    print("  Left/Right Arrow    : Previous/Next image")
    print("  Page Up/Down        : Jump 10 images")
    print("  Ctrl+Page Up/Down   : Jump 100 images")
    print("  ESC                 : Exit")
    print("=" * 60)
    print("\nMarkers:")
    print("  Green +  : Slider center position (with dashed box)")
    print("  Red +    : Gap center position (with dashed box)")
    print("  Blue x   : Fake gap position (with dashed box)")
    print("  Yellow line : Distance from slider to gap")
    print("  Text     : Rotation angles for gaps")
    print("=" * 60)
    
    root.mainloop()


if __name__ == "__main__":
    main()