#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPY数据集交互式GUI可视化工具
支持键盘和鼠标导航，显示热力图和坐标信息
"""
import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class NPYVisualizerGUI:
    """NPY数据集GUI可视化器"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("NPY Dataset Visualizer")
        self.master.geometry("1400x900")
        
        # 数据状态
        self.data_root = Path("data/processed")
        self.current_mode = "train"  # train, val, test
        self.current_batch_id = 0
        self.current_sample_idx = 0
        self.batch_data = None
        self.batch_meta = None
        self.max_batch_id = 0
        
        # 创建UI
        self.setup_ui()
        
        # 加载初始数据
        self.update_mode_info()
        self.load_batch(0)
        
        # 绑定键盘事件
        self.master.bind("<Left>", lambda e: self.prev_sample())
        self.master.bind("<Right>", lambda e: self.next_sample())
        self.master.bind("<Prior>", lambda e: self.prev_batch())  # Page Up
        self.master.bind("<Next>", lambda e: self.next_batch())   # Page Down
        self.master.bind("<Home>", lambda e: self.first_sample())
        self.master.bind("<End>", lambda e: self.last_sample())
        
    def setup_ui(self):
        """设置UI布局"""
        # 顶部控制栏
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 模式选择
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="train")
        mode_combo = ttk.Combobox(control_frame, textvariable=self.mode_var, 
                                  values=["train", "val", "test"], width=10)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.switch_mode())
        
        # 批次信息
        self.batch_label = ttk.Label(control_frame, text="Batch: 0/0")
        self.batch_label.pack(side=tk.LEFT, padx=20)
        
        self.sample_label = ttk.Label(control_frame, text="Sample: 0/0")
        self.sample_label.pack(side=tk.LEFT, padx=20)
        
        # 导航按钮
        ttk.Button(control_frame, text="<<", command=self.first_batch, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="<100", command=self.prev_100, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="<", command=self.prev_sample, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text=">", command=self.next_sample, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text=">100", command=self.next_100, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text=">>", command=self.last_batch, width=3).pack(side=tk.LEFT, padx=2)
        
        # 跳转输入
        ttk.Label(control_frame, text="Go to:").pack(side=tk.LEFT, padx=10)
        self.goto_entry = ttk.Entry(control_frame, width=10)
        self.goto_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Jump", command=self.goto_sample).pack(side=tk.LEFT, padx=2)
        
        # 主内容区域
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：图像显示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建标签页
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像标签页
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image + Overlay")
        
        # 热力图标签页
        self.heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="Heatmaps")
        
        # 权重掩码标签页
        self.weight_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.weight_frame, text="Weight Mask")
        
        # 图像画布
        self.image_canvas = tk.Canvas(self.image_frame, bg='black')
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # matplotlib图形
        self.setup_matplotlib_figures()
        
        # 右侧：信息显示
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.pack_propagate(False)
        
        # 坐标公式显示
        self.info_text = scrolledtext.ScrolledText(right_frame, width=50, height=40, 
                                                   font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_bar = ttk.Label(self.master, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_matplotlib_figures(self):
        """设置matplotlib图形"""
        # 热力图
        self.heatmap_fig = plt.Figure(figsize=(10, 5), dpi=80)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, self.heatmap_frame)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 权重掩码
        self.weight_fig = plt.Figure(figsize=(10, 5), dpi=80)
        self.weight_canvas = FigureCanvasTkAgg(self.weight_fig, self.weight_frame)
        self.weight_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_mode_info(self):
        """更新模式信息"""
        # 查找该模式下的批次数
        mode_dir = self.data_root / "images" / self.current_mode
        if mode_dir.exists():
            batch_files = sorted(mode_dir.glob(f"{self.current_mode}_*.npy"))
            if batch_files:
                # 提取最大批次ID
                last_file = batch_files[-1]
                self.max_batch_id = int(last_file.stem.split('_')[-1])
            else:
                self.max_batch_id = 0
        else:
            self.max_batch_id = 0
            
    def load_batch(self, batch_id):
        """加载指定批次的数据"""
        try:
            # 构建文件路径
            prefix = f"{self.current_mode}_{batch_id:04d}"
            image_path = self.data_root / "images" / self.current_mode / f"{prefix}.npy"
            heatmap_path = self.data_root / "labels" / self.current_mode / f"{prefix}_heatmaps.npy"
            offset_path = self.data_root / "labels" / self.current_mode / f"{prefix}_offsets.npy"
            weight_path = self.data_root / "labels" / self.current_mode / f"{prefix}_weights.npy"
            meta_path = self.data_root / "labels" / self.current_mode / f"{prefix}_meta.json"
            
            # 检查文件是否存在
            if not all(p.exists() for p in [image_path, heatmap_path, offset_path, weight_path]):
                self.status_bar.config(text=f"Error: Missing files for batch {batch_id}")
                return False
            
            # 加载数据
            self.batch_data = {
                'images': np.load(image_path),      # [B, 4, 256, 512]
                'heatmaps': np.load(heatmap_path),  # [B, 2, 64, 128]
                'offsets': np.load(offset_path),    # [B, 4, 64, 128]
                'weights': np.load(weight_path)     # [B, 1, 64, 128] or [B, 64, 128]
            }
            
            # 处理权重维度
            if self.batch_data['weights'].ndim == 3:
                self.batch_data['weights'] = np.expand_dims(self.batch_data['weights'], axis=1)
            
            # 加载元数据
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.batch_meta = json.load(f)
            else:
                self.batch_meta = None
            
            self.current_batch_id = batch_id
            self.current_sample_idx = 0
            
            # 更新UI
            batch_size = self.batch_data['images'].shape[0]
            self.batch_label.config(text=f"Batch: {batch_id}/{self.max_batch_id}")
            self.sample_label.config(text=f"Sample: 1/{batch_size}")
            
            # 显示第一个样本
            self.display_sample()
            
            self.status_bar.config(text=f"Loaded batch {batch_id} ({batch_size} samples)")
            return True
            
        except Exception as e:
            self.status_bar.config(text=f"Error loading batch: {str(e)}")
            return False
    
    def display_sample(self):
        """显示当前样本"""
        if self.batch_data is None:
            return
            
        idx = self.current_sample_idx
        batch_size = self.batch_data['images'].shape[0]
        
        # 更新样本标签
        self.sample_label.config(text=f"Sample: {idx+1}/{batch_size}")
        
        # 显示图像和叠加层
        self.display_image_with_overlay(idx)
        
        # 显示热力图
        self.display_heatmaps(idx)
        
        # 显示权重掩码
        self.display_weight_mask(idx)
        
        # 显示坐标信息
        self.display_coordinate_info(idx)
        
    def display_image_with_overlay(self, idx):
        """显示图像和叠加的标注"""
        # 获取图像数据 [4, 256, 512]
        image_data = self.batch_data['images'][idx]
        
        # 提取RGB通道 [3, 256, 512] -> [256, 512, 3]
        rgb_image = image_data[:3].transpose(1, 2, 0)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_image)
        
        # 创建绘制对象
        draw = ImageDraw.Draw(pil_image)
        
        # 获取热力图峰值位置
        gap_heatmap = self.batch_data['heatmaps'][idx, 0]  # [64, 128]
        slider_heatmap = self.batch_data['heatmaps'][idx, 1]
        
        # 找到峰值
        gap_y, gap_x = np.unravel_index(np.argmax(gap_heatmap), gap_heatmap.shape)
        slider_y, slider_x = np.unravel_index(np.argmax(slider_heatmap), slider_heatmap.shape)
        
        # 获取偏移
        gap_offset_x = self.batch_data['offsets'][idx, 0, gap_y, gap_x]
        gap_offset_y = self.batch_data['offsets'][idx, 1, gap_y, gap_x]
        slider_offset_x = self.batch_data['offsets'][idx, 2, slider_y, slider_x]
        slider_offset_y = self.batch_data['offsets'][idx, 3, slider_y, slider_x]
        
        # 计算实际坐标（上采样到原始分辨率）
        gap_x_coord = 4 * (gap_x + 0.5 + gap_offset_x)
        gap_y_coord = 4 * (gap_y + 0.5 + gap_offset_y)
        slider_x_coord = 4 * (slider_x + 0.5 + slider_offset_x)
        slider_y_coord = 4 * (slider_y + 0.5 + slider_offset_y)
        
        # 绘制标记
        # Gap center (红色)
        draw.ellipse([gap_x_coord-5, gap_y_coord-5, gap_x_coord+5, gap_y_coord+5], 
                    outline='red', width=2)
        draw.line([gap_x_coord-10, gap_y_coord, gap_x_coord+10, gap_y_coord], 
                 fill='red', width=1)
        draw.line([gap_x_coord, gap_y_coord-10, gap_x_coord, gap_y_coord+10], 
                 fill='red', width=1)
        draw.text((gap_x_coord+10, gap_y_coord-10), "Gap", fill='red')
        
        # Slider center (绿色)
        draw.ellipse([slider_x_coord-5, slider_y_coord-5, slider_x_coord+5, slider_y_coord+5], 
                    outline='lime', width=2)
        draw.line([slider_x_coord-10, slider_y_coord, slider_x_coord+10, slider_y_coord], 
                 fill='lime', width=1)
        draw.line([slider_x_coord, slider_y_coord-10, slider_x_coord, slider_y_coord+10], 
                 fill='lime', width=1)
        draw.text((slider_x_coord+10, slider_y_coord-10), "Slider", fill='lime')
        
        # 如果有混淆缺口，用蓝色十字标记
        if self.batch_meta and 'confusing_gaps' in self.batch_meta:
            confusing_gaps = self.batch_meta['confusing_gaps'][idx]
            for i, gap in enumerate(confusing_gaps):
                if gap and len(gap) >= 2:  # 确保不是空的且有足够的坐标
                    # 混淆缺口使用蓝色十字标记
                    fake_x = 4 * (gap[0] + 0.5)
                    fake_y = 4 * (gap[1] + 0.5)
                    
                    # 绘制蓝色十字
                    cross_size = 12
                    draw.line([fake_x - cross_size, fake_y, fake_x + cross_size, fake_y], 
                             fill='blue', width=2)
                    draw.line([fake_x, fake_y - cross_size, fake_x, fake_y + cross_size], 
                             fill='blue', width=2)
                    
                    # 添加标签
                    draw.text((fake_x + cross_size + 3, fake_y - 8), f"Fake{i+1}", fill='blue')
        
        # 调整显示大小
        display_width = 768  # 512 * 1.5
        display_height = 384  # 256 * 1.5
        pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter图像
        photo = ImageTk.PhotoImage(pil_image)
        
        # 获取画布大小以居中显示
        self.image_canvas.update_idletasks()
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # 如果画布还没有初始化，使用默认值
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
            
        # 计算居中位置
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        
        # 更新画布
        self.image_canvas.delete("all")
        self.image_canvas.create_image(x_center, y_center, image=photo, anchor='center')
        self.image_canvas.image = photo  # 保持引用
        
    def display_heatmaps(self, idx):
        """显示热力图"""
        self.heatmap_fig.clear()
        
        # 创建子图
        ax1 = self.heatmap_fig.add_subplot(121)
        ax2 = self.heatmap_fig.add_subplot(122)
        
        # Gap热力图
        gap_heatmap = self.batch_data['heatmaps'][idx, 0]
        im1 = ax1.imshow(gap_heatmap, cmap='hot', interpolation='nearest')
        ax1.set_title('Gap Heatmap')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        self.heatmap_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 标记峰值
        gap_y, gap_x = np.unravel_index(np.argmax(gap_heatmap), gap_heatmap.shape)
        ax1.plot(gap_x, gap_y, 'b+', markersize=10, markeredgewidth=2)
        
        # Slider热力图
        slider_heatmap = self.batch_data['heatmaps'][idx, 1]
        im2 = ax2.imshow(slider_heatmap, cmap='hot', interpolation='nearest')
        ax2.set_title('Slider Heatmap')
        ax2.set_xlabel('Width')
        ax2.set_ylabel('Height')
        self.heatmap_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 标记峰值
        slider_y, slider_x = np.unravel_index(np.argmax(slider_heatmap), slider_heatmap.shape)
        ax2.plot(slider_x, slider_y, 'b+', markersize=10, markeredgewidth=2)
        
        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw()
        
    def display_weight_mask(self, idx):
        """显示权重掩码"""
        self.weight_fig.clear()
        
        ax = self.weight_fig.add_subplot(111)
        
        # 获取权重掩码
        weight_mask = self.batch_data['weights'][idx, 0]  # [64, 128]
        
        im = ax.imshow(weight_mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title('Weight Mask (1=valid, 0=padding)')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        self.weight_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 显示统计信息
        valid_ratio = np.mean(weight_mask)
        ax.text(0.02, 0.98, f"Valid: {valid_ratio:.1%}", 
               transform=ax.transAxes, va='top', color='white', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        self.weight_fig.tight_layout()
        self.weight_canvas.draw()
        
    def display_coordinate_info(self, idx):
        """显示坐标恢复公式信息"""
        # 获取数据
        gap_heatmap = self.batch_data['heatmaps'][idx, 0]
        slider_heatmap = self.batch_data['heatmaps'][idx, 1]
        
        # 找到峰值网格坐标
        gap_y, gap_x = np.unravel_index(np.argmax(gap_heatmap), gap_heatmap.shape)
        slider_y, slider_x = np.unravel_index(np.argmax(slider_heatmap), slider_heatmap.shape)
        
        # 获取偏移
        gap_offset_x = self.batch_data['offsets'][idx, 0, gap_y, gap_x]
        gap_offset_y = self.batch_data['offsets'][idx, 1, gap_y, gap_x]
        slider_offset_x = self.batch_data['offsets'][idx, 2, slider_y, slider_x]
        slider_offset_y = self.batch_data['offsets'][idx, 3, slider_y, slider_x]
        
        # 计算letterbox坐标
        gap_letterbox_x = 4 * (gap_x + 0.5 + gap_offset_x)
        gap_letterbox_y = 4 * (gap_y + 0.5 + gap_offset_y)
        slider_letterbox_x = 4 * (slider_x + 0.5 + slider_offset_x)
        slider_letterbox_y = 4 * (slider_y + 0.5 + slider_offset_y)
        
        # 格式化信息文本
        info = f"""
╔══════════════════════════════════════════════════════════════╗
║              Coordinate Recovery Formula                     ║
╚══════════════════════════════════════════════════════════════╝

Sample: {idx+1}/{self.batch_data['images'].shape[0]}
Batch: {self.current_batch_id} | Mode: {self.current_mode}

[Step 1] Grid → Letterbox (512×256)
  letterbox = 4 × (grid + 0.5 + offset)
  
  Gap X:    {gap_letterbox_x:6.1f} = 4×({gap_x}+0.5+{gap_offset_x:6.3f})
  Gap Y:    {gap_letterbox_y:6.1f} = 4×({gap_y}+0.5+{gap_offset_y:6.3f})
  
  Slider X: {slider_letterbox_x:6.1f} = 4×({slider_x}+0.5+{slider_offset_x:6.3f})
  Slider Y: {slider_letterbox_y:6.1f} = 4×({slider_y}+0.5+{slider_offset_y:6.3f})

[Heatmap Statistics]
  Gap Heatmap:
    Max value: {gap_heatmap.max():.4f}
    Peak position: ({gap_x}, {gap_y})
    
  Slider Heatmap:
    Max value: {slider_heatmap.max():.4f}
    Peak position: ({slider_x}, {slider_y})

[Weight Mask]
  Valid pixels: {np.mean(self.batch_data['weights'][idx, 0]):.1%}
  Shape: {self.batch_data['weights'][idx, 0].shape}
"""

        # 如果有元数据，添加更多信息
        if self.batch_meta:
            if 'gap_coords' in self.batch_meta and idx < len(self.batch_meta['gap_coords']):
                gap_meta = self.batch_meta['gap_coords'][idx]
                slider_meta = self.batch_meta['slider_coords'][idx]
                info += f"""
[Metadata Coordinates]
  Gap: ({gap_meta[0]:.1f}, {gap_meta[1]:.1f})
  Slider: ({slider_meta[0]:.1f}, {slider_meta[1]:.1f})
"""
            
            if 'confusing_gaps' in self.batch_meta and idx < len(self.batch_meta['confusing_gaps']):
                conf_gaps = self.batch_meta['confusing_gaps'][idx]
                if conf_gaps:
                    info += f"\n[Confusing Gaps]\n"
                    for i, gap in enumerate(conf_gaps):
                        if gap:
                            info += f"  Fake {i+1}: Grid ({gap[0]}, {gap[1]})\n"
        
        info += """
[Navigation Keys]
  Left/Right Arrow: Previous/Next sample
  Page Up/Down: Previous/Next batch (-100/+100 samples)
  Home/End: First/Last sample in batch
  
[Mouse]
  Click buttons or use dropdown to switch modes
"""
        
        # 更新文本框
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        
    def switch_mode(self):
        """切换数据集模式"""
        self.current_mode = self.mode_var.get()
        self.current_batch_id = 0
        self.current_sample_idx = 0
        self.update_mode_info()
        self.load_batch(0)
        
    def prev_sample(self):
        """上一个样本"""
        if self.batch_data is None:
            return
            
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.display_sample()
        else:
            # 尝试加载上一个批次的最后一个样本
            if self.current_batch_id > 0:
                if self.load_batch(self.current_batch_id - 1):
                    self.current_sample_idx = self.batch_data['images'].shape[0] - 1
                    self.display_sample()
                    
    def next_sample(self):
        """下一个样本"""
        if self.batch_data is None:
            return
            
        batch_size = self.batch_data['images'].shape[0]
        if self.current_sample_idx < batch_size - 1:
            self.current_sample_idx += 1
            self.display_sample()
        else:
            # 尝试加载下一个批次的第一个样本
            if self.current_batch_id < self.max_batch_id:
                if self.load_batch(self.current_batch_id + 1):
                    self.current_sample_idx = 0
                    self.display_sample()
                    
    def prev_100(self):
        """后退100个样本"""
        self.jump_samples(-100)
        
    def next_100(self):
        """前进100个样本"""
        self.jump_samples(100)
        
    def prev_batch(self):
        """上一个批次"""
        if self.current_batch_id > 0:
            self.load_batch(self.current_batch_id - 1)
            
    def next_batch(self):
        """下一个批次"""
        if self.current_batch_id < self.max_batch_id:
            self.load_batch(self.current_batch_id + 1)
            
    def first_batch(self):
        """第一个批次"""
        self.load_batch(0)
        
    def last_batch(self):
        """最后一个批次"""
        self.load_batch(self.max_batch_id)
        
    def first_sample(self):
        """批次中第一个样本"""
        self.current_sample_idx = 0
        self.display_sample()
        
    def last_sample(self):
        """批次中最后一个样本"""
        if self.batch_data is not None:
            self.current_sample_idx = self.batch_data['images'].shape[0] - 1
            self.display_sample()
            
    def jump_samples(self, delta):
        """跳转指定数量的样本"""
        if self.batch_data is None:
            return
            
        # 计算全局样本索引
        global_idx = self.current_batch_id * 64 + self.current_sample_idx
        new_global_idx = global_idx + delta
        
        # 限制范围
        new_global_idx = max(0, new_global_idx)
        
        # 计算新的批次和样本索引
        new_batch_id = new_global_idx // 64
        new_sample_idx = new_global_idx % 64
        
        # 限制批次范围
        new_batch_id = min(new_batch_id, self.max_batch_id)
        
        # 加载新批次
        if new_batch_id != self.current_batch_id:
            if self.load_batch(new_batch_id):
                # 确保样本索引在范围内
                batch_size = self.batch_data['images'].shape[0]
                self.current_sample_idx = min(new_sample_idx, batch_size - 1)
                self.display_sample()
        else:
            # 同一批次内跳转
            batch_size = self.batch_data['images'].shape[0]
            self.current_sample_idx = min(new_sample_idx, batch_size - 1)
            self.display_sample()
            
    def goto_sample(self):
        """跳转到指定样本"""
        try:
            target = self.goto_entry.get().strip()
            if '/' in target:
                # 格式: batch/sample
                batch_id, sample_idx = map(int, target.split('/'))
                if self.load_batch(batch_id):
                    batch_size = self.batch_data['images'].shape[0]
                    self.current_sample_idx = min(sample_idx, batch_size - 1)
                    self.display_sample()
            else:
                # 全局索引
                global_idx = int(target)
                self.jump_samples(global_idx - (self.current_batch_id * 64 + self.current_sample_idx))
        except ValueError:
            self.status_bar.config(text="Invalid input. Use 'batch/sample' or global index")


def main():
    """主函数"""
    root = tk.Tk()
    app = NPYVisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()