#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式NPY数据可视化工具
使用左右键切换样本，ESC退出
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, FancyArrowPatch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path
from typing import Dict, Tuple
import cv2

class InteractiveVisualizer:
    def __init__(self, data_dir: str = None):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            data_dir = Path(__file__).parents[1] / "data" / "processed"
        self.data_dir = Path(data_dir)
        
        # 加载数据
        self.load_data()
        
        # 当前样本索引
        self.current_batch = 0
        self.current_sample = 0
        self.total_samples_in_batch = 128
        
        # 创建图形（黑色背景）
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 8), facecolor='black')
        self.fig.suptitle('CAPTCHA Dataset Visualization (Use ← → to navigate, ESC to exit)', 
                          fontsize=14, color='white')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 显示第一个样本
        self.update_display()
        
    def load_data(self):
        """加载所有批次的数据"""
        self.batches = []
        
        # 查找所有训练数据批次
        image_files = sorted(self.data_dir.glob("images/train_*.npy"))
        
        for img_file in image_files:
            batch_id = img_file.stem.split('_')[-1]
            
            # 加载对应的标签和元数据
            label_file = self.data_dir / "labels" / f"train_{batch_id}.npz"
            meta_file = self.data_dir / "labels" / f"train_{batch_id}_meta.json"
            
            if label_file.exists() and meta_file.exists():
                batch_data = {
                    'images': np.load(img_file),
                    'labels': np.load(label_file),
                    'meta': json.load(open(meta_file, 'r'))
                }
                self.batches.append(batch_data)
        
        print(f"Loaded {len(self.batches)} batches")
        
    def get_current_sample(self) -> Dict:
        """获取当前样本的数据"""
        batch = self.batches[self.current_batch]
        
        # 提取当前样本
        sample = {
            'image': batch['images'][self.current_sample],  # [4, 256, 512]
            'heatmap': batch['labels']['heatmaps'][self.current_sample],  # [2, 64, 128]
            'offset': batch['labels']['offsets'][self.current_sample],  # [4, 64, 128]
            'weight_mask': batch['labels']['weight_masks'][self.current_sample],  # [64, 128]
            'sample_id': batch['meta']['sample_ids'][self.current_sample],
            'gap_grid': batch['meta']['grid_coords'][self.current_sample]['gap'],
            'slider_grid': batch['meta']['grid_coords'][self.current_sample]['slider'],
            'gap_offset': batch['meta']['offsets_meta'][self.current_sample]['gap'],
            'slider_offset': batch['meta']['offsets_meta'][self.current_sample]['slider']
        }
        
        return sample
    
    def update_display(self):
        """更新显示内容"""
        self.fig.clear()
        
        # 获取当前样本
        sample = self.get_current_sample()
        
        # 创建子图
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 原始图像 + 热图叠加
        ax1 = self.fig.add_subplot(gs[0, :2])
        self.visualize_image_with_heatmap(ax1, sample)
        
        # 2. Weight Mask
        ax2 = self.fig.add_subplot(gs[0, 2])
        self.visualize_weight_mask(ax2, sample)
        
        # 3. Gap热图 + Offset
        ax3 = self.fig.add_subplot(gs[1, 0])
        self.visualize_heatmap_with_offset(ax3, sample, 'gap')
        
        # 4. Slider热图 + Offset
        ax4 = self.fig.add_subplot(gs[1, 1])
        self.visualize_heatmap_with_offset(ax4, sample, 'slider')
        
        # 5. 信息面板
        ax5 = self.fig.add_subplot(gs[1, 2])
        self.show_info_panel(ax5, sample)
        
        # 更新标题
        total_samples = len(self.batches) * self.total_samples_in_batch
        current_global = self.current_batch * self.total_samples_in_batch + self.current_sample + 1
        self.fig.suptitle(
            f'Sample {current_global}/{total_samples} | '
            f'Batch {self.current_batch+1}/{len(self.batches)} | '
            f'ID: {sample["sample_id"]} | '
            f'(Use ← → to navigate, ESC to exit)',
            fontsize=12,
            color='white'
        )
        
        plt.draw()
    
    def visualize_image_with_heatmap(self, ax, sample):
        """在原始图像上叠加热图"""
        # 提取RGB图像
        image = sample['image'][:3].transpose(1, 2, 0)  # [256, 512, 3]
        padding_mask = sample['image'][3]  # [256, 512]
        
        # 创建黑色背景
        display_image = np.zeros_like(image)
        # 只在非padding区域显示图像
        mask_3ch = np.stack([1-padding_mask]*3, axis=-1)
        display_image = image * mask_3ch + (1-mask_3ch) * 1.0  # padding区域显示为白色
        
        # 显示图像
        ax.imshow(display_image)
        
        # 上采样热图到原始分辨率
        gap_heatmap = cv2.resize(sample['heatmap'][0], (512, 256))
        slider_heatmap = cv2.resize(sample['heatmap'][1], (512, 256))
        
        # 叠加热图（半透明）
        gap_overlay = np.zeros((256, 512, 4))
        gap_overlay[:, :, 0] = 1.0  # 红色通道
        gap_overlay[:, :, 3] = gap_heatmap * 0.5  # Alpha通道
        ax.imshow(gap_overlay)
        
        slider_overlay = np.zeros((256, 512, 4))
        slider_overlay[:, :, 1] = 1.0  # 绿色通道
        slider_overlay[:, :, 3] = slider_heatmap * 0.5  # Alpha通道
        ax.imshow(slider_overlay)
        
        # 标记中心点
        gap_x = sample['gap_grid'][0] * 4  # 从1/4分辨率恢复（第一个是列x）
        gap_y = sample['gap_grid'][1] * 4  # 第二个是行y
        slider_x = sample['slider_grid'][0] * 4
        slider_y = sample['slider_grid'][1] * 4
        
        ax.plot(gap_x, gap_y, 'r+', markersize=15, markeredgewidth=2, label='Gap')
        ax.plot(slider_x, slider_y, 'g+', markersize=15, markeredgewidth=2, label='Slider')
        
        # 显示padding边界（用青色虚线标记）
        padding_contour = np.where(padding_mask > 0.5, 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(padding_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'c--', linewidth=2, alpha=0.7, label='Padding')
        
        ax.set_title('Image with Heatmap Overlay (Red=Gap, Green=Slider, Cyan=Padding)', color='white')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.axis('off')
    
    def visualize_weight_mask(self, ax, sample):
        """可视化权重掩码"""
        weight_mask = sample['weight_mask']
        
        im = ax.imshow(weight_mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Weight Mask\n(White=Valid, Black=Padding)\nMean: {weight_mask.mean():.2%}', 
                    color='white')
        ax.set_xlabel('Width (128)', color='white')
        ax.set_ylabel('Height (64)', color='white')
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    def visualize_heatmap_with_offset(self, ax, sample, target='gap'):
        """可视化热图和偏移"""
        if target == 'gap':
            heatmap = sample['heatmap'][0]
            offset_x = sample['offset'][0]
            offset_y = sample['offset'][1]
            grid_pos = sample['gap_grid']
            offset_val = sample['gap_offset']
            color = 'red'
            title = 'Gap'
        else:
            heatmap = sample['heatmap'][1]
            offset_x = sample['offset'][2]
            offset_y = sample['offset'][3]
            grid_pos = sample['slider_grid']
            offset_val = sample['slider_offset']
            color = 'green'
            title = 'Slider'
        
        # 显示热图
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        
        # 标记中心点
        cx, cy = grid_pos[0], grid_pos[1]  # grid_pos是(列x, 行y)
        ax.plot(cx, cy, 'b+', markersize=12, markeredgewidth=2)
        
        # 绘制偏移箭头
        if abs(offset_val[0]) > 0.01 or abs(offset_val[1]) > 0.01:
            # X方向偏移（红色箭头）
            if abs(offset_val[0]) > 0.01:
                arrow_x = FancyArrowPatch(
                    (cx, cy), (cx + offset_val[0]*5, cy),
                    arrowstyle='->', mutation_scale=15,
                    color='red', linewidth=2
                )
                ax.add_patch(arrow_x)
                ax.text(cx + offset_val[0]*5, cy - 1, f'{offset_val[0]:.3f}', 
                       color='red', fontsize=8, ha='center')
            
            # Y方向偏移（蓝色箭头）
            if abs(offset_val[1]) > 0.01:
                arrow_y = FancyArrowPatch(
                    (cx, cy), (cx, cy + offset_val[1]*5),
                    arrowstyle='->', mutation_scale=15,
                    color='blue', linewidth=2
                )
                ax.add_patch(arrow_y)
                ax.text(cx + 1, cy + offset_val[1]*5, f'{offset_val[1]:.3f}', 
                       color='blue', fontsize=8, va='center')
        
        ax.set_title(f'{title} Heatmap\nGrid: ({cx}, {cy})\nOffset: ({offset_val[0]:.3f}, {offset_val[1]:.3f})',
                    color='white')
        ax.set_xlabel('Width (128)', color='white')
        ax.set_ylabel('Height (64)', color='white')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    def show_info_panel(self, ax, sample):
        """显示信息面板"""
        ax.axis('off')
        
        info_text = f"""
Sample Information
==================
ID: {sample['sample_id']}
Batch: {self.current_batch + 1}
Index: {self.current_sample + 1}

Gap Position:
  Grid: {sample['gap_grid']}
  Offset: ({sample['gap_offset'][0]:.3f}, {sample['gap_offset'][1]:.3f})
  Final: ({sample['gap_grid'][0]*4 + sample['gap_offset'][0]*4:.1f}, 
          {sample['gap_grid'][1]*4 + sample['gap_offset'][1]*4:.1f})

Slider Position:
  Grid: {sample['slider_grid']}
  Offset: ({sample['slider_offset'][0]:.3f}, {sample['slider_offset'][1]:.3f})
  Final: ({sample['slider_grid'][0]*4 + sample['slider_offset'][0]*4:.1f},
          {sample['slider_grid'][1]*4 + sample['slider_offset'][1]*4:.1f})

Statistics:
  Heatmap Max: Gap={sample['heatmap'][0].max():.3f}, 
               Slider={sample['heatmap'][1].max():.3f}
  Valid Area: {sample['weight_mask'].mean():.1%}
  
Controls:
  ← / →        : Previous/Next sample
  Page Up/Down : Jump 10 samples backward/forward
  Home/End     : Jump 100 samples backward/forward
  ESC          : Exit
"""
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace', color='white')
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'escape':
            plt.close('all')
        elif event.key == 'right':
            self.next_sample()
        elif event.key == 'left':
            self.prev_sample()
        elif event.key == 'pagedown':
            self.jump_samples(10)  # Page Down: 前进10张
        elif event.key == 'pageup':
            self.jump_samples(-10)  # Page Up: 后退10张
        elif event.key == 'end':
            self.jump_samples(100)  # End: 前进100张
        elif event.key == 'home':
            self.jump_samples(-100)  # Home: 后退100张
    
    def jump_samples(self, count):
        """跳转指定数量的样本
        
        Args:
            count: 要跳转的样本数量（正数向前，负数向后）
        """
        # 计算总样本数
        total_samples = len(self.batches) * self.total_samples_in_batch
        
        # 计算当前全局索引
        current_global = self.current_batch * self.total_samples_in_batch + self.current_sample
        
        # 计算新的全局索引（使用模运算实现循环）
        new_global = (current_global + count) % total_samples
        
        # 如果是负数，确保正确的模运算
        if new_global < 0:
            new_global += total_samples
        
        # 转换回批次和样本索引
        self.current_batch = new_global // self.total_samples_in_batch
        self.current_sample = new_global % self.total_samples_in_batch
        
        # 更新显示
        self.update_display()
    
    def next_sample(self):
        """切换到下一个样本"""
        self.current_sample += 1
        if self.current_sample >= self.total_samples_in_batch:
            self.current_sample = 0
            self.current_batch = (self.current_batch + 1) % len(self.batches)
        self.update_display()
    
    def prev_sample(self):
        """切换到上一个样本"""
        self.current_sample -= 1
        if self.current_sample < 0:
            self.current_sample = self.total_samples_in_batch - 1
            self.current_batch = (self.current_batch - 1) % len(self.batches)
        self.update_display()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Interactive NPY visualization')
    parser.add_argument('--data-dir', type=str, 
                       default=str(Path(__file__).parents[1] / "data" / "processed"),
                       help='Path to processed data directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Interactive CAPTCHA Dataset Visualizer")
    print("=" * 60)
    print("Controls:")
    print("  → / Right Arrow  : Next sample")
    print("  ← / Left Arrow   : Previous sample")
    print("  Page Down        : Jump forward 10 samples")
    print("  Page Up          : Jump backward 10 samples")
    print("  End              : Jump forward 100 samples")
    print("  Home             : Jump backward 100 samples")
    print("  ESC              : Exit")
    print("=" * 60)
    
    visualizer = InteractiveVisualizer(args.data_dir)
    plt.show()


if __name__ == "__main__":
    main()