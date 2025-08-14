#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式NPY数据可视化工具
使用左右键切换样本，ESC退出
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import json
from pathlib import Path
from typing import Dict
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
        
        # 加载原始标签数据用于对比
        self.load_original_labels()
        
        # 初始化状态变量
        self.current_batch = 0
        self.current_sample = 0
        self.current_batch_data = None  # 当前加载的批次数据
        self.total_samples_in_batch = 64  # 默认值
        
        # 扫描可用批次并加载第一个
        self.scan_batches()
        
        # 创建图形（黑色背景）
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 8), facecolor='black')
        self.fig.suptitle('CAPTCHA Dataset Visualization (Use ← → to navigate, ESC to exit)', 
                          fontsize=14, color='white')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 显示第一个样本
        self.update_display()
    
    def load_original_labels(self):
        """加载原始标签数据用于对比"""
        labels_path = Path(__file__).parents[1] / "data" / "labels" / "labels_by_pic.json"
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.original_labels = json.load(f)
            print(f"Loaded original labels for {len(self.original_labels)} pictures")
        else:
            self.original_labels = {}
            print("Warning: Original labels file not found")
        
    def scan_batches(self):
        """扫描可用的批次文件（不加载数据）"""
        self.batch_files = []
        
        # 查找所有训练数据批次
        image_files = sorted((self.data_dir / "images").glob("train_*.npy"))
        
        for img_file in image_files:
            # 提取批次ID（4位数字格式）
            batch_id = img_file.stem.replace('train_', '')
            
            # 检查对应的标签和元数据文件是否存在
            heatmap_file = self.data_dir / "labels" / f"train_{batch_id}_heatmaps.npy"
            offset_file = self.data_dir / "labels" / f"train_{batch_id}_offsets.npy"
            weight_file = self.data_dir / "labels" / f"train_{batch_id}_weights.npy"
            meta_file = self.data_dir / "labels" / f"train_{batch_id}_meta.json"
            
            if heatmap_file.exists() and offset_file.exists() and weight_file.exists() and meta_file.exists():
                self.batch_files.append({
                    'batch_id': batch_id,
                    'image_file': img_file,
                    'heatmap_file': heatmap_file,
                    'offset_file': offset_file,
                    'weight_file': weight_file,
                    'meta_file': meta_file
                })
        
        print(f"Found {len(self.batch_files)} batches")
        if self.batch_files:
            # 加载第一个批次
            success = self.load_batch(0)
            if not success:
                print("Failed to load first batch")
                self.current_batch_data = None
    
    def load_batch(self, batch_index: int):
        """按需加载指定批次的数据"""
        if batch_index < 0 or batch_index >= len(self.batch_files):
            print(f"Invalid batch index: {batch_index}")
            return False
            
        batch_info = self.batch_files[batch_index]
        print(f"Loading batch {batch_index} (ID: {batch_info['batch_id']})...")
        
        try:
            # 加载数据
            self.current_batch_data = {
                'images': np.load(batch_info['image_file']),
                'heatmaps': np.load(batch_info['heatmap_file']),
                'offsets': np.load(batch_info['offset_file']),
                'weight_masks': np.load(batch_info['weight_file']),
                'meta': json.load(open(batch_info['meta_file'], 'r', encoding='utf-8'))
            }
            
            # 更新批次大小
            self.total_samples_in_batch = self.current_batch_data['images'].shape[0]
            print(f"Loaded batch with {self.total_samples_in_batch} samples")
            
            # 更新当前批次索引
            self.current_batch = batch_index
            # 重置样本索引
            self.current_sample = 0
            return True
        except Exception as e:
            print(f"Error loading batch: {e}")
            import traceback
            traceback.print_exc()
            self.current_batch_data = None
            return False
        
    def get_current_sample(self) -> Dict:
        """获取当前样本的数据"""
        if self.current_batch_data is None:
            raise ValueError("No batch loaded")
            
        batch = self.current_batch_data
        
        # 提取当前样本
        sample = {
            'image': batch['images'][self.current_sample],  # [4, 256, 512]
            'heatmap': batch['heatmaps'][self.current_sample],  # [2, 64, 128]
            'offset': batch['offsets'][self.current_sample],  # [4, 64, 128]
            'weight_mask': batch['weight_masks'][self.current_sample],  # [64, 128]
            'sample_id': batch['meta']['sample_ids'][self.current_sample],
            'gap_grid': batch['meta']['grid_coords'][self.current_sample]['gap'],
            'slider_grid': batch['meta']['grid_coords'][self.current_sample]['slider'],
            'gap_offset': batch['meta']['offsets_meta'][self.current_sample]['gap'],
            'slider_offset': batch['meta']['offsets_meta'][self.current_sample]['slider']
        }
        
        # 添加混淆缺口信息（如果存在）
        if 'confusing_gaps' in batch['meta'] and self.current_sample < len(batch['meta']['confusing_gaps']):
            sample['confusing_gaps'] = batch['meta']['confusing_gaps'][self.current_sample]
        else:
            sample['confusing_gaps'] = []
            
        # 添加原始标签数据和图像尺寸用于对比
        original_data = self.get_original_data(sample['sample_id'])
        if original_data:
            sample['original_labels'] = original_data['labels']
            sample['original_size'] = original_data['image_size']
            # 从原始数据中获取混淆缺口信息
            if 'augmented_labels' in original_data and 'fake_gaps' in original_data['augmented_labels']:
                sample['fake_gaps'] = original_data['augmented_labels']['fake_gaps']
            else:
                sample['fake_gaps'] = []
        else:
            sample['original_labels'] = None
            sample['original_size'] = None
            sample['fake_gaps'] = []
        
        return sample
    
    def get_original_data(self, sample_id: str):
        """获取原始标签数据和图像尺寸"""
        # 从sample_id提取pic_id (如: Pic0003_Bgx95Bgy35_Sdx25Sdy35_b77f03f9 -> Pic0003)
        if '_' in sample_id:
            pic_id = sample_id.split('_')[0]
        else:
            return None
        
        # 查找对应的原始标签
        if pic_id in self.original_labels:
            for sample in self.original_labels[pic_id]:
                if sample['sample_id'] == sample_id:
                    return sample
        
        return None
    
    def update_display(self):
        """更新显示内容"""
        self.fig.clear()
        
        # 获取当前样本
        sample = self.get_current_sample()
        
        # 创建子图 - 简化布局
        gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.2)
        
        # 1. 原始图像 + 热图叠加
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.visualize_image_with_heatmap(ax1, sample)
        
        # 2. 信息面板（更大的空间）
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.show_info_panel(ax2, sample)
        
        # 更新标题
        total_batches = len(self.batch_files)
        current_in_batch = self.current_sample + 1
        self.fig.suptitle(
            f'Batch {self.current_batch+1}/{total_batches} | Sample {current_in_batch}/{self.total_samples_in_batch} | '
            f'{sample["sample_id"]} | (← → navigate, B/Shift+B batch, ESC exit)',
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
        
        # 标记中心点 - 正确的计算方式（包含中心偏移和子像素偏移）
        # 公式: coord = downsample * (grid + 0.5 + offset)
        gap_x = 4 * (sample['gap_grid'][0] + 0.5 + sample['gap_offset'][0])
        gap_y = 4 * (sample['gap_grid'][1] + 0.5 + sample['gap_offset'][1])
        slider_x = 4 * (sample['slider_grid'][0] + 0.5 + sample['slider_offset'][0])
        slider_y = 4 * (sample['slider_grid'][1] + 0.5 + sample['slider_offset'][1])
        
        ax.plot(gap_x, gap_y, 'r+', markersize=15, markeredgewidth=2, label='Real Gap')
        ax.plot(slider_x, slider_y, 'g+', markersize=15, markeredgewidth=2, label='Slider')
        
        # 显示混淆缺口（如果存在）
        if sample.get('fake_gaps'):
            for i, fake_gap in enumerate(sample['fake_gaps']):
                # 获取原始坐标
                fake_x, fake_y = fake_gap['center']
                
                # 如果有原始尺寸信息，进行letterbox变换
                if sample.get('original_size'):
                    orig_width = sample['original_size']['width']
                    orig_height = sample['original_size']['height']
                    
                    # 计算letterbox缩放和偏移
                    scale = min(512.0 / orig_width, 256.0 / orig_height)
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                    pad_x = (512 - new_width) // 2
                    pad_y = (256 - new_height) // 2
                    
                    # 变换坐标到letterbox空间
                    fake_letterbox_x = fake_x * scale + pad_x
                    fake_letterbox_y = fake_y * scale + pad_y
                else:
                    # 如果没有原始尺寸，假设坐标已经在letterbox空间
                    fake_letterbox_x = fake_x
                    fake_letterbox_y = fake_y
                
                # 画混淆缺口标记（蓝色X和圆圈）
                ax.plot(fake_letterbox_x, fake_letterbox_y, 'bx', markersize=12, 
                       markeredgewidth=2, alpha=0.8)
                circle = plt.Circle((fake_letterbox_x, fake_letterbox_y), radius=5, 
                                   fill=False, edgecolor='blue', linewidth=2, alpha=0.8)
                ax.add_patch(circle)
                
                # 显示混淆缺口编号和坐标
                ax.text(fake_letterbox_x, fake_letterbox_y - 10, 
                       f'Confusion #{i+1}\n({int(fake_x)}, {int(fake_y)})',
                       color='blue', fontsize=8, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', 
                                edgecolor='blue', alpha=0.7))
        
        # 显示padding边界（用青色虚线标记）
        padding_contour = np.where(padding_mask > 0.5, 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(padding_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:1]:  # 只显示第一个轮廓，避免过多干扰
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'c--', linewidth=1, alpha=0.5)
        
        # 更新标题
        title = 'Image with Overlay (Red=Gap, Green=Slider'
        if sample.get('fake_gaps'):
            title += f', Blue=Confusion Gaps[{len(sample["fake_gaps"])}]'
        title += ')'
        ax.set_title(title, color='white')
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
        
        # 步骤1: 计算letterbox空间(512×256)中的坐标
        gap_letterbox_x = 4 * (sample['gap_grid'][0] + 0.5 + sample['gap_offset'][0])
        gap_letterbox_y = 4 * (sample['gap_grid'][1] + 0.5 + sample['gap_offset'][1])
        slider_letterbox_x = 4 * (sample['slider_grid'][0] + 0.5 + sample['slider_offset'][0])
        slider_letterbox_y = 4 * (sample['slider_grid'][1] + 0.5 + sample['slider_offset'][1])
        
        # 准备对比信息
        comparison_text = ""
        if sample.get('original_labels') and sample.get('original_size'):
            orig_gap = sample['original_labels']['bg_gap_center']
            orig_slider = sample['original_labels']['comp_piece_center']
            
            # 步骤2: 计算letterbox变换参数
            # 原始图像尺寸从labels_by_pic.json读取
            orig_width = sample['original_size']['width']
            orig_height = sample['original_size']['height']
            
            # 计算letterbox padding (保持2:1宽高比)
            current_ratio = orig_width / orig_height
            target_ratio = 2.0  # 512/256 = 2
            
            if current_ratio > target_ratio:
                # 图像太宽，需要上下padding
                padded_width = orig_width
                padded_height = orig_width / target_ratio
                pad_left = 0
                pad_top = (padded_height - orig_height) / 2
            else:
                # 图像太高，需要左右padding
                padded_height = orig_height
                padded_width = orig_height * target_ratio
                pad_left = (padded_width - orig_width) / 2
                pad_top = 0
            
            # 计算缩放比例 (padded -> 512×256)
            scale = 512 / padded_width  # 或 256 / padded_height
            
            # 步骤3: 从letterbox空间转换回原始空间
            # 公式: orig = (letterbox - pad*scale) / scale
            gap_final_x = (gap_letterbox_x - pad_left * scale) / scale
            gap_final_y = (gap_letterbox_y - pad_top * scale) / scale
            slider_final_x = (slider_letterbox_x - pad_left * scale) / scale
            slider_final_y = (slider_letterbox_y - pad_top * scale) / scale
            
            # 计算像素差距
            gap_diff_x = abs(gap_final_x - orig_gap[0])
            gap_diff_y = abs(gap_final_y - orig_gap[1])
            gap_dist = (gap_diff_x**2 + gap_diff_y**2)**0.5
            
            slider_diff_x = abs(slider_final_x - orig_slider[0])
            slider_diff_y = abs(slider_final_y - orig_slider[1])
            slider_dist = (slider_diff_x**2 + slider_diff_y**2)**0.5
            
            comparison_text = f"""
╔══════════════════════════════════════════════════════════════╗
║              Coordinate Recovery Formula                     ║
╚══════════════════════════════════════════════════════════════╝

[Step 1] Grid → Letterbox (512×256)
  letterbox = 4 × (grid + 0.5 + offset)
  
  Gap:   {gap_letterbox_x:.1f} = 4×({sample['gap_grid'][0]}+0.5+{sample['gap_offset'][0]:.3f})
  Slider: {slider_letterbox_x:.1f} = 4×({sample['slider_grid'][0]}+0.5+{sample['slider_offset'][0]:.3f})

[Step 2] Letterbox → Original ({orig_width}×{orig_height})
  original = (letterbox - pad×scale) / scale
  
  Padding: left={pad_left:.1f}, top={pad_top:.1f}
  Scale: {scale:.4f}
  
  Gap X:   {gap_final_x:.1f} = ({gap_letterbox_x:.1f} - {pad_left:.1f}×{scale:.4f}) / {scale:.4f}
  Gap Y:   {gap_final_y:.1f} = ({gap_letterbox_y:.1f} - {pad_top:.1f}×{scale:.4f}) / {scale:.4f}

[Validation Results]
┌─────────────┬──────────────┬──────────────┬────────┐
│   Target    │  Original    │  Recovered   │ Error  │
├─────────────┼──────────────┼──────────────┼────────┤
│ Gap Center  │ ({orig_gap[0]:6.1f}, {orig_gap[1]:6.1f}) │ ({gap_final_x:6.1f}, {gap_final_y:6.1f}) │ {gap_dist:5.2f}px │
│ Slider      │ ({orig_slider[0]:6.1f}, {orig_slider[1]:6.1f}) │ ({slider_final_x:6.1f}, {slider_final_y:6.1f}) │ {slider_dist:5.2f}px │
└─────────────┴──────────────┴──────────────┴────────┘
"""
        else:
            comparison_text = """
Coordinate Comparison
==========================================
Original labels not found for this sample
"""
        
        # 添加混淆缺口信息
        confusion_text = ""
        if sample.get('fake_gaps'):
            confusion_text = f"\n[Confusion Gaps ({len(sample['fake_gaps'])} found)]\n"
            for i, fake_gap in enumerate(sample['fake_gaps'], 1):
                center_x, center_y = fake_gap['center']
                rotation = fake_gap.get('delta_theta_deg', 0)
                scale = fake_gap.get('scale', 1.0)
                confusion_text += f"  Gap #{i}: Center=({center_x:6.1f}, {center_y:6.1f})  "
                confusion_text += f"Rot={rotation:+6.1f}°  Scale={scale:.2f}\n"
        else:
            confusion_text = "\n[Confusion Gaps: None]\n"
        
        info_text = f"""Sample ID: {sample['sample_id']}

[Grid Coordinates (128×64)]
  Gap:    grid=({sample['gap_grid'][0]:3d}, {sample['gap_grid'][1]:3d})  offset=({sample['gap_offset'][0]:+.3f}, {sample['gap_offset'][1]:+.3f})
  Slider: grid=({sample['slider_grid'][0]:3d}, {sample['slider_grid'][1]:3d})  offset=({sample['slider_offset'][0]:+.3f}, {sample['slider_offset'][1]:+.3f})
{confusion_text}
{comparison_text}
Controls: ← → (Navigate) | PageUp/Down (±10) | Home/End (±100) | ESC (Exit)
"""
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top', fontfamily='monospace', color='white')
    
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
        elif event.key == 'b':
            # 切换到下一个批次
            self.switch_batch()
        elif event.key == 'B':
            # 切换到上一个批次
            self.switch_batch_prev()
    
    def jump_samples(self, count):
        """跳转指定数量的样本
        
        Args:
            count: 要跳转的样本数量（正数向前，负数向后）
        """
        # 在当前批次内跳转
        new_sample = self.current_sample + count
        
        # 处理跨批次的情况
        while new_sample >= self.total_samples_in_batch:
            new_sample -= self.total_samples_in_batch
            self.current_batch += 1
            if self.current_batch >= len(self.batch_files):
                self.current_batch = 0
            self.load_batch(self.current_batch)
            
        while new_sample < 0:
            self.current_batch -= 1
            if self.current_batch < 0:
                self.current_batch = len(self.batch_files) - 1
            self.load_batch(self.current_batch)
            new_sample += self.total_samples_in_batch
            
        self.current_sample = new_sample
        self.update_display()
    
    def next_sample(self):
        """切换到下一个样本"""
        self.current_sample += 1
        if self.current_sample >= self.total_samples_in_batch:
            # 自动切换到下一个批次
            self.current_sample = 0
            self.current_batch += 1
            if self.current_batch >= len(self.batch_files):
                self.current_batch = 0
            self.load_batch(self.current_batch)
        self.update_display()
    
    def switch_batch(self):
        """切换到下一个批次"""
        self.current_batch += 1
        if self.current_batch >= len(self.batch_files):
            self.current_batch = 0
        self.load_batch(self.current_batch)
        self.update_display()
    
    def switch_batch_prev(self):
        """切换到上一个批次"""
        self.current_batch -= 1
        if self.current_batch < 0:
            self.current_batch = len(self.batch_files) - 1
        self.load_batch(self.current_batch)
        self.update_display()
    
    def prev_sample(self):
        """切换到上一个样本"""
        self.current_sample -= 1
        if self.current_sample < 0:
            # 自动切换到上一个批次
            self.current_batch -= 1
            if self.current_batch < 0:
                self.current_batch = len(self.batch_files) - 1
            self.load_batch(self.current_batch)
            self.current_sample = self.total_samples_in_batch - 1
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
    print("  B                : Next batch")
    print("  Shift+B          : Previous batch")
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