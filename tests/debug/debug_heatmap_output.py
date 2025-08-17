"""
调试脚本：输出模型的热力图和偏移量
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.lite_hrnet_18_fpn import create_lite_hrnet_18_fpn
from src.preprocessing.preprocessor import LetterboxTransform

def visualize_heatmap_and_offset():
    """可视化热力图和偏移量"""
    
    # 1. 加载图片
    image_path = r"D:\Hacker\Sider_CAPTCHA_Solver\data\real_captchas\merged\site2\1.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"原始图像尺寸: {image.shape[:2]}")
    
    # 2. Letterbox变换
    letterbox = LetterboxTransform(target_size=(512, 256), fill_value=255)
    image_transformed, transform_params = letterbox.apply(image)
    valid_mask = letterbox.create_padding_mask(transform_params)  # 1=有效，0=padding
    
    print(f"\n变换参数:")
    print(f"  scale={transform_params['scale']:.3f}")
    print(f"  pad_left={transform_params['pad_left']}")
    print(f"  pad_top={transform_params['pad_top']}")
    print(f"  pad_right={transform_params['pad_right']}")
    print(f"  pad_bottom={transform_params['pad_bottom']}")
    
    # 3. 创建4通道输入
    image_tensor = torch.from_numpy(image_transformed).float().permute(2, 0, 1) / 255.0
    valid_mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0)
    
    input_tensor = torch.cat([image_tensor, valid_mask_tensor], dim=0)
    input_tensor = input_tensor.unsqueeze(0).cuda()  # [1, 4, 256, 512]
    
    print(f"\n第4通道（valid mask）统计:")
    print(f"  min={input_tensor[0, 3].min():.2f}")
    print(f"  max={input_tensor[0, 3].max():.2f}")
    print(f"  mean={input_tensor[0, 3].mean():.2f}")
    
    # 4. 下采样valid mask到1/4分辨率
    valid_mask_input = input_tensor[:, 3:4, :, :]  # [1, 1, 256, 512]
    
    # 使用最小池化
    valid_mask_1_4 = -F.max_pool2d(-valid_mask_input, kernel_size=4, stride=4).squeeze(1)  # [1, 64, 128]
    padding_mask_1_4 = 1 - valid_mask_1_4  # padding=1, valid=0
    
    print(f"\n1/4分辨率valid mask统计:")
    print(f"  shape={valid_mask_1_4.shape}")
    print(f"  min={valid_mask_1_4.min():.2f}")
    print(f"  max={valid_mask_1_4.max():.2f}")
    print(f"  padding区域数量={(padding_mask_1_4 > 0.5).sum().item()}")
    
    # 5. 加载模型
    model = create_lite_hrnet_18_fpn()
    model.cuda()
    model.eval()
    
    checkpoint_path = project_root / "src" / "checkpoints" / "1.1.0" / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print("\n✅ 模型加载成功")
    
    # 6. 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 获取热力图和偏移量
        gap_heatmap = outputs['heatmap_gap'][0, 0].cpu().numpy()      # [64, 128]
        slider_heatmap = outputs['heatmap_slider'][0, 0].cpu().numpy() # [64, 128]
        gap_offset = outputs['offset_gap'][0].cpu().numpy()            # [2, 64, 128]
        slider_offset = outputs['offset_slider'][0].cpu().numpy()      # [2, 64, 128]
        
        print(f"\n模型输出统计:")
        print(f"  gap_heatmap: min={gap_heatmap.min():.4f}, max={gap_heatmap.max():.4f}")
        print(f"  slider_heatmap: min={slider_heatmap.min():.4f}, max={slider_heatmap.max():.4f}")
        print(f"  gap_offset_x: min={gap_offset[0].min():.4f}, max={gap_offset[0].max():.4f}")
        print(f"  gap_offset_y: min={gap_offset[1].min():.4f}, max={gap_offset[1].max():.4f}")
        
        # 找到原始峰值位置（未应用mask）
        gap_peak_idx_orig = np.unravel_index(gap_heatmap.argmax(), gap_heatmap.shape)
        slider_peak_idx_orig = np.unravel_index(slider_heatmap.argmax(), slider_heatmap.shape)
        
        print(f"\n【原始峰值位置】（未应用mask）:")
        print(f"  Gap: row={gap_peak_idx_orig[0]}, col={gap_peak_idx_orig[1]}, 值={gap_heatmap[gap_peak_idx_orig]:.4f}")
        print(f"  Slider: row={slider_peak_idx_orig[0]}, col={slider_peak_idx_orig[1]}, 值={slider_heatmap[slider_peak_idx_orig]:.4f}")
        
        # 检查原始峰值是否在padding区域
        padding_mask_np = padding_mask_1_4[0].cpu().numpy()
        print(f"\n原始峰值的mask状态:")
        print(f"  Gap位置 mask值: {padding_mask_np[gap_peak_idx_orig]:.2f} (1=padding, 0=valid)")
        print(f"  Slider位置 mask值: {padding_mask_np[slider_peak_idx_orig]:.2f}")
        
        if padding_mask_np[gap_peak_idx_orig] > 0.5:
            print(f"  ⚠️ Gap原始峰值在padding区域！")
        if padding_mask_np[slider_peak_idx_orig] > 0.5:
            print(f"  ⚠️ Slider原始峰值在padding区域！")
        
        # 应用mask后找峰值
        gap_heatmap_masked = np.copy(gap_heatmap)
        slider_heatmap_masked = np.copy(slider_heatmap)
        gap_heatmap_masked[padding_mask_np > 0.5] = -np.inf
        slider_heatmap_masked[padding_mask_np > 0.5] = -np.inf
        
        gap_peak_idx_masked = np.unravel_index(np.nanargmax(gap_heatmap_masked), gap_heatmap_masked.shape)
        slider_peak_idx_masked = np.unravel_index(np.nanargmax(slider_heatmap_masked), slider_heatmap_masked.shape)
        
        print(f"\n【Masked后峰值位置】（应用mask后）:")
        print(f"  Gap: row={gap_peak_idx_masked[0]}, col={gap_peak_idx_masked[1]}, 值={gap_heatmap[gap_peak_idx_masked]:.4f}")
        print(f"  Slider: row={slider_peak_idx_masked[0]}, col={slider_peak_idx_masked[1]}, 值={slider_heatmap[slider_peak_idx_masked]:.4f}")
        
        # 获取Masked后位置的偏移量
        gap_offset_x_masked = gap_offset[0][gap_peak_idx_masked]
        gap_offset_y_masked = gap_offset[1][gap_peak_idx_masked]
        slider_offset_x_masked = slider_offset[0][slider_peak_idx_masked]
        slider_offset_y_masked = slider_offset[1][slider_peak_idx_masked]
        
        print(f"\nMasked位置的偏移量:")
        print(f"  Gap: dx={gap_offset_x_masked:.4f}, dy={gap_offset_y_masked:.4f}")
        print(f"  Slider: dx={slider_offset_x_masked:.4f}, dy={slider_offset_y_masked:.4f}")
        
        # 计算Masked后的最终坐标
        gap_x_manual = (gap_peak_idx_masked[1] + 0.5 + gap_offset_x_masked) * 4
        gap_y_manual = (gap_peak_idx_masked[0] + 0.5 + gap_offset_y_masked) * 4
        slider_x_manual = (slider_peak_idx_masked[1] + 0.5 + slider_offset_x_masked) * 4
        slider_y_manual = (slider_peak_idx_masked[0] + 0.5 + slider_offset_y_masked) * 4
        
        print(f"\n【手动计算】网络空间坐标（512×256）:")
        print(f"  Gap: x={gap_x_manual:.2f}, y={gap_y_manual:.2f}")
        print(f"  Slider: x={slider_x_manual:.2f}, y={slider_y_manual:.2f}")
        
        # 检查是否在有效区域
        valid_left = transform_params['pad_left']
        valid_right = 512 - transform_params['pad_right']
        print(f"\n有效区域: [{valid_left}, {valid_right}]")
        
        if gap_x_manual < valid_left or gap_x_manual > valid_right:
            print(f"  ❌ Gap仍在padding区域!")
        else:
            print(f"  ✅ Gap在有效区域内")
            
        if slider_x_manual < valid_left or slider_x_manual > valid_right:
            print(f"  ❌ Slider在padding区域!")
        else:
            print(f"  ✅ Slider在有效区域内")
        
        # 7. 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 原图和mask
        axes[0, 0].imshow(image_transformed)
        axes[0, 0].set_title('Letterboxed Image')
        axes[0, 0].axvline(transform_params['pad_left'], color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(512 - transform_params['pad_right'], color='r', linestyle='--', alpha=0.5)
        
        axes[0, 1].imshow(valid_mask, cmap='gray')
        axes[0, 1].set_title('Valid Mask (1=valid, 0=padding)')
        
        # 1/4分辨率mask
        axes[0, 2].imshow(valid_mask_1_4[0].cpu().numpy(), cmap='gray')
        axes[0, 2].set_title('Valid Mask 1/4')
        
        axes[0, 3].imshow(padding_mask_np, cmap='gray')
        axes[0, 3].set_title('Padding Mask 1/4 (1=padding)')
        
        # 热力图
        axes[1, 0].imshow(gap_heatmap, cmap='hot')
        axes[1, 0].plot(gap_peak_idx_orig[1], gap_peak_idx_orig[0], 'r*', markersize=10, label='Original Peak')
        axes[1, 0].plot(gap_peak_idx_masked[1], gap_peak_idx_masked[0], 'g*', markersize=10, label='Masked Peak')
        axes[1, 0].legend()
        axes[1, 0].set_title(f'Gap Heatmap')
        
        axes[1, 1].imshow(slider_heatmap, cmap='hot')
        axes[1, 1].plot(slider_peak_idx_orig[1], slider_peak_idx_orig[0], 'r*', markersize=10, label='Original Peak')
        axes[1, 1].plot(slider_peak_idx_masked[1], slider_peak_idx_masked[0], 'g*', markersize=10, label='Masked Peak')
        axes[1, 1].legend()
        axes[1, 1].set_title(f'Slider Heatmap')
        
        # 偏移量
        axes[1, 2].imshow(gap_offset[0], cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 2].plot(gap_peak_idx_masked[1], gap_peak_idx_masked[0], 'g*', markersize=10)
        axes[1, 2].set_title(f'Gap Offset X (masked peak: {gap_offset_x_masked:.3f})')
        
        axes[1, 3].imshow(slider_offset[0], cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 3].plot(slider_peak_idx_masked[1], slider_peak_idx_masked[0], 'g*', markersize=10)
        axes[1, 3].set_title(f'Slider Offset X (masked peak: {slider_offset_x_masked:.3f})')
        
        plt.tight_layout()
        output_path = Path(__file__).parent / 'debug_heatmap_visualization.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n✅ 可视化已保存为 {output_path}")
        plt.show()
        
        # 8. 使用decode_predictions验证
        print("\n" + "="*60)
        print("【模型decode_predictions】自动解码:")
        decoded = model.decode_predictions(outputs, input_images=input_tensor)
        
        gap_coords = decoded['gap_coords'][0].cpu().numpy()
        slider_coords = decoded['slider_coords'][0].cpu().numpy()
        gap_score = decoded['gap_score'][0].cpu().item()
        slider_score = decoded['slider_score'][0].cpu().item()
        
        print(f"  Gap: x={gap_coords[0]:.2f}, y={gap_coords[1]:.2f}, 置信度={gap_score:.4f}")
        print(f"  Slider: x={slider_coords[0]:.2f}, y={slider_coords[1]:.2f}, 置信度={slider_score:.4f}")
        
        # 对比手动计算和自动解码
        print(f"\n【对比】手动 vs 自动:")
        print(f"  Gap X差异: {abs(gap_x_manual - gap_coords[0]):.2f}px")
        print(f"  Gap Y差异: {abs(gap_y_manual - gap_coords[1]):.2f}px")
        print(f"  Slider X差异: {abs(slider_x_manual - slider_coords[0]):.2f}px")
        print(f"  Slider Y差异: {abs(slider_y_manual - slider_coords[1]):.2f}px")
        
        if abs(gap_x_manual - gap_coords[0]) < 1 and abs(slider_x_manual - slider_coords[0]) < 1:
            print("  ✅ 手动计算与自动解码一致！")
        else:
            print("  ⚠️ 存在差异，请检查mask逻辑")
        
        # 转换到原始空间
        from src.preprocessing.preprocessor import CoordinateTransform
        coord_transform = CoordinateTransform(downsample=4)
        
        gap_orig = coord_transform.input_to_original((gap_coords[0], gap_coords[1]), transform_params)
        slider_orig = coord_transform.input_to_original((slider_coords[0], slider_coords[1]), transform_params)
        
        print(f"\n原始图像空间坐标:")
        print(f"  gap: x={gap_orig[0]:.2f}, y={gap_orig[1]:.2f}")
        print(f"  slider: x={slider_orig[0]:.2f}, y={slider_orig[1]:.2f}")
        print(f"  滑动距离: {gap_orig[0] - slider_orig[0]:.2f}px")


if __name__ == "__main__":
    visualize_heatmap_and_offset()