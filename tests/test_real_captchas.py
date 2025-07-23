"""
使用最优模型测试真实验证码数据
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import random
import json


# ==================== 模型定义 ====================
class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18Lite(nn.Module):
    """轻量级ResNet18骨干网络"""
    
    def __init__(self, in_channels=3):
        super(ResNet18Lite, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
    
    def _make_layer(self, in_planes, out_planes, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class CaptchaDetector(nn.Module):
    """滑块验证码检测模型"""
    
    def __init__(self, num_classes=2):
        super(CaptchaDetector, self).__init__()
        
        # 骨干网络
        self.backbone = ResNet18Lite()
        
        # 上采样层
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4]
        )
        
        # 检测头
        self.heads = nn.ModuleDict({
            'hm': self._make_head(64, num_classes),  # 热力图（2个类别）
            'reg': self._make_head(64, 4),  # 偏移回归（每个类别2个通道，共4个）
        })
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_channels = 512
        
        for i in range(num_layers):
            kernel = num_kernels[i]
            filters = num_filters[i]
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=2,
                    padding=kernel // 2 - 1,
                    output_padding=0,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))
            in_channels = filters
        
        return nn.Sequential(*layers)
    
    def _make_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
    
    def forward(self, x):
        # 通过骨干网络
        x = self.backbone(x)
        
        # 上采样
        x = self.deconv_layers(x)
        
        # 生成预测
        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(x)
        
        return outputs


def load_model_and_config():
    """加载最优模型和配置"""
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / 'src' / 'checkpoints'
    
    # 加载配置 - 尝试从checkpoint文件中获取
    best_checkpoint_path = checkpoint_dir / 'best_model.pth'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # 使用默认配置
            config = {
                'model_type': 'resnet18_lite',
                'num_classes': 2,
                'data_dir': 'data',
                'annotations_file': 'data/captchas/annotations.json'
            }
    else:
        print(f"Checkpoint not found: {best_checkpoint_path}")
        config = {
            'model_type': 'resnet18_lite',
            'num_classes': 2,
            'data_dir': 'data',
            'annotations_file': 'data/captchas/annotations.json'
        }
        checkpoint = None
    
    # 创建模型
    model = CaptchaDetector(num_classes=2)
    
    # 加载权重
    if checkpoint and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {best_checkpoint_path}")
    else:
        print("Using randomly initialized model")
    
    model.eval()
    
    print("Loaded best model")
    
    return model, config


def predict_single_image(model, image_path, device='cpu'):
    """预测单张图片"""
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None
    
    # 转换颜色通道
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 确保尺寸正确
    if image.shape[:2] != (160, 320):
        image = cv2.resize(image, (320, 160))
    
    # 预处理
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 解码热力图
    heatmaps = torch.sigmoid(outputs['hm'])
    offsets = outputs['reg']
    
    # 解码缺口位置
    hm_bg = heatmaps[0, 0].cpu().numpy()
    if hm_bg.max() > 0.05:  # 降低阈值到0.05
        y, x = np.unravel_index(hm_bg.argmax(), hm_bg.shape)
        offset_x = offsets[0, 0, y, x].cpu().numpy()
        offset_y = offsets[0, 1, y, x].cpu().numpy()
        gap_center = ((x + offset_x) * 4, (y + offset_y) * 4)
    else:
        gap_center = None
    
    # 解码滑块位置
    hm_slider = heatmaps[0, 1].cpu().numpy()
    if hm_slider.max() > 0.05:  # 降低阈值到0.05
        y, x = np.unravel_index(hm_slider.argmax(), hm_slider.shape)
        offset_x = offsets[0, 2, y, x].cpu().numpy()
        offset_y = offsets[0, 3, y, x].cpu().numpy()
        piece_center = ((x + offset_x) * 4, (y + offset_y) * 4)
    else:
        piece_center = None
    
    return image, gap_center, piece_center


def visualize_results(results, output_dir, site_name, num_figures=5, samples_per_figure=20):
    """可视化测试结果 - 生成多张图，每张4x5=20个验证码"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置中文字体
    font_path = "C:/Windows/Fonts/msyh.ttc"
    if Path(font_path).exists():
        font_prop = font_manager.FontProperties(fname=font_path, size=10)
        plt.rcParams['font.family'] = ['Microsoft YaHei']
    
    # 每张图显示20个验证码（4行5列）
    rows, cols = 4, 5
    total_samples = min(len(results), num_figures * samples_per_figure)
    
    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{site_name} Real CAPTCHA Detection Results - Figure {fig_idx + 1}', fontsize=16)
        
        for idx in range(samples_per_figure):
            result_idx = fig_idx * samples_per_figure + idx
            if result_idx >= total_samples:
                # 隐藏多余的子图
                row, col = idx // cols, idx % cols
                axes[row, col].axis('off')
                continue
            
            result = results[result_idx]
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            # 显示图片
            ax.imshow(result['image'])
            
            # 绘制检测结果
            if result['gap_center'] is not None:
                gap_x, gap_y = result['gap_center']
                # 绘制缺口中心点和边框
                ax.plot(gap_x, gap_y, 'ro', markersize=8)
                rect = patches.Rectangle((gap_x-30, gap_y-30), 60, 60, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            
            if result['piece_center'] is not None:
                piece_x, piece_y = result['piece_center']
                # 绘制滑块中心点和边框
                ax.plot(piece_x, piece_y, 'go', markersize=8)
                rect = patches.Rectangle((piece_x-30, piece_y-30), 60, 60, 
                                       linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            
            # 设置标题
            gap_status = "✓" if result['gap_center'] is not None else "✗"
            piece_status = "✓" if result['piece_center'] is not None else "✗"
            title = f"Gap: {gap_status}, Piece: {piece_status}"
            
            # 根据检测结果设置标题颜色
            title_color = 'green' if gap_status == "✓" and piece_status == "✓" else 'red'
            ax.set_title(title, fontsize=10, color=title_color)
            
            ax.axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f'{site_name}_detection_results_{fig_idx + 1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {output_path}")


def test_real_captchas():
    """测试真实验证码"""
    # 加载模型
    print("Loading model...")
    model, config = load_model_and_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    real_captchas_dir = project_root / "data" / "real_captchas" / "merged"
    output_dir = project_root / "results" / "real_captcha_results"
    output_dir.mkdir(exist_ok=True)
    
    # 测试每个站点
    for site_name in ['site1']:
        site_dir = real_captchas_dir / site_name
        if not site_dir.exists():
            print(f"Site directory not found: {site_dir}")
            continue
        
        # 获取所有合并后的图片
        image_files = sorted(site_dir.glob("*_merged.png"))
        print(f"\nFound {len(image_files)} images in {site_name}")
        
        # 随机抽取100张
        num_samples = min(100, len(image_files))
        sampled_files = random.sample(image_files, num_samples)
        
        # 测试每张图片
        results = []
        detection_stats = {'gap_detected': 0, 'piece_detected': 0, 'both_detected': 0}
        
        print(f"Testing {num_samples} images from {site_name}...")
        for img_path in sampled_files:
            image, gap_center, piece_center = predict_single_image(model, img_path, device)
            
            if image is not None:
                results.append({
                    'image': image,
                    'gap_center': gap_center,
                    'piece_center': piece_center,
                    'filename': img_path.name
                })
                
                # 统计检测结果
                if gap_center is not None:
                    detection_stats['gap_detected'] += 1
                if piece_center is not None:
                    detection_stats['piece_detected'] += 1
                if gap_center is not None and piece_center is not None:
                    detection_stats['both_detected'] += 1
        
        # 打印统计结果
        print(f"\n{site_name} Detection Statistics:")
        print(f"  - Gap detected: {detection_stats['gap_detected']}/{num_samples} ({detection_stats['gap_detected']/num_samples*100:.1f}%)")
        print(f"  - Piece detected: {detection_stats['piece_detected']}/{num_samples} ({detection_stats['piece_detected']/num_samples*100:.1f}%)")
        print(f"  - Both detected: {detection_stats['both_detected']}/{num_samples} ({detection_stats['both_detected']/num_samples*100:.1f}%)")
        
        # 生成可视化结果（5张图，每张20个验证码）
        if results:
            visualize_results(results, output_dir, site_name, num_figures=5, samples_per_figure=20)
        
        # 保存统计结果
        stats_file = output_dir / f"{site_name}_detection_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(detection_stats, f, indent=2)


if __name__ == "__main__":
    test_real_captchas()