import torch
import torch.nn as nn
import numpy as np
import json
import cv2
import yaml
from pathlib import Path
import random
import matplotlib.pyplot as plt


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


# ==================== 测试函数 ====================
def load_model_and_config(checkpoint_dir):
    """加载模型和配置"""
    # 查找最新的训练目录
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        # 使用默认配置
        config = {
            'model_type': 'resnet18_lite',
            'num_classes': 2,
            'batch_size': 32,
            'data_dir': 'data',
            'annotations_file': 'data/captchas/annotations.json'
        }
        return CaptchaDetector(num_classes=2), config
    
    # 首先检查是否有日期子目录
    train_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith('20')]
    
    if train_dirs:
        # 使用最新的训练目录
        latest_dir = sorted(train_dirs)[-1]
        config_path = latest_dir / 'config.yaml'
        best_model_path = latest_dir / 'best.pth'
        if not best_model_path.exists():
            best_model_path = latest_dir / 'latest.pth'
    else:
        # 检查是否直接在checkpoint目录下有模型文件
        latest_dir = checkpoint_path
        config_path = checkpoint_path / 'config.yaml'
        best_model_path = checkpoint_path / 'best_model.pth'
        if not best_model_path.exists():
            best_model_path = checkpoint_path / 'latest_checkpoint.pth'
    
    # 加载配置
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Config file not found: {config_path}, using default config")
        config = {
            'model_type': 'resnet18_lite',
            'num_classes': 2,
            'batch_size': 32,
            'data_dir': 'data',
            'annotations_file': 'data/captchas/annotations.json'
        }
    
    # 创建模型
    model = CaptchaDetector(num_classes=2)
    
    # 加载权重
    if best_model_path.exists():
        print(f"Loading model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"No model checkpoint found at {best_model_path}, using random initialization")
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, config


def load_test_data(annotations_file, data_dir, num_samples=100):
    """加载测试数据"""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # 从测试目录获取所有实际存在的图片
    test_dir = Path(data_dir) / 'test'
    existing_files = set(f.name for f in test_dir.glob('*.png'))
    
    # 从annotations列表中找出测试集样本
    test_samples = []
    for sample in annotations:
        filename = sample['filename']
        if filename in existing_files:
            test_samples.append({
                'image_path': f'data/test/{filename}',
                'filename': filename,
                'bg_center': sample['bg_center'],
                'sd_center': sample['sd_center'],
                'shape': sample['shape'],
                'size': sample['size']
            })
    
    # 随机选择num_samples个样本
    if len(test_samples) > num_samples:
        test_samples = random.sample(test_samples, num_samples)
    
    print(f"Found {len(test_samples)} valid test samples")
    return test_samples


def preprocess_image(image_path, target_size=(320, 160)):
    """预处理图片"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    image = cv2.resize(image, target_size)
    
    # 归一化并转换为tensor
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    return image, image_tensor


def predict_single_image(model, image_tensor, config):
    """对单张图片进行预测"""
    with torch.no_grad():
        # 添加batch维度
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        image_tensor = image_tensor.unsqueeze(0)
        
        # 预测
        outputs = model(image_tensor)
        
        # 解码预测结果
        heatmaps = torch.sigmoid(outputs['hm'])  # 应用sigmoid获得概率
        offsets = outputs['reg']
        
        gap_pred = []
        piece_pred = []
        
        # 背景缺口中心
        hm_bg = heatmaps[0, 0].cpu().numpy()
        if hm_bg.max() > 0.1:
            y, x = np.unravel_index(hm_bg.argmax(), hm_bg.shape)
            # 背景缺口的偏移量在前2个通道
            offset_x = offsets[0, 0, y, x].cpu().numpy()
            offset_y = offsets[0, 1, y, x].cpu().numpy()
            center_x = (x + offset_x) * 4  # 恢复到原始尺寸
            center_y = (y + offset_y) * 4
            gap_pred = [(center_x, center_y)]
        
        # 滑块中心
        hm_slider = heatmaps[0, 1].cpu().numpy()
        if hm_slider.max() > 0.1:
            y, x = np.unravel_index(hm_slider.argmax(), hm_slider.shape)
            # 滑块的偏移量在后2个通道
            offset_x = offsets[0, 2, y, x].cpu().numpy()
            offset_y = offsets[0, 3, y, x].cpu().numpy()
            center_x = (x + offset_x) * 4  # 恢复到原始尺寸
            center_y = (y + offset_y) * 4
            piece_pred = [(center_x, center_y)]
        
        return gap_pred, piece_pred


def calculate_accuracy(pred_pos, gt_pos, threshold=10.0):
    """计算预测准确率"""
    if len(pred_pos) == 0:
        return False
    
    # 计算预测点与真实点的距离
    pred_x, pred_y = pred_pos[0]
    gt_x, gt_y = gt_pos
    
    distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    
    return distance <= threshold


def visualize_results(test_results, output_dir, num_figures=5, samples_per_figure=20):
    """可视化测试结果 - 生成多张图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 选择要显示的样本
    total_display = min(num_figures * samples_per_figure, len(test_results))
    display_samples = test_results[:total_display]
    
    # 生成多张可视化图
    for fig_idx in range(num_figures):
        start_idx = fig_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, len(display_samples))
        
        if start_idx >= len(display_samples):
            break
            
        # 创建当前图
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            result = display_samples[global_idx]
            ax = axes[local_idx]
            
            # 显示图片
            ax.imshow(result['image'])
            
            # 绘制真实位置
            gap_gt = result['gap_gt']
            piece_gt = result['piece_gt']
            
            # 绘制预测位置
            if result['gap_pred']:
                gap_pred = result['gap_pred'][0]
                # 绘制预测的缺口位置
                ax.scatter(gap_pred[0], gap_pred[1], color='yellow', s=100, marker='x', linewidths=3)
            
            if result['piece_pred']:
                piece_pred = result['piece_pred'][0]
                # 绘制预测的滑块位置
                ax.scatter(piece_pred[0], piece_pred[1], color='cyan', s=100, marker='o', linewidths=3)
            
            # 绘制真实位置参考（小一点的标记）
            ax.scatter(gap_gt[0], gap_gt[1], color='green', s=50, marker='+', linewidths=2, alpha=0.7)
            ax.scatter(piece_gt[0], piece_gt[1], color='blue', s=50, marker='+', linewidths=2, alpha=0.7)
            
            # 设置标题，显示正确/错误的标记
            gap_mark = "✓" if result['gap_correct'] else "✗"
            gap_color = 'green' if result['gap_correct'] else 'red'
            piece_mark = "✓" if result['piece_correct'] else "✗"
            piece_color = 'green' if result['piece_correct'] else 'red'
            
            # 创建标题
            ax.set_title(f"Sample {global_idx+1}", fontsize=12, pad=15)
            
            # 在标题下方添加Gap和Piece的标记（同一行）
            gap_text = f"Gap: {gap_mark}"
            piece_text = f"Piece: {piece_mark}"
            
            # 使用不同颜色的文本
            ax.text(0.3, 1.02, gap_text, transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=11, color=gap_color, weight='bold')
            ax.text(0.7, 1.02, piece_text, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=11, color=piece_color, weight='bold')
            
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(end_idx - start_idx, len(axes)):
            axes[idx].axis('off')
        
        # 添加图例
        fig.text(0.5, 0.02, 
                 'Yellow X: Gap Prediction | Cyan O: Piece Prediction | Green +: GT Gap | Blue +: GT Piece', 
                 ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        plt.savefig(output_dir / f'test_results_visualization_{fig_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization figure {fig_idx+1}/{num_figures}")
    
    # 保存统计结果
    gap_accuracy = sum(r['gap_correct'] for r in test_results) / len(test_results) * 100
    piece_accuracy = sum(r['piece_correct'] for r in test_results) / len(test_results) * 100
    
    stats_text = f"""Test Results Summary:
Total Samples: {len(test_results)}
Gap Detection Accuracy: {gap_accuracy:.2f}%
Piece Detection Accuracy: {piece_accuracy:.2f}%
"""
    
    with open(output_dir / 'test_results_summary.txt', 'w') as f:
        f.write(stats_text)
    
    return gap_accuracy, piece_accuracy


def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 路径设置
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / 'src' / 'checkpoints'
    
    print("Loading model and configuration...")
    model, config = load_model_and_config(checkpoint_dir)
    
    print("Loading test data...")
    annotations_file = project_root / 'data' / 'captchas' / 'annotations.json'
    data_dir = project_root / 'data'
    test_samples = load_test_data(annotations_file, data_dir, num_samples=1000)
    
    print(f"Testing on {len(test_samples)} samples...")
    
    test_results = []
    
    for i, sample in enumerate(test_samples):
        if (i + 1) % 20 == 0:
            print(f"Processing {i + 1}/{len(test_samples)}...")
        
        # 加载图片
        image_path = project_root / sample['image_path']
        image, image_tensor = preprocess_image(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # 预测
        gap_pred, piece_pred = predict_single_image(model, image_tensor, config)
        
        # 获取真实位置
        gap_gt = tuple(sample['bg_center'])
        piece_gt = tuple(sample['sd_center'])
        
        # 计算准确率
        gap_correct = calculate_accuracy(gap_pred, gap_gt, threshold=10.0)
        piece_correct = calculate_accuracy(piece_pred, piece_gt, threshold=10.0)
        
        test_results.append({
            'image': image,
            'gap_gt': gap_gt,
            'piece_gt': piece_gt,
            'gap_pred': gap_pred,
            'piece_pred': piece_pred,
            'gap_correct': gap_correct,
            'piece_correct': piece_correct
        })
    
    # 可视化结果
    print("\nGenerating visualization...")
    gap_acc, piece_acc = visualize_results(test_results, project_root / 'results' / 'test_results', num_figures=5, samples_per_figure=20)
    
    print(f"\nTest Results:")
    print(f"Gap Detection Accuracy: {gap_acc:.2f}%")
    print(f"Piece Detection Accuracy: {piece_acc:.2f}%")
    print(f"\nResults saved to: {project_root / 'results' / 'test_results'}")


if __name__ == "__main__":
    main()