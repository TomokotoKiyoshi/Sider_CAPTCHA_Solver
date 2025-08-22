#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 解决OpenMP冲突问题 - 必须在导入numpy/torch之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
更新 BatchNorm 统计量脚本 - 仿照 predictor.py 风格
使用与 predictor.py 相同的前向传播方式更新 BN 统计量

使用方法：
python scripts/training/update_bn_predictor_style.py
"""

import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import logging
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.lite_hrnet_18_fpn import create_lite_hrnet_18_fpn
from src.preprocessing.preprocessor import LetterboxTransform


class BNUpdater:
    """BatchNorm 统计量更新器 - 仿照 predictor.py 风格"""
    
    def __init__(self, config_path: str = None):
        """初始化 BN 更新器
        
        Args:
            config_path: 配置文件路径，默认使用 updating_BN.yaml
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设备配置
        self.device = self._setup_device(self.config['bn_update']['device'])
        
        # 初始化预处理器 - 与 predictor.py 保持一致
        self.letterbox = LetterboxTransform(
            target_size=(512, 256),  # 宽x高
            fill_value=255  # 白色填充
        )
        
        # 设置输入通道模式
        self.input_channels = self.config['bn_update']['input_channels']
        
        # 加载模型 - 始终创建默认模型，然后调整权重
        self.model = create_lite_hrnet_18_fpn()
        
        self._load_checkpoint()
        
        self.model.to(self.device)
        
        # 收集 BN 层
        self.bn_layers = self._collect_bn_layers()
        
        logging.info(f"✅ BN 更新器初始化成功 | 设备: {self.device} | 输入通道: {self.input_channels}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = project_root / 'config' / 'updating_BN.yaml'
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"加载配置文件: {config_path}")
        return config
    
    def _setup_device(self, device: str) -> torch.device:
        """设置运行设备 - 与 predictor.py 保持一致"""
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA 不可用，使用 CPU")
            return torch.device('cpu')
        return torch.device(device)
    
    def _load_checkpoint(self):
        """加载模型检查点 - 与 predictor.py 保持一致"""
        checkpoint_path = project_root / self.config['paths']['input_checkpoint']
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到检查点: {checkpoint_path}")
        
        logging.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载状态字典（模型架构固定为2通道，所以不需要调整权重）
        self.model.load_state_dict(state_dict, strict=True)
    
    def _collect_bn_layers(self) -> List:
        """收集 stem/stage2 的 BN 层
        
        注：Lite-HRNet-18 模型结构：
        - stem: 初始特征提取（Stage1）
        - stage2: 双分支结构
        - stage3: 三分支结构（不更新）
        - stage4: 四分支结构（不更新）
        - lite_fpn: 特征金字塔（不更新）
        - dual_head: 预测头（不更新）
        """
        bn_layers = []
        target_prefixes = ['stem.', 'stage2.']  # 只更新早期层的BN
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                # 检查是否是目标模块的BN层
                if any(name.startswith(prefix) for prefix in target_prefixes):
                    bn_layers.append((name, module))
                    logging.info(f"  收集 BN 层: {name}")
        
        logging.info(f"找到 {len(bn_layers)} 个目标 BN 层 (仅 stem/stage2)")
        return bn_layers
    
    def _preprocess_rgb(self, image_path: Union[str, Path]) -> torch.Tensor:
        """RGB 图像预处理 - 仿照 predictor.py 的 _preprocess 方法"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用Letterbox变换 - 等比缩放 + 居中填充
        image_letterboxed, transform_params = self.letterbox.apply(image)
        
        # 无论配置如何，都转换为2通道格式（因为模型架构是固定的）
        # 生成valid mask
        valid_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(image_letterboxed).float().permute(2, 0, 1) / 255.0
        valid_mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0)
        
        # 转换为灰度图
        gray_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        gray_tensor = gray_tensor.unsqueeze(0)
        
        # 组合输入 [Grayscale(1) + Valid_Mask(1)]
        image_tensor = torch.cat([gray_tensor, valid_mask_tensor], dim=0)
        
        return image_tensor
    
    def reset_bn_statistics(self):
        """重置所有 BN 层的统计量"""
        if self.config['bn_update']['reset_stats']:
            logging.info("重置 BN 统计量...")
            for name, bn in self.bn_layers:
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0)
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked.zero_()
    
    def update_bn_statistics(self):
        """使用数据更新 BN 统计量 - 仿照 predictor.py 的前向传播方式"""
        # 准备数据
        data_dir = project_root / self.config['paths']['rgb_data_dir']
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 收集所有图片路径
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(sorted(data_dir.glob(ext)))
        
        if not image_paths:
            raise FileNotFoundError(f"未找到图片: {data_dir}")
        
        logging.info(f"找到 {len(image_paths)} 张图片")
        
        # 限制样本数量
        num_samples = self.config['bn_update']['num_samples']
        if num_samples is not None and str(num_samples).lower() != 'none':
            try:
                num_samples = int(num_samples)
                if num_samples > 0:
                    image_paths = image_paths[:num_samples]
                    logging.info(f"使用前 {len(image_paths)} 张图片")
            except:
                pass  # 使用所有图片
        
        # 设置 BN 动量
        momentum = self.config['bn_update']['momentum']
        for name, bn in self.bn_layers:
            bn.momentum = momentum
        
        # 设置模型为训练模式（仅更新 BN 统计量）
        self.model.train()
        
        # 批处理更新
        batch_size = self.config['bn_update']['batch_size']
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        logging.info("开始更新 BN 统计量...")
        
        with torch.no_grad():  # 不计算梯度，仅更新 BN 统计量
            for batch_idx in tqdm(range(num_batches), desc="更新 BN"):
                # 获取当前批次的图片路径
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                # 预处理批次数据
                batch_tensors = []
                for image_path in batch_paths:
                    try:
                        # 使用与 predictor.py 类似的预处理
                        image_tensor = self._preprocess_rgb(image_path)
                        batch_tensors.append(image_tensor)
                    except Exception as e:
                        logging.warning(f"跳过图片 {image_path}: {e}")
                        continue
                
                if not batch_tensors:
                    continue
                
                # 堆叠为批次
                batch = torch.stack(batch_tensors).to(self.device)
                
                # 前向传播 - 与 predictor.py 的 predict 方法类似
                # 这会自动更新 BN 的 running_mean 和 running_var
                outputs = self.model(batch)
                
                # 注意：不需要调用 decode_predictions，
                # 因为我们只关心 BN 统计量的更新
        
        # 设置回评估模式
        self.model.eval()
        
        logging.info("BN 统计量更新完成！")
    
    def print_bn_statistics(self):
        """打印 BN 统计量摘要"""
        print("\n" + "="*60)
        print("BatchNorm 统计量摘要 (仅 stem/stage2)")
        print("="*60)
        
        # 打印所有被更新的BN层（因为数量有限）
        for name, bn in self.bn_layers:
            mean = bn.running_mean
            var = bn.running_var
            
            print(f"\n{name}:")
            print(f"  Running mean: min={mean.min():.6f}, max={mean.max():.6f}, mean={mean.mean():.6f}")
            print(f"  Running var:  min={var.min():.6f}, max={var.max():.6f}, mean={var.mean():.6f}")
            
            if bn.num_batches_tracked is not None:
                print(f"  Batches tracked: {bn.num_batches_tracked.item()}")
    
    def save_model(self):
        """保存更新后的模型"""
        output_path = project_root / self.config['paths']['output_checkpoint']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的字典
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'bn_updated': True,
            'bn_update_config': {
                'num_samples': self.config['bn_update']['num_samples'],
                'momentum': self.config['bn_update']['momentum'],
                'reset_stats': self.config['bn_update']['reset_stats'],
                'data_dir': str(self.config['paths']['rgb_data_dir']),
                'input_channels': self.input_channels,
                'update_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 保存
        torch.save(save_dict, output_path)
        logging.info(f"模型已保存到: {output_path}")
    
    def run(self):
        """执行完整的 BN 更新流程"""
        start_time = time.time()
        
        # 1. 打印原始统计量
        print("\n原始 BN 统计量:")
        self.print_bn_statistics()
        
        # 2. 重置统计量（如果配置）
        self.reset_bn_statistics()
        
        # 3. 更新 BN 统计量
        self.update_bn_statistics()
        
        # 4. 打印更新后的统计量
        print("\n更新后的 BN 统计量:")
        self.print_bn_statistics()
        
        # 5. 保存模型
        self.save_model()
        
        elapsed_time = time.time() - start_time
        logging.info(f"✅ BN 更新完成！总用时: {elapsed_time:.2f} 秒")


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建 BN 更新器并运行
    updater = BNUpdater()
    updater.run()


if __name__ == '__main__':
    main()