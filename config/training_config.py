# -*- coding: utf-8 -*-
"""
训练配置管理
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    K: int = 24  # 方向分解通道数
    topk: int = 5  # TopK候选数
    enable_subpixel: bool = True  # 是否启用亚像素精修
    backbone: str = 'resnet18_lite'  # 骨干网络
    stride: int = 4  # 下采样倍数


@dataclass
class DataConfig:
    """数据配置"""
    train_label_file: str = 'data/captchas_with_labels/train_labels.json'
    val_label_file: str = 'data/captchas_with_labels/val_labels.json'
    test_label_file: str = 'data/captchas_with_labels/test_labels.json'
    root_dir: Optional[str] = None
    batch_size: int = 256
    num_workers: int = 12
    pin_memory: bool = True
    
    # 数据增强
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        'random_brightness': 0.0,
        'random_contrast': 0.0,
        'gaussian_noise': 0.0,
        'blur_prob': 0.0,
        'rotation_limit': 5
    })


@dataclass
class OptimizerConfig:
    """优化器配置"""
    name: str = 'AdamW'
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # 学习率调度
    scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', 'reduce_on_plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 100,  # cosine
        'eta_min': 1e-6,
        # 'step_size': 30,  # step
        # 'gamma': 0.1,
        # 'milestones': [30, 60, 90],  # multistep
        # 'factor': 0.5,  # reduce_on_plateau
        # 'patience': 10,
        # 'min_lr': 1e-7
    })


@dataclass
class LossConfig:
    """损失函数配置"""
    # 损失权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        'pose_loss': 1.0,  # 位姿损失
        'ce_loss': 1.0,    # 分类损失  
        'coord_loss': 5.0, # 坐标损失
        'topk_loss': 0.5,  # TopK损失
        'margin_loss': 0.1 # 边界损失
    })
    
    # 焦点损失参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 标签平滑
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """训练总配置"""
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # 训练参数
    epochs: int = 25
    gradient_clip: float = 1.0
    accumulation_steps: int = 4  # 梯度累积步数
    
    # 验证和保存
    val_interval: int = 1  # 每N个epoch验证一次
    save_interval: int = 1  # 每N个epoch保存一次
    save_best: bool = True  # 保存最佳模型
    early_stopping_patience: int = 20  # 早停耐心值
    
    # 路径配置
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    experiment_name: str = 'slider_captcha'
    
    # 设备配置
    device: str = 'cuda'  # 'cuda', 'cpu', 'mps'
    mixed_precision: bool = True  # 是否使用混合精度训练
    
    # 随机种子
    seed: int = 42
    deterministic: bool = True
    
    # 日志配置
    log_interval: int = 10  # 每N个batch记录一次
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'slider-captcha'
    
    def save(self, path: str):
        """保存配置到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 递归创建嵌套的dataclass
        if 'model' in data:
            data['model'] = ModelConfig(**data['model'])
        if 'data' in data:
            data['data'] = DataConfig(**data['data'])
        if 'optimizer' in data:
            data['optimizer'] = OptimizerConfig(**data['optimizer'])
        if 'loss' in data:
            data['loss'] = LossConfig(**data['loss'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 尝试更新子配置
                if '.' in key:
                    parts = key.split('.')
                    obj = self
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)


def create_default_config() -> TrainingConfig:
    """创建默认配置"""
    config = TrainingConfig()
    
    # 根据硬件自动调整
    import torch
    if torch.cuda.is_available():
        config.device = 'cuda'
        config.data.batch_size = 32
        config.mixed_precision = True
    elif torch.backends.mps.is_available():
        config.device = 'mps'
        config.data.batch_size = 16
        config.mixed_precision = False
    else:
        config.device = 'cpu'
        config.data.batch_size = 8
        config.mixed_precision = False
    
    return config


if __name__ == "__main__":
    # 测试配置
    config = create_default_config()
    
    # 保存配置
    config.save('configs/default_config.json')
    print("Saved default config to configs/default_config.json")
    
    # 加载配置
    loaded_config = TrainingConfig.load('configs/default_config.json')
    print("\nLoaded config:")
    print(f"  Model K: {loaded_config.model.K}")
    print(f"  Batch size: {loaded_config.data.batch_size}")
    print(f"  Learning rate: {loaded_config.optimizer.lr}")
    print(f"  Epochs: {loaded_config.epochs}")
    print(f"  Device: {loaded_config.device}")
    
    # 更新配置
    loaded_config.update(epochs=200, device='cuda')
    print(f"\nUpdated epochs: {loaded_config.epochs}")
    print(f"Updated device: {loaded_config.device}")