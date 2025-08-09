# 滑块验证码识别训练系统设计

## 1. 系统架构总览

### 1.1 设计原则
- **模块化设计**: 各组件独立可测试，便于维护
- **配置驱动**: 通过YAML配置文件管理超参数
- **可扩展性**: 支持新增损失函数、数据增强策略
- **实验管理**: 集成TensorBoard/WandB进行实验跟踪

### 1.2 技术栈
- **深度学习框架**: PyTorch 2.0+
- **配置管理**: Hydra/OmegaConf
- **实验跟踪**: TensorBoard + WandB (可选)
- **数据处理**: NumPy, OpenCV, Albumentations
- **加速优化**: Mixed Precision (AMP), DataParallel/DDP

## 2. 数据管道设计

### 2.1 数据集类
```python
class SliderCaptchaDataset(Dataset):
    """
    负责加载三路输入：拼图、背景、合成图
    """
    def __init__(self, label_file, img_dir, transform=None):
        # 加载标签
        # 建立图片索引
        # 初始化变换
        pass
    
    def __getitem__(self, idx):
        # 返回: {
        #   'piece': tensor(3, H_p, W_p),
        #   'background': tensor(3, 160, 320),
        #   'composite': tensor(3, 160, 320),
        #   'labels': {...}
        # }
        pass
```

### 2.2 数据增强策略
```python
class CaptchaAugmentation:
    """
    验证码专用增强策略
    """
    def __init__(self, config):
        self.transforms = A.Compose([
            # 光照变化
            A.RandomBrightnessContrast(p=0.3),
            # 噪声添加
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            # JPEG压缩
            A.ImageCompression(quality_lower=70, p=0.3),
            # 模糊
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
        ])
```

### 2.3 数据加载器配置
```yaml
dataloader:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  sampler:
    type: "balanced"  # 平衡不同拼图尺寸
```

## 3. 模型架构实现

### 3.1 骨干网络
```python
class SliderBackbone(nn.Module):
    """
    改进的ResNet18骨干，使用BlurPool抗锯齿
    """
    def __init__(self):
        super().__init__()
        # 替换maxpool为BlurPool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BlurPool2d(stride=2)  # 抗锯齿池化
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
```

### 3.2 特征金字塔网络
```python
class FPN(nn.Module):
    """
    只取P2层(stride=4)用于相关计算
    """
    def __init__(self, in_channels=[64, 128], out_channel=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # 构建FPN层
```

### 3.3 方向分解头
```python
class OrientationHead(nn.Module):
    """
    K通道方向分解，使用Softmax归一化
    """
    def __init__(self, in_channels=128, K=24):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.head = nn.Conv2d(in_channels, K, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        ori = self.head(x)
        ori = F.softmax(ori, dim=1)  # 通道维度Softmax
        return ori
```

### 3.4 相关计算模块
```python
class CorrelationModule(nn.Module):
    """
    实现2D同向相关和1D互补相关
    """
    def forward_2d_correlation(self, ori_piece, ori_composite):
        # 同向相关: k <-> k
        pass
    
    def forward_1d_correlation(self, ori_piece, ori_background):
        # 互补相关: k <-> k+K/2
        pass
```

## 4. 损失函数设计

### 4.1 复合损失函数
```python
class CompositeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.margin_loss = MarginLoss(margin=0.2)
        self.chamfer_loss = OrientedChamferLoss()
        
        # 损失权重
        self.weights = config.loss_weights
    
    def forward(self, predictions, targets):
        losses = {}
        
        # 阶段A损失
        losses['comp_focal'] = self.focal_loss(
            predictions['comp_heatmap'], 
            targets['comp_gaussian']
        )
        losses['comp_subpx'] = self.huber_loss(
            predictions['comp_offset'],
            targets['comp_offset']
        )
        
        # 阶段B损失
        losses['bg_focal'] = self.focal_loss(
            predictions['bg_heatmap'],
            targets['bg_gaussian']
        )
        losses['pose_reg'] = self.huber_loss(
            predictions['pose'],
            targets['pose']
        )
        losses['ori_margin'] = self.margin_loss(
            predictions['ori_score_pos'],
            predictions['ori_score_neg']
        )
        losses['chamfer'] = self.chamfer_loss(
            predictions['piece_edge'],
            targets['gap_edge']
        )
        
        # 加权总损失
        total_loss = sum(
            self.weights[k] * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses
```

## 5. 训练流程

### 5.1 训练器类
```python
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = SliderCaptchaNet(config.model)
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 损失函数
        self.criterion = CompositeLoss(config)
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_amp else None
        
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = defaultdict(float)
        
        for batch in tqdm(dataloader):
            # 前向传播
            with autocast(enabled=self.config.use_amp):
                predictions = self.model(batch)
                loss, loss_dict = self.criterion(predictions, batch['labels'])
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 记录损失
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
        
        return epoch_losses
```

### 5.2 训练配置
```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 1e-4
  
  optimizer:
    type: "AdamW"
    betas: [0.9, 0.999]
    
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 2
    eta_min: 1e-6
    
  amp:
    enabled: true
    
  checkpoint:
    save_interval: 5
    save_best: true
    metric: "val_mae"
    
  early_stopping:
    patience: 15
    min_delta: 0.001
```

## 6. 评估指标

### 6.1 评估器
```python
class Evaluator:
    def __init__(self):
        self.reset()
    
    def update(self, predictions, targets):
        # 计算误差
        piece_error = torch.norm(
            predictions['piece_coord'] - targets['piece_coord'], 
            dim=1
        )
        gap_error = torch.norm(
            predictions['gap_coord'] - targets['gap_coord'],
            dim=1
        )
        
        self.piece_errors.extend(piece_error.cpu().numpy())
        self.gap_errors.extend(gap_error.cpu().numpy())
    
    def compute_metrics(self):
        metrics = {}
        
        # MAE
        metrics['piece_mae'] = np.mean(self.piece_errors)
        metrics['gap_mae'] = np.mean(self.gap_errors)
        metrics['total_mae'] = (metrics['piece_mae'] + metrics['gap_mae']) / 2
        
        # PCK@2px
        metrics['piece_pck2'] = np.mean(np.array(self.piece_errors) <= 2) * 100
        metrics['gap_pck2'] = np.mean(np.array(self.gap_errors) <= 2) * 100
        metrics['total_pck2'] = (metrics['piece_pck2'] + metrics['gap_pck2']) / 2
        
        return metrics
```

## 7. 实验管理

### 7.1 实验跟踪
```python
class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        
        # TensorBoard
        self.tb_writer = SummaryWriter(config.log_dir)
        
        # WandB (可选)
        if config.use_wandb:
            wandb.init(
                project="slider-captcha",
                config=config,
                name=config.exp_name
            )
    
    def log_metrics(self, metrics, step):
        # 记录到TensorBoard
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, step)
        
        # 记录到WandB
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_images(self, images, step):
        # 可视化预测结果
        pass
```

### 7.2 模型部署准备
```python
class ModelExporter:
    @staticmethod
    def export_onnx(model, save_path):
        """导出ONNX格式用于部署"""
        dummy_input = {
            'piece': torch.randn(1, 3, 50, 50),
            'background': torch.randn(1, 3, 160, 320),
            'composite': torch.randn(1, 3, 160, 320)
        }
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=['piece', 'background', 'composite'],
            output_names=['gap_coord', 'piece_coord'],
            dynamic_axes={'piece': {2: 'height', 3: 'width'}},
            opset_version=11
        )
    
    @staticmethod
    def export_torchscript(model, save_path):
        """导出TorchScript格式"""
        scripted = torch.jit.script(model)
        scripted.save(save_path)
```

## 8. 实施计划

### Phase 1: 基础实现 (Week 1)
- [ ] 实现数据加载器
- [ ] 构建基础模型架构
- [ ] 实现核心损失函数

### Phase 2: 训练系统 (Week 2)
- [ ] 完成训练循环
- [ ] 添加评估指标
- [ ] 集成TensorBoard

### Phase 3: 优化与调试 (Week 3)
- [ ] 超参数调优
- [ ] 数据增强优化
- [ ] 模型剪枝与量化

### Phase 4: 部署准备 (Week 4)
- [ ] 模型导出
- [ ] 推理优化
- [ ] API开发

## 9. 性能目标

- **训练速度**: 单GPU每epoch < 10分钟
- **收敛速度**: 50 epochs内达到目标精度
- **内存占用**: 批大小32时 < 8GB GPU内存
- **推理速度**: 单张图片 < 50ms (GPU) / < 200ms (CPU)
- **精度目标**: 
  - MAE < 1.5 pixels
  - PCK@2px > 95%