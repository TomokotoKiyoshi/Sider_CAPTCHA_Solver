import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Windows多进程共享内存优化
if os.name == 'nt':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 帮助调试CUDA错误
    # 设置共享内存方式
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
from datetime import datetime
import yaml
import time
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 移除混合精度训练相关导入

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import threading

# 尝试导入GPUtil，如果失败则禁用GPU监控
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not found. GPU monitoring disabled.")


# 从独立模块导入数据集类，解决Windows多进程pickle问题
try:
    from dataset import OptimizedCaptchaDataset
except ImportError:
    # 如果在当前目录找不到，尝试从scripts.training导入
    from scripts.training.dataset import OptimizedCaptchaDataset


def gaussian2D(shape, sigma=1):
    """生成2D高斯核"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """在热力图上绘制高斯分布"""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter/6)
    
    x = int(center[0])
    y = int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class FocalLoss(nn.Module):
    """CenterNet的Focal Loss"""
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        
        neg_weights = torch.pow(1 - gt, self.beta)
        
        loss = 0
        
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


# 导入正确的模型
from src.models.captcha_solver import CaptchaSolver


class OptimizedTrainer:
    """优化的训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.output_dir / f'training_log_{timestamp}.txt'
        
        # 记录配置
        self._log("Training configuration:")
        for key, value in config.items():
            self._log(f"  {key}: {value}")
        
        # 准备数据
        self.prepare_data()
        
        # 准备模型
        self.prepare_model()
        
        # 准备优化器和调度器
        self.prepare_optimizer()
        
        # 损失函数
        self.focal_loss = FocalLoss()
        self.l1_loss = nn.L1Loss()
        
        # 不使用混合精度训练
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 监控资源使用
        self.start_monitoring()
    
    def _log(self, message):
        """写入日志文件并打印"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def start_monitoring(self):
        """启动资源监控线程"""
        def monitor_resources():
            while True:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # 内存使用
                    memory = psutil.virtual_memory()
                    
                    # GPU使用率
                    if torch.cuda.is_available() and HAS_GPUTIL:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_util = gpu.load * 100
                            gpu_memory = gpu.memoryUtil * 100
                            gpu_power = gpu.powerDraw if hasattr(gpu, 'powerDraw') else 0
                            
                            self.writer.add_scalar('resources/gpu_util', gpu_util, self.global_step)
                            self.writer.add_scalar('resources/gpu_memory', gpu_memory, self.global_step)
                            self.writer.add_scalar('resources/gpu_power', gpu_power, self.global_step)
                    
                    self.writer.add_scalar('resources/cpu_percent', cpu_percent, self.global_step)
                    self.writer.add_scalar('resources/memory_percent', memory.percent, self.global_step)
                    
                    time.sleep(10)  # 每10秒更新一次
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def prepare_data(self):
        """准备数据加载器"""
        # 数据增强
        train_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 创建数据集
        train_annotations = Path(self.config['data_dir']) / 'train' / 'annotations.json'
        test_annotations = Path(self.config['data_dir']) / 'test' / 'annotations.json'
        
        # 确保annotations文件存在
        if not train_annotations.exists():
            raise FileNotFoundError(f"Train annotations not found: {train_annotations}")
        if not test_annotations.exists():
            raise FileNotFoundError(f"Test annotations not found: {test_annotations}")
        
        train_dataset = OptimizedCaptchaDataset(
            data_dir=self.config['data_dir'],
            annotations_file=str(train_annotations),
            split='train',
            transform=train_transform
        )
        
        val_dataset = OptimizedCaptchaDataset(
            data_dir=self.config['data_dir'],
            annotations_file=str(test_annotations),
            split='test',
            transform=val_transform
        )
        
        # 创建数据加载器
        # 获取配置的workers数量
        num_workers = self.config.get('num_workers', 8)
        
        # Windows下的特殊处理
        if os.name == 'nt':
            # Windows下禁用可能导致共享内存问题的特性
            persistent_workers = False
            pin_memory = False  # 禁用pin_memory避免共享内存错误
            self._log(f"Running on Windows. Using {num_workers} workers.")
            self._log("Disabled pin_memory and persistent_workers for Windows compatibility.")
            if num_workers > 4:
                self._log(f"Warning: Using {num_workers} workers on Windows may cause issues. Consider reducing if you encounter errors.")
        else:
            persistent_workers = True if num_workers > 0 else False
            pin_memory = True if torch.cuda.is_available() else False
        
        # 优化的批量大小
        batch_size = self.config.get('batch_size', 64)
        
        # Windows特定的DataLoader配置
        if os.name == 'nt':
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # Windows上禁用
                persistent_workers=False,  # Windows上禁用
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,  # Windows上禁用
                persistent_workers=False  # Windows上禁用
            )
        else:
            # Linux/Mac上使用完整优化
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        
        self._log(f"Data loaders created with {num_workers} workers, batch size {batch_size}")
    
    def prepare_model(self):
        """准备模型"""
        # 使用符合README要求的CaptchaSolver模型
        self.model = CaptchaSolver().to(self.device)
        
        # 如果有多个GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            self._log(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # 如果有预训练权重，加载它们
        if self.config.get('pretrained_weights'):
            self._log(f"Loading pretrained weights from {self.config['pretrained_weights']}")
            self.model.load_state_dict(torch.load(self.config['pretrained_weights']))
    
    def prepare_optimizer(self):
        """准备优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 使用OneCycleLR调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'],
            epochs=self.config['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def generate_targets(self, batch):
        """生成训练目标（热力图和偏移量）"""
        batch_size = batch['image'].shape[0]
        output_h, output_w = 40, 80  # 下采样4倍后的尺寸
        
        hms = []
        regs = []
        reg_masks = []
        
        for i in range(batch_size):
            # 初始化
            hm = np.zeros((2, output_h, output_w), dtype=np.float32)
            reg = np.zeros((4, output_h, output_w), dtype=np.float32)
            reg_mask = np.zeros((2, output_h, output_w), dtype=np.float32)
            
            # 背景缺口
            bg_center = batch['bg_center'][i].cpu().numpy() / 4  # 下采样4倍
            radius = max(1, int(self.config.get('gaussian_radius', 3)))
            draw_gaussian(hm[0], bg_center, radius)
            
            # 计算偏移量
            center_int = bg_center.astype(np.int32)
            offset = bg_center - center_int
            if 0 <= center_int[0] < output_w and 0 <= center_int[1] < output_h:
                reg[0, center_int[1], center_int[0]] = offset[0]
                reg[1, center_int[1], center_int[0]] = offset[1]
                reg_mask[0, center_int[1], center_int[0]] = 1
            
            # 滑块
            slider_center = batch['slider_center'][i].cpu().numpy() / 4
            draw_gaussian(hm[1], slider_center, radius)
            
            center_int = slider_center.astype(np.int32)
            offset = slider_center - center_int
            if 0 <= center_int[0] < output_w and 0 <= center_int[1] < output_h:
                reg[2, center_int[1], center_int[0]] = offset[0]
                reg[3, center_int[1], center_int[0]] = offset[1]
                reg_mask[1, center_int[1], center_int[0]] = 1
            
            hms.append(hm)
            regs.append(reg)
            reg_masks.append(reg_mask)
        
        return {
            'hm': torch.from_numpy(np.stack(hms)).to(self.device),
            'reg': torch.from_numpy(np.stack(regs)).to(self.device),
            'reg_mask': torch.from_numpy(np.stack(reg_masks)).to(self.device)
        }
    
    def compute_loss(self, outputs, targets):
        """计算损失"""
        # 热力图损失（outputs['hm']已经包含Sigmoid）
        hm_pred = outputs['hm']
        hm_loss = self.focal_loss(hm_pred, targets['hm'])
        
        # 偏移回归损失
        reg_mask_expanded = torch.zeros_like(outputs['reg'])
        reg_mask_expanded[:, 0:2, :, :] = targets['reg_mask'][:, 0:1, :, :].repeat(1, 2, 1, 1)
        reg_mask_expanded[:, 2:4, :, :] = targets['reg_mask'][:, 1:2, :, :].repeat(1, 2, 1, 1)
        
        reg_loss = self.l1_loss(
            outputs['reg'] * reg_mask_expanded,
            targets['reg'] * reg_mask_expanded
        )
        
        # 总损失
        total_loss = hm_loss + self.config['reg_weight'] * reg_loss
        
        return total_loss, hm_loss, reg_loss
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_hm_loss = 0
        total_reg_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备上
            # Windows上不使用non_blocking以避免共享内存问题
            non_blocking = False if os.name == 'nt' else True
            images = batch['image'].to(self.device, non_blocking=non_blocking)
            
            # 生成目标
            targets = self.generate_targets(batch)
            
            # 标准训练（不使用混合精度）
            outputs = self.model(images)
            
            # 转换输出格式以兼容原有的loss计算
            legacy_outputs = {
                'hm': torch.cat([outputs['gap_heatmap'], outputs['piece_heatmap']], dim=1),
                'reg': torch.cat([outputs['gap_offset'], outputs['piece_offset']], dim=1)
            }
            loss, hm_loss, reg_loss = self.compute_loss(legacy_outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            # 更新统计
            total_loss += loss.item()
            total_hm_loss += hm_loss.item()
            total_reg_loss += reg_loss.item()
            
            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hm_loss': f'{hm_loss.item():.4f}',
                'reg_loss': f'{reg_loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # 记录到tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/hm_loss', hm_loss.item(), self.global_step)
                self.writer.add_scalar('train/reg_loss', reg_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            self.global_step += 1
        
        # 返回平均损失
        n_batches = len(self.train_loader)
        return total_loss / n_batches, total_hm_loss / n_batches, total_reg_loss / n_batches
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_mae = 0  # 平均绝对误差
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Windows上不使用non_blocking以避免共享内存问题
                non_blocking = False if os.name == 'nt' else True
                images = batch['image'].to(self.device, non_blocking=non_blocking)
                targets = self.generate_targets(batch)
                
                # 前向传播
                outputs = self.model(images)
                
                # 转换输出格式以兼容原有的loss计算
                legacy_outputs = {
                    'hm': torch.cat([outputs['gap_heatmap'], outputs['piece_heatmap']], dim=1),
                    'reg': torch.cat([outputs['gap_offset'], outputs['piece_offset']], dim=1)
                }
                loss, _, _ = self.compute_loss(legacy_outputs, targets)
                
                total_loss += loss.item()
                
                # 解码预测并计算MAE
                pred_centers = self.decode_predictions(legacy_outputs)
                gt_bg_centers = batch['bg_center'].cpu().numpy()
                gt_slider_centers = batch['slider_center'].cpu().numpy()
                
                for i in range(len(pred_centers['bg'])):
                    if pred_centers['bg'][i] is not None:
                        mae_bg = np.abs(pred_centers['bg'][i] - gt_bg_centers[i]).mean()
                        total_mae += mae_bg
                    
                    if pred_centers['slider'][i] is not None:
                        mae_slider = np.abs(pred_centers['slider'][i] - gt_slider_centers[i]).mean()
                        total_mae += mae_slider
                    
                    total_samples += 2
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        
        # 记录到tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/mae', avg_mae, epoch)
        
        return avg_loss, avg_mae
    
    def decode_predictions(self, outputs):
        """解码模型预测，提取中心点坐标"""
        batch_size = outputs['hm'].shape[0]
        heatmaps = torch.sigmoid(outputs['hm'])
        offsets = outputs['reg']
        
        bg_centers = []
        slider_centers = []
        
        for i in range(batch_size):
            # 背景缺口中心
            hm_bg = heatmaps[i, 0].cpu().numpy()
            if hm_bg.max() > 0.1:
                y, x = np.unravel_index(hm_bg.argmax(), hm_bg.shape)
                offset_x = offsets[i, 0, y, x].cpu().numpy()
                offset_y = offsets[i, 1, y, x].cpu().numpy()
                center = np.array([x + offset_x, y + offset_y])
                bg_centers.append(center * 4)
            else:
                bg_centers.append(None)
            
            # 滑块中心
            hm_slider = heatmaps[i, 1].cpu().numpy()
            if hm_slider.max() > 0.1:
                y, x = np.unravel_index(hm_slider.argmax(), hm_slider.shape)
                offset_x = offsets[i, 2, y, x].cpu().numpy()
                offset_y = offsets[i, 3, y, x].cpu().numpy()
                center = np.array([x + offset_x, y + offset_y])
                slider_centers.append(center * 4)
            else:
                slider_centers.append(None)
        
        return {'bg': bg_centers, 'slider': slider_centers}
    
    def save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 保存最新的检查点
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        self._log(f"Saved latest checkpoint: {latest_path}")
        
        # 保存每个epoch的检查点
        epoch_path = self.output_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, epoch_path)
        self._log(f"Saved epoch checkpoint: {epoch_path}")
        
        # 如果是最佳模型，额外保存
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self._log(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        self._log("\n" + "="*50)
        self._log("Starting training...")
        self._log("="*50 + "\n")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_hm_loss, train_reg_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_mae = self.validate(epoch)
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss)
            
            # 记录日志
            self._log(f"\nEpoch {epoch}/{self.config['epochs']} Summary:")
            self._log(f"  Train Loss: {train_loss:.4f} (HM: {train_hm_loss:.4f}, Reg: {train_reg_loss:.4f})")
            self._log(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f} pixels")
            self._log(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}\n")
        
        self._log("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='训练滑块验证码检测模型')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 加载配置
    config_path = project_root / args.config
    
    # 如果配置文件不存在，使用默认配置
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 设置优化的默认配置
    default_config = {
        'data_dir': str(project_root / 'data'),
        'output_dir': str(project_root / 'src' / 'checkpoints'),
        'batch_size': 512,  # Windows下使用较小的批量大小
        'num_workers': 12,
        'epochs': 20,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'reg_weight': 1.0,
        # CaptchaSolver模型使用固定的ResNet18Lite架构，不需要backbone参数
        'gaussian_radius': 3,
        'pretrained_weights': None
    }
    
    # 合并配置
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # 创建训练器并开始训练
    trainer = OptimizedTrainer(config)
    trainer.train()


if __name__ == '__main__':
    # Windows下的多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 设置启动方法为spawn（Windows默认）
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()