# -*- coding: utf-8 -*-
"""
Padding BCE损失（Padding Binary Cross-Entropy Loss）实现
用于抑制padding区域（无效区域）的误检测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddingBCELoss(nn.Module):
    """
    Padding BCE损失函数
    
    确保模型在padding区域（无效区域）不产生误检测。
    由于输入包含padding区域，模型可能在这些区域产生虚假响应，需要显式抑制。
    
    数学公式：
    L_pad_bce = (1/N_pad) * Σ_{y,x} [-log(1-P_g(y,x)) - log(1-P_p(y,x))] * (1-M(y,x))
    
    其中：
    - P_g, P_p: 模型输出的gap和piece热力图概率值（已经过Sigmoid激活）
    - M: 有效区域掩码（M=1表示有效区域，M=0表示padding区域）
    - 1-M: padding区域选择器
    - N_pad: padding像素总数，用于归一化
    
    特点：
    - 只在padding区域计算损失
    - 目标是让padding区域的预测概率接近0
    - 使用BCE损失的负样本部分（目标为0）
    """
    
    def __init__(self,
                 eps: float = 1e-8,
                 max_loss_per_pixel: float = 10.0):
        """
        Args:
            eps: 数值稳定性常数，防止log(0)
            max_loss_per_pixel: 单个像素的最大损失值，防止梯度爆炸
        """
        super().__init__()
        self.eps = eps
        self.max_loss_per_pixel = max_loss_per_pixel
    
    def forward(self,
                heatmap_gap: torch.Tensor,
                heatmap_piece: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算Padding BCE损失
        
        Args:
            heatmap_gap: gap热力图预测 [B, 1, H, W]，已经过Sigmoid激活，值域(0,1)
            heatmap_piece: piece热力图预测 [B, 1, H, W]，已经过Sigmoid激活，值域(0,1)
            mask: 有效区域掩码 [B, 1, H, W]，1表示有效区域，0表示padding区域
        
        Returns:
            标量损失值
        """
        # 【数值稳定性修复】强制使用FP32
        heatmap_gap = heatmap_gap.float()
        heatmap_piece = heatmap_piece.float()
        mask = mask.float()
        
        # 确保输入维度正确
        assert heatmap_gap.dim() == 4 and heatmap_piece.dim() == 4, \
            f"Expected 4D tensors, got gap: {heatmap_gap.dim()}D, piece: {heatmap_piece.dim()}D"
        assert mask.dim() == 4, f"Expected 4D mask tensor, got {mask.dim()}D"
        
        # 计算padding区域掩码（1-M）
        padding_mask = 1 - mask  # [B, 1, H, W]
        
        # 计算padding像素总数
        N_pad = padding_mask.sum()
        
        # 如果没有padding区域，返回0
        if N_pad < self.eps:
            return torch.tensor(0.0, device=heatmap_gap.device, requires_grad=True)
        
        # 数值稳定性：使用更大的eps避免FP16下溢
        # 注意：不能使用clamp_（原地操作），会破坏梯度计算
        heatmap_gap_clamped = heatmap_gap.clamp(min=1e-4, max=1 - 1e-4)
        heatmap_piece_clamped = heatmap_piece.clamp(min=1e-4, max=1 - 1e-4)
        
        # 计算BCE损失（目标为0）
        # 使用log1p更稳定：-log(1-p) = -log1p(-p)
        loss_gap = -torch.log1p(-heatmap_gap_clamped)
        loss_piece = -torch.log1p(-heatmap_piece_clamped)
        
        # 防止单个像素贡献过大损失（梯度裁剪）
        loss_gap = torch.clamp(loss_gap, max=self.max_loss_per_pixel)
        loss_piece = torch.clamp(loss_piece, max=self.max_loss_per_pixel)
        
        # 只在padding区域计算损失
        loss_gap = loss_gap * padding_mask
        loss_piece = loss_piece * padding_mask
        
        # 总损失：gap和piece的BCE损失之和
        total_loss = loss_gap.sum() + loss_piece.sum()
        
        # 归一化：除以padding像素总数
        L_pad_bce = total_loss / N_pad
        
        return L_pad_bce


def create_padding_bce_loss(config: dict = None) -> PaddingBCELoss:
    """
    工厂函数：创建Padding BCE损失实例
    
    Args:
        config: 配置字典，包含：
            - eps: 数值稳定性常数（默认1e-8）
            - max_loss_per_pixel: 单像素最大损失（默认10.0）
    
    Returns:
        PaddingBCELoss实例
    """
    if config is None:
        config = {}
    
    return PaddingBCELoss(
        eps=config.get('eps', 1e-8),
        max_loss_per_pixel=config.get('max_loss_per_pixel', 10.0)
    )


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    B, H, W = 2, 64, 128
    
    # 模拟热力图（包含一些padding区域的响应）
    heatmap_gap = torch.rand(B, 1, H, W) * 0.3  # 随机低响应
    heatmap_piece = torch.rand(B, 1, H, W) * 0.3
    
    # 创建有效区域掩码（模拟letterbox padding）
    mask = torch.ones(B, 1, H, W)
    # 左右各padding 20列
    mask[:, :, :, :20] = 0
    mask[:, :, :, -20:] = 0
    
    # 在padding区域故意设置一些高响应（模拟误检测）
    heatmap_gap[:, :, :, :10] = 0.8  # 左侧padding区域高响应
    heatmap_piece[:, :, :, -10:] = 0.7  # 右侧padding区域高响应
    
    # Sigmoid激活（模拟网络输出）
    heatmap_gap = torch.sigmoid(heatmap_gap)
    heatmap_piece = torch.sigmoid(heatmap_piece)
    
    # 设置requires_grad以测试梯度
    heatmap_gap.requires_grad = True
    heatmap_piece.requires_grad = True
    
    # 创建损失函数
    loss_fn = create_padding_bce_loss({
        'eps': 1e-8,
        'max_loss_per_pixel': 10.0
    })
    
    # 计算损失
    loss = loss_fn(heatmap_gap, heatmap_piece, mask)
    
    print(f"Padding BCE Loss: {loss.item():.4f}")
    print(f"Padding像素数: {int((1-mask).sum().item())}")
    
    # 测试没有padding的情况
    mask_no_padding = torch.ones(B, 1, H, W)
    loss_no_padding = loss_fn(heatmap_gap, heatmap_piece, mask_no_padding)
    print(f"\n无padding时的损失: {loss_no_padding.item():.4f}")
    
    # 验证梯度
    loss.backward()
    print(f"\n梯度计算成功!")
    print(f"Gap热力图梯度范数: {heatmap_gap.grad.norm().item():.4f}")
    print(f"Piece热力图梯度范数: {heatmap_piece.grad.norm().item():.4f}")