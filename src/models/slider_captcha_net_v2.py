# -*- coding: utf-8 -*-
"""
滑块验证码识别网络 V2 - 完整修复版
解决了FPN步幅、亚像素接入、TopK候选、损失优化等所有关键问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class BlurPool2d(nn.Module):
    """抗锯齿池化层"""
    def __init__(self, channels=None, stride=2, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.channels = channels
        
        # 高斯核
        if kernel_size == 3:
            kernel = torch.tensor([[1, 2, 1]], dtype=torch.float32)
        elif kernel_size == 5:
            kernel = torch.tensor([[1, 4, 6, 4, 1]], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")
        
        kernel = kernel.T @ kernel
        kernel = kernel / kernel.sum()
        
        self.register_buffer('kernel', kernel[None, None, :, :])
        
    def forward(self, x):
        if self.channels is None:
            channels = x.shape[1]
        else:
            channels = self.channels
            
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        
        # 分组卷积实现模糊
        x = F.conv2d(x, kernel, stride=self.stride, 
                     padding=self.kernel_size//2, groups=channels)
        return x


class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class SliderBackbone(nn.Module):
    """共享骨干网络 - 改进的ResNet18"""
    def __init__(self):
        super().__init__()
        
        # Stem with BlurPool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BlurPool2d(channels=64, stride=2)
        )
        
        # ResNet stages
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)  # 64 channels, stride=4
        c2 = self.layer2(c1)  # 128 channels, stride=8
        return c1, c2


class FPN(nn.Module):
    """特征金字塔网络 - 修复版，输出stride=4的P2"""
    def __init__(self, in_channels=[64, 128], out_channel=128):
        super().__init__()
        
        # 横向连接
        self.lateral_c2 = nn.Conv2d(in_channels[0], out_channel, 1)  # C2: stride=4
        self.lateral_c3 = nn.Conv2d(in_channels[1], out_channel, 1)  # C3: stride=8
        
        # 3x3卷积消除混叠
        self.smooth = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        
    def forward(self, c1, c2):
        # c1=64ch, stride=4 ; c2=128ch, stride=8
        p3 = self.lateral_c3(c2)  # stride=8
        p2 = self.lateral_c2(c1) + F.interpolate(p3, scale_factor=2,  # -> stride=4
                                                 mode='bilinear', align_corners=False)
        p2 = self.smooth(p2)
        return p2  # 只返回P2 (stride=4)


class OrientationHead(nn.Module):
    """方向分解头"""
    def __init__(self, in_channels, K=24):
        super().__init__()
        assert K % 2 == 0, "K must be even for direction complementarity"
        self.K = K
        
        self.conv = nn.Conv2d(in_channels, K, 1)
        
    def forward(self, x):
        ori_features = self.conv(x)
        ori_features = F.softmax(ori_features, dim=1)
        return ori_features


class CorrelationModule(nn.Module):
    """相关计算模块 - 修复版"""
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def compute_2d_correlation(ori_piece, ori_composite):
        """
        2D同向相关 - 修复版使用unfold+bmm
        ori_piece: [B, K, h_f, w_f]
        ori_composite: [B, K, 40, 80]
        返回: [B, L_y, L_x] 相关图
        """
        B, K, h_f, w_f = ori_piece.shape
        _, _, H, W = ori_composite.shape
        
        # 展平模板
        tpl = ori_piece.reshape(B, K*h_f*w_f)  # [B, K*h_f*w_f]
        
        # unfold成滑窗
        patches = F.unfold(ori_composite, kernel_size=(h_f, w_f), stride=1)  # [B, K*h_f*w_f, L]
        
        # 批量内积
        corr = torch.bmm(tpl.unsqueeze(1), patches).squeeze(1)  # [B, L]
        
        # 复原为二维热力图
        Ly = H - h_f + 1
        Lx = W - w_f + 1
        return corr.view(B, Ly, Lx)
    
    @staticmethod
    def compute_1d_correlation(ori_piece, ori_background_strip, K):
        """
        1D互补方向相关 - 修复版使用unfold+einsum
        ori_piece: [B, K, h_f, w_f]
        ori_background_strip: [B, K, h_f, 80]
        返回: [B, L_x] 相关曲线
        """
        B, _, h_f, w_f = ori_piece.shape
        W = ori_background_strip.shape[-1]
        K_half = K // 2
        
        # 互补方向
        piece_comp = torch.roll(ori_piece, shifts=K_half, dims=1)  # [B,K,h_f,w_f]
        
        # 展平模板
        tpl = piece_comp.reshape(B, K*h_f*w_f)  # [B, Ck]
        
        # 背景滑窗
        patches = F.unfold(ori_background_strip, kernel_size=(h_f, w_f), stride=1)  # [B, Ck, Lx]
        
        # 相关计算
        corr = torch.einsum('bc, bcl -> bl', tpl, patches)  # [B, Lx]
        
        return corr


class PoseRegressionHead(nn.Module):
    """位姿回归头 - 预测角度和尺度"""
    def __init__(self, in_channels=None, K=24):
        super().__init__()
        
        # 输入: 拼图特征 + 背景特征 + 方向特征
        # 如果提供了in_channels，使用它；否则使用默认配置
        total_channels = in_channels if in_channels else (2 * K + 64)  # 2K for orientations, 64 for features
        
        self.conv1 = nn.Conv2d(total_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc_pose = nn.Linear(64, 3)  # cos(theta), sin(theta), log(scale)
        self.fc_ori_score = nn.Linear(64, 1)  # orientation consistency score
        
    def forward(self, features):
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        
        pose = self.fc_pose(x)
        ori_score = torch.sigmoid(self.fc_ori_score(x))
        
        return pose, ori_score


class SubpixelRefinement(nn.Module):
    """亚像素精修模块 - 修复版"""
    def __init__(self, window_size=9, temperature=0.04):
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature
        
    def zncc_corr(self, template, image):
        """
        向量化 ZNCC：对背景展开滑窗并逐窗口做(去均值/归一化)后与模板内积
        template: [B,1,Ht,Wt]
        image:    [B,1,Hi,Wi]
        return:   [B, Hi-Ht+1, Wi-Wt+1]
        """
        B, _, Ht, Wt = template.shape
        Hi, Wi = image.shape[-2:]
        Lh, Lw = Hi - Ht + 1, Wi - Wt + 1
        assert Lh > 0 and Lw > 0, "ROI must be >= template size"

        # 模板归一化
        t = template - template.mean(dim=(2,3), keepdim=True)
        t_std = t.std(dim=(2,3), keepdim=True).clamp_min(1e-6)
        t_norm = t / t_std              # [B,1,Ht,Wt]
        t_flat = t_norm.view(B, -1, 1)  # [B, Ht*Wt, 1]

        # 背景滑窗展开
        patches = F.unfold(image, kernel_size=(Ht, Wt), stride=1)      # [B, Ht*Wt, L]
        # 每个窗口去均值/归一化
        mu = patches.mean(dim=1, keepdim=True)
        std = patches.var(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6).sqrt()
        patches_norm = (patches - mu) / std                            # [B, Ht*Wt, L]

        # ZNCC = 逐窗口归一化后与模板内积
        corr = (t_flat * patches_norm).sum(dim=1)                      # [B, L]
        return corr.view(B, Lh, Lw)
    
    def soft_argmax(self, heatmap):
        """Soft-argmax for subpixel accuracy"""
        # 处理不同维度的输入
        if heatmap.dim() == 3:
            B, H, W = heatmap.shape
            heatmap = heatmap.unsqueeze(1)  # 添加通道维度
        else:
            B, C, H, W = heatmap.shape
            assert C == 1, f"Expected single channel, got {C}"
        
        # Apply softmax
        heatmap_flat = heatmap.view(B, -1)
        probs = F.softmax(heatmap_flat / self.temperature, dim=-1)
        probs = probs.view(B, 1, H, W)
        
        # Create coordinate grids  
        y_coords = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype)
        x_coords = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype)
        y_coords = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = x_coords.view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Compute expected coordinates
        expected_y = (probs * y_coords).sum(dim=(2,3))
        expected_x = (probs * x_coords).sum(dim=(2,3))
        
        # Get center position
        center_y = (H - 1) / 2.0
        center_x = (W - 1) / 2.0
        
        # Compute offset from center
        offset_y = expected_y.squeeze(1) - center_y
        offset_x = expected_x.squeeze(1) - center_x
        
        return torch.stack([offset_x, offset_y], dim=1)
    
    def warp_affine(self, img, cos_theta, sin_theta, log_scale):
        """仿射变换预配准"""
        B = img.shape[0]
        s = log_scale.exp()
        
        theta = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
        theta[:,0,0] = s * cos_theta
        theta[:,0,1] = -s * sin_theta  
        theta[:,1,0] = s * sin_theta
        theta[:,1,1] = s * cos_theta
        
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        return F.grid_sample(img, grid, align_corners=False)
    
    def forward(self, roi_piece, roi_background, pose=None):
        """
        亚像素精修
        roi_piece: [B, 1, H_p, W_p] 灰度图
        roi_background: [B, 1, H_roi, W_roi] 灰度图
        pose: Optional[Tensor] [B, 3] (cos, sin, log_s) for pre-alignment
        """
        # 输入验证
        if roi_piece.shape[2] > roi_background.shape[2] or roi_piece.shape[3] > roi_background.shape[3]:
            raise ValueError(f"Piece size {roi_piece.shape[2:]} larger than background {roi_background.shape[2:]}")
        
        # 如果有位姿，先预配准
        if pose is not None:
            cos_theta = pose[:, 0]
            sin_theta = pose[:, 1]
            log_scale = pose[:, 2]
            roi_piece = self.warp_affine(roi_piece, cos_theta, sin_theta, log_scale)
        
        # ZNCC相关
        corr_map = self.zncc_corr(roi_piece, roi_background)
        
        # Soft-argmax获取亚像素偏移
        offset = self.soft_argmax(corr_map)
        
        # 返回偏移和相关峰值
        max_corr = corr_map.max(dim=-1)[0].max(dim=-1)[0]
        
        return offset, max_corr


def to_gray(img):
    """RGB转灰度"""
    if img.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got {img.dim()}D")
    if img.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {img.shape[1]} channels")
    return (0.2989*img[:,0:1] + 0.5870*img[:,1:2] + 0.1140*img[:,2:3])


class SliderCaptchaNet(nn.Module):
    """滑块验证码识别主网络 V2 - 完整修复版"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.get('K', 24) % 2 == 0, "K must be even for direction complementarity"
        self.K = config.get('K', 24)
        self.topk = config.get('topk', 5)  # TopK候选数
        
        # 综合评分权重（可通过config调整）
        self.score_alpha = config.get('score_alpha', 1.0)  # CorrPeak权重
        self.score_beta = config.get('score_beta', 1.0)   # OriScore权重
        self.score_delta = config.get('score_delta', 0.2)  # Sharpness权重
        
        # 共享骨干
        self.backbone = SliderBackbone()
        
        # FPN - 修复版，只输出P2
        self.fpn = FPN(in_channels=[64, 128], out_channel=128)
        
        # 方向分解头
        self.ori_head = OrientationHead(in_channels=128, K=self.K)
        
        # 相关模块
        self.correlation = CorrelationModule()
        
        # 特征降维（用于候选拼接）
        self.feat_reduce = nn.Conv2d(128, 32, 1)
        
        # 位姿回归头
        self.pose_head = PoseRegressionHead(in_channels=128, K=self.K)
        
        # 亚像素精修
        self.subpixel = SubpixelRefinement()
        
    def forward(self, piece, background, composite=None, targets=None):
        """
        前向传播
        piece: [B, 3, H_p, W_p] 拼图
        background: [B, 3, H_b, W_b] 背景
        composite: [B, 3, H_c, W_c] 合成图（可选）
        targets: 真值字典（训练时提供）
        """
        B = piece.shape[0]
        device = piece.device
        
        # 如果没有合成图，简单拼接
        if composite is None:
            composite = background.clone()
        
        # 1. 特征提取
        p_c1, p_c2 = self.backbone(piece)
        b_c1, b_c2 = self.backbone(background)
        c_c1, c_c2 = self.backbone(composite)
        
        # 2. FPN - 修复：只返回P2 (stride=4)
        p2_piece = self.fpn(p_c1, p_c2)   # [B,128,hf,wf], hf≈Hp/4
        p2_bg = self.fpn(b_c1, b_c2)      # [B,128,40,80]
        p2_comp = self.fpn(c_c1, c_c2)    # [B,128,40,80]
        
        # 动态计算步幅（支持非正方形）
        stride_y = background.shape[-2] // p2_bg.shape[-2]  # 160 / 40 = 4
        stride_x = background.shape[-1] // p2_bg.shape[-1]  # 320 / 80 = 4
        # 分别处理x和y方向的步幅
        if stride_x != stride_y:
            print(f"Warning: Different strides - x:{stride_x}, y:{stride_y}")
        
        # 3. 方向分解
        ori_piece = self.ori_head(p2_piece)
        ori_bg = self.ori_head(p2_bg)
        ori_comp = self.ori_head(p2_comp)
        
        # 4. 阶段A: 在合成图中定位拼图
        comp_corr = self.correlation.compute_2d_correlation(ori_piece, ori_comp)
        
        # 峰值定位
        flat_idx = comp_corr.view(B, -1).argmax(dim=-1)
        Ly, Lx = comp_corr.shape[1], comp_corr.shape[2]
        y0 = flat_idx // Lx
        x0 = flat_idx % Lx
        
        # 合成图阶段亚像素精修
        comp_offset = torch.zeros(B, 2, device=device)
        if self.config.get('enable_subpixel', True):
            Hp, Wp = piece.shape[-2], piece.shape[-1]
            pad_y, pad_x = 4, 8
            comp_gray = to_gray(composite)
            piece_gray = to_gray(piece)
            
            roi_list = []
            for i in range(B):
                xl = int(x0[i]) * stride_x
                yt = int(y0[i]) * stride_y
                # 计算ROI边界，确保不超出图像范围
                yl = max(0, yt - pad_y)
                yr = min(composite.shape[-2], yt + Hp + pad_y)
                xl = max(0, xl - pad_x)
                xr = min(composite.shape[-1], xl + Wp + pad_x)
                
                # 确保ROI大小正确（改进的边界处理）
                min_height = Hp + 2*pad_y
                min_width = Wp + 2*pad_x
                
                if yr - yl < min_height:
                    center_y = (yl + yr) // 2
                    yl = max(0, center_y - min_height//2)
                    yr = min(composite.shape[-2], yl + min_height)
                    if yr - yl < min_height:
                        yr = min(composite.shape[-2], min_height)
                        yl = max(0, yr - min_height)
                        
                if xr - xl < min_width:
                    center_x = (xl + xr) // 2
                    xl = max(0, center_x - min_width//2)
                    xr = min(composite.shape[-1], xl + min_width)
                    if xr - xl < min_width:
                        xr = min(composite.shape[-1], min_width)
                        xl = max(0, xr - min_width)
                
                roi_list.append(comp_gray[i:i+1, :, yl:yr, xl:xr])
                
            roi_comp = torch.cat(roi_list, dim=0)  # [B,1,Hp+2py,Wp+2px]
            comp_offset, _ = self.subpixel(piece_gray, roi_comp)  # [B,2] (dx,dy) in pixels
        
        # 5. 阶段B: 在背景中定位缺口
        # 提取背景条带（考虑y轴容差）
        h_f = ori_piece.shape[2]
        
        # 可选：如果合成图与背景非同构，进行y坐标映射
        # y_b = a * y_p + b 的线性映射
        if self.config.get('enable_y_mapping', False):
            # 从config中获取映射参数，或使用默认值（同构映射）
            y_map_scale = self.config.get('y_map_scale', 1.0)
            y_map_offset = self.config.get('y_map_offset', 0.0)
            y_strip_base = (y0.float() * y_map_scale + y_map_offset).long()
        else:
            # 默认：同构映射（合成图与背景像素级对应）
            y_strip_base = y0
        
        # 修复：正确的批处理索引
        ori_bg_strips = []
        p2_bg_strips = []
        for i in range(B):
            y_start = y_strip_base[i].item()
            y_end = y_start + h_f
            # 确保不超出范围
            y_start = max(0, y_start)
            y_end = min(ori_bg.shape[2], y_end)
            ori_bg_strips.append(ori_bg[i:i+1, :, y_start:y_end, :])
            p2_bg_strips.append(p2_bg[i:i+1, :, y_start:y_end, :])
        
        ori_bg_strip = torch.cat(ori_bg_strips, dim=0)
        p2_bg_strip = torch.cat(p2_bg_strips, dim=0)
        
        # 1D相关
        bg_corr = self.correlation.compute_1d_correlation(ori_piece, ori_bg_strip, self.K)
        
        # 获取TopK候选
        topk_scores, topk_indices = torch.topk(bg_corr, k=min(self.topk, bg_corr.shape[1]), dim=-1)
        
        # 处理每个候选
        all_poses = []
        all_ori_scores = []
        w_f = ori_piece.shape[3]
        
        # 检查背景条带宽度
        if ori_bg_strip.shape[3] < w_f:
            raise ValueError(f"Background strip width {ori_bg_strip.shape[3]} smaller than piece width {w_f}")
        
        for k in range(topk_indices.shape[1]):
            # 提取第k个候选的背景窗口
            ori_bg_wins = []
            p2_bg_wins = []
            for i in range(B):
                xs = int(topk_indices[i, k])
                xe = min(xs + w_f, ori_bg_strip.shape[3])
                xs = max(0, xe - w_f)  # 确保窗口大小正确
                ori_bg_wins.append(ori_bg_strip[i:i+1, :, :, xs:xe])
                p2_bg_wins.append(self.feat_reduce(p2_bg_strip[i:i+1, :, :, xs:xe]))
            
            ori_bg_win = torch.cat(ori_bg_wins, dim=0)
            p2_bg_win = torch.cat(p2_bg_wins, dim=0)
            
            # 准备候选特征
            p2_piece_reduced = self.feat_reduce(p2_piece)
            cand_features = torch.cat([
                ori_piece,
                ori_bg_win,
                p2_piece_reduced,
                p2_bg_win
            ], dim=1)
            
            # 位姿回归
            pose, ori_score = self.pose_head(cand_features)
            all_poses.append(pose)
            all_ori_scores.append(ori_score)
        
        # Stack所有候选的结果
        all_poses = torch.stack(all_poses, dim=1)  # [B, K, 3]
        all_ori_scores = torch.stack(all_ori_scores, dim=1)  # [B, K, 1]
        
        # 增强的候选重排 - 融合多个评分指标
        if self.training:
            # 训练时仅使用ori_score
            best_k = all_ori_scores.squeeze(-1).argmax(dim=1)
        else:
            # 推理时使用综合评分
            best_k = self.select_best_candidate(
                bg_corr, topk_indices, topk_values, all_ori_scores,
                y_strip_h=y_strip.shape[-1], stride_x=stride_x
            )
        
        best_idx = topk_indices.gather(1, best_k.unsqueeze(1)).squeeze(1)
        best_pose = all_poses[torch.arange(B), best_k]
        
        # 背景阶段亚像素精修
        bg_offset = torch.zeros(B, 2, device=device)
        if self.config.get('enable_subpixel', True):
            Hp, Wp = piece.shape[-2], piece.shape[-1]
            pad_y, pad_x = 4, 8
            bg_gray = to_gray(background)
            piece_gray = to_gray(piece)
            
            roi_list = []
            for i in range(B):
                xl = int(best_idx[i]) * stride_x
                yt = int(y_strip_base[i]) * stride_y
                # 计算ROI边界
                yl = max(0, yt - pad_y)
                yr = min(background.shape[-2], yt + Hp + pad_y)
                xl = max(0, xl - pad_x)
                xr = min(background.shape[-1], xl + Wp + pad_x)
                
                # 确保ROI大小正确（改进的边界处理）
                min_height = Hp + 2*pad_y
                min_width = Wp + 2*pad_x
                
                if yr - yl < min_height:
                    center_y = (yl + yr) // 2
                    yl = max(0, center_y - min_height//2)
                    yr = min(background.shape[-2], yl + min_height)
                    if yr - yl < min_height:
                        yr = min(background.shape[-2], min_height)
                        yl = max(0, yr - min_height)
                        
                if xr - xl < min_width:
                    center_x = (xl + xr) // 2
                    xl = max(0, center_x - min_width//2)
                    xr = min(background.shape[-1], xl + min_width)
                    if xr - xl < min_width:
                        xr = min(background.shape[-1], min_width)
                        xl = max(0, xr - min_width)
                
                roi_list.append(bg_gray[i:i+1, :, yl:yr, xl:xr])
                
            roi_bg = torch.cat(roi_list, dim=0)  # [B,1,Hp+2py,Wp+2px]
            
            # 位姿预配准后做ZNCC
            bg_offset, _ = self.subpixel(piece_gray, roi_bg, pose=best_pose)
        
        # 6. 最终坐标计算（使用动态步幅）
        piece_x = (x0.float() + comp_offset[:, 0] / stride_x) * stride_x
        piece_y = (y0.float() + comp_offset[:, 1] / stride_y) * stride_y
        
        gap_x = (best_idx.float() + bg_offset[:, 0] / stride_x) * stride_x
        gap_y = (y_strip_base.float() + bg_offset[:, 1] / stride_y) * stride_y
        
        piece_coord = torch.stack([piece_x, piece_y], dim=1)
        gap_coord = torch.stack([gap_x, gap_y], dim=1)
        
        outputs = {
            'comp_corr': comp_corr,
            'comp_offset': comp_offset,
            'bg_corr': bg_corr,
            'bg_offset': bg_offset,
            'topk_indices': topk_indices,
            'all_poses': all_poses,
            'all_ori_scores': all_ori_scores,
            'best_pose': best_pose,
            'piece_coord': piece_coord,
            'gap_coord': gap_coord,
        }
        
        # 7. 计算损失（训练时）
        if targets is not None:
            losses = self.compute_losses(outputs, targets)
            outputs['losses'] = losses
            outputs['total_loss'] = sum(losses.values())
        
        return outputs
    
    def compute_losses(self, outputs, targets):
        """计算所有损失项 - 优化版"""
        losses = {}
        
        # 1. 合成图2D相关损失 (Softmax-CE)
        if 'comp_gt' in targets:
            comp_loss = self.softmax_ce_2d(
                outputs['comp_corr'], 
                targets['comp_gt'],
                sigma=1.0
            )
            losses['L_comp'] = comp_loss
        
        # 2. 背景1D相关损失 (Softmax-CE)
        if 'bg_gt' in targets:
            bg_loss = self.softmax_ce_1d(
                outputs['bg_corr'],
                targets['bg_gt'],
                sigma=1.0
            )
            losses['L_bg'] = bg_loss
        
        # 3. TopK候选的位姿和方向损失
        if 'pose_gt' in targets and 'gt_idx' in targets:
            all_poses = outputs['all_poses']  # [B, K, 3]
            all_ori_scores = outputs['all_ori_scores']  # [B, K, 1]
            pose_gt = targets['pose_gt']
            gt_idx = targets['gt_idx']  # 真值在topk中的索引
            
            # 找到正样本候选（最接近真值的）
            topk_indices = outputs['topk_indices']
            B = topk_indices.shape[0]
            
            pos_pose_losses = []
            for i in range(B):
                # 找到最接近gt_idx的候选
                dist_to_gt = (topk_indices[i] - gt_idx[i]).abs()
                pos_k = dist_to_gt.argmin()
                
                # 计算位姿损失（仅对正样本）
                pos_pose = all_poses[i, pos_k]
                theta_loss = F.smooth_l1_loss(pos_pose[:2], pose_gt[i, :2])
                scale_loss = F.smooth_l1_loss(pos_pose[2:3], pose_gt[i, 2:3])
                pos_pose_losses.append(theta_loss + 0.5 * scale_loss)
            
            losses['L_pose'] = torch.stack(pos_pose_losses).mean()
            
            # 4. 方向一致性Margin Loss
            if 'ori_label' in targets:
                ori_labels = targets['ori_label']  # [B, K] 标记哪些是正/负样本
                margin = 0.2
                
                ori_losses = []
                for i in range(B):
                    pos_mask = ori_labels[i] == 1
                    neg_mask = ori_labels[i] == 0
                    
                    if pos_mask.any():
                        pos_scores = all_ori_scores[i, pos_mask]
                        pos_loss = (1 - pos_scores).clamp_min(0).mean()
                    else:
                        pos_loss = torch.tensor(0.0, device=all_ori_scores.device, dtype=all_ori_scores.dtype)
                    
                    if neg_mask.any():
                        neg_scores = all_ori_scores[i, neg_mask]
                        neg_loss = (neg_scores - margin).clamp_min(0).mean()
                    else:
                        neg_loss = torch.tensor(0.0, device=all_ori_scores.device, dtype=all_ori_scores.dtype)
                    
                    ori_losses.append(pos_loss + neg_loss)
                
                losses['L_ori_margin'] = torch.stack(ori_losses).mean() * 0.5
        
        # 5. 亚像素偏移损失
        if 'comp_offset_gt' in targets:
            comp_offset_loss = F.smooth_l1_loss(
                outputs['comp_offset'],
                targets['comp_offset_gt']
            )
            losses['L_subpx_comp'] = comp_offset_loss * 0.5
        
        if 'bg_offset_gt' in targets:
            bg_offset_loss = F.smooth_l1_loss(
                outputs['bg_offset'],
                targets['bg_offset_gt']
            )
            losses['L_subpx_bg'] = bg_offset_loss
        
        # 6. 坐标损失（端到端）
        if 'piece_coord_gt' in targets:
            coord_loss = F.smooth_l1_loss(
                outputs['piece_coord'],
                targets['piece_coord_gt']
            )
            losses['L_coord_piece'] = coord_loss
            
        if 'gap_coord_gt' in targets:
            gap_coord_loss = F.smooth_l1_loss(
                outputs['gap_coord'],
                targets['gap_coord_gt']
            )
            losses['L_coord_gap'] = gap_coord_loss
        
        return losses
    
    def softmax_ce_2d(self, corr, center_uv, sigma=1.0):
        """2D Softmax-CE with Gaussian soft labels"""
        B, H, W = corr.shape
        
        # Log-softmax
        logP = F.log_softmax(corr.view(B, -1), dim=-1).view(B, H, W)
        
        # 创建高斯软标签
        target = self.create_gaussian_2d(center_uv, (H, W), sigma)
        
        # Cross-entropy
        loss = -(target * logP).sum(dim=(1,2)) / (target.sum(dim=(1,2)) + 1e-6)
        
        return loss.mean()
    
    def softmax_ce_1d(self, corr, center_x, sigma=1.0):
        """1D Softmax-CE with Gaussian soft labels"""
        B, L = corr.shape
        
        # Log-softmax
        logP = F.log_softmax(corr, dim=-1)
        
        # 创建高斯软标签
        target = self.create_gaussian_1d(center_x, L, sigma)
        
        # Cross-entropy
        loss = -(target * logP).sum(dim=1) / (target.sum(dim=1) + 1e-6)
        
        return loss.mean()
    
    @staticmethod
    def create_gaussian_2d(centers, size, sigma=1.0):
        """创建2D高斯软标签 - 向量化版本"""
        B = centers.shape[0]
        H, W = size
        device = centers.device
        
        # 创建网格
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # 扩展维度 [1, H, W]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        # centers: [B, 2] -> [B, 1, 1]
        cx = centers[:, 0].view(B, 1, 1)
        cy = centers[:, 1].view(B, 1, 1)
        
        # 计算距离平方
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        return gaussian
    
    @staticmethod  
    def create_gaussian_1d(centers, length, sigma=1.0):
        """创建1D高斯软标签 - 向量化版本"""
        B = centers.shape[0]
        device = centers.device
        
        # 创建位置坐标 [1, length]
        x = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
        
        # centers: [B] -> [B, 1]
        if centers.dim() == 1:
            cx = centers.unsqueeze(1)
        else:
            cx = centers
        
        # 计算距离平方 [B, length]
        dist_sq = (x - cx) ** 2
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        return gaussian
    
    def select_best_candidate(self, bg_corr, topk_indices, topk_values, 
                              all_ori_scores, y_strip_h, stride_x):
        """
        增强的候选重排 - 融合多个评分指标
        
        Args:
            bg_corr: [B, L] 1D相关响应
            topk_indices: [B, K] TopK索引
            topk_values: [B, K] TopK相关值（CorrPeak）
            all_ori_scores: [B, K, 1] 方向一致性分数
            y_strip_h: y带高度
            stride_x: x方向步幅
        
        Returns:
            best_k: [B] 每个样本的最佳候选索引
        """
        B, K = topk_indices.shape
        device = bg_corr.device
        
        # 1. 相关峰值分数（已归一化）
        corr_peak_scores = topk_values  # [B, K]
        
        # 2. 方向一致性分数
        ori_scores = all_ori_scores.squeeze(-1)  # [B, K]
        
        # 3. 峰值尖锐度分数（通过二阶差分估算）
        sharpness_scores = torch.zeros(B, K, device=device)
        for b in range(B):
            for k in range(K):
                idx = topk_indices[b, k].item()
                # 确保不越界
                if idx > 0 and idx < bg_corr.shape[1] - 1:
                    # 二阶差分：f''(x) ≈ f(x-1) - 2*f(x) + f(x+1)
                    second_diff = bg_corr[b, idx-1] - 2*bg_corr[b, idx] + bg_corr[b, idx+1]
                    # 尖锐度：负二阶导数（峰值处应为负值，取绝对值）
                    sharpness_scores[b, k] = abs(second_diff)
        
        # 归一化各项分数到[0,1]
        if corr_peak_scores.max() > corr_peak_scores.min():
            corr_peak_scores = (corr_peak_scores - corr_peak_scores.min()) / \
                               (corr_peak_scores.max() - corr_peak_scores.min() + 1e-6)
        
        if sharpness_scores.max() > sharpness_scores.min():
            sharpness_scores = (sharpness_scores - sharpness_scores.min()) / \
                              (sharpness_scores.max() - sharpness_scores.min() + 1e-6)
        
        # 4. 综合评分
        # Score = α * CorrPeak + β * OriScore + δ * Sharpness
        composite_scores = (self.score_alpha * corr_peak_scores + 
                           self.score_beta * ori_scores + 
                           self.score_delta * sharpness_scores)
        
        # 选择最高分的候选
        best_k = composite_scores.argmax(dim=1)  # [B]
        
        return best_k