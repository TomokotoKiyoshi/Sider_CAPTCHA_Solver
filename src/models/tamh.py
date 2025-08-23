# -*- coding: utf-8 -*-
"""
TAMH (Template-Aware Matching Head) Module
模板感知匹配头 - 通过模板匹配增强缺口定位精度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .modules import ConvBNAct


class TAMH(nn.Module):
    """
    Template-Aware Matching Head (TAMH)
    
    通过从拼图热图提取模板，使用动态卷积进行相关匹配，
    提高缺口定位的鲁棒性和准确性。
    
    架构流程:
        1. 从Hf根据H_piece热图提取滑块模板R_piece
        2. 通过GAP+FC生成动态卷积核
        3. 使用depthwise卷积进行相关搜索
        4. 融合原始H_gap和相关热图H_corr得到最终预测
    
    输入:
        - Hf: [B,128,64,128] - 来自Stage5的特征图
        - H_piece: [B,1,64,128] - 拼图中心热图
        - H_gap: [B,1,64,128] - 缺口中心热图（原始预测）
    
    输出:
        - H_gap_final: [B,1,64,128] - 增强后的缺口中心热图
    """
    
    def __init__(self,
                 feature_channels: int = 128,
                 template_size: int = 16,
                 kernel_size: int = 3,
                 fusion_channels: int = 32,
                 use_l2_norm: bool = True,
                 padding_mode: str = 'replicate'):
        """
        Args:
            feature_channels: 特征图通道数 (默认128)
            template_size: 模板大小 (默认16x16)
            kernel_size: 动态卷积核大小 (默认3x3)
            fusion_channels: 融合层中间通道数 (默认32)
            use_l2_norm: 是否对动态卷积核进行L2归一化 (默认True)
            padding_mode: 模板提取时的padding模式 ('replicate', 'zeros', 'reflect')
        """
        super().__init__()
        
        # 保存配置
        self.feature_channels = feature_channels
        self.template_size = template_size
        self.kernel_size = kernel_size
        self.use_l2_norm = use_l2_norm
        self.padding_mode = padding_mode
        
        # Step 2: 动态卷积核生成网络
        # GAP后的特征维度
        gap_features = feature_channels
        # 输出维度：每个通道一个kernel_size×kernel_size的卷积核
        kernel_features = feature_channels * kernel_size * kernel_size
        
        # 全连接层：将GAP特征映射到卷积核参数
        self.kernel_generator = nn.Sequential(
            nn.Linear(gap_features, kernel_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(kernel_features // 2, kernel_features)
        )
        
        # Step 4: 热图融合网络
        # 拼接原始热图和相关热图，然后融合
        self.fusion_net = nn.Sequential(
            ConvBNAct(
                in_channels=2,  # H_gap + H_corr
                out_channels=fusion_channels,
                kernel_size=3,
                stride=1,
                groups=1,
                act_type='silu',
                bias=True
            ),
            nn.Conv2d(
                in_channels=fusion_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 动态卷积核生成器初始化
        for m in self.kernel_generator:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 融合网络最后一层初始化
        nn.init.normal_(self.fusion_net[-2].weight, mean=0, std=0.001)
        nn.init.constant_(self.fusion_net[-2].bias, 0)
    
    def extract_template(self, 
                        Hf: torch.Tensor, 
                        H_piece: torch.Tensor) -> torch.Tensor:
        """
        Step 1: 从特征图中提取滑块模板
        
        注意：此函数使用argmax操作定位拼图中心，该操作不可微分。
        因此H_piece不会接收梯度。这是设计上的权衡，因为：
        1. H_piece主要用于定位，其预测质量由原始DualHead负责
        2. 避免了复杂的可微分采样操作，保持了推理效率
        
        Args:
            Hf: 特征图 [B,128,64,128]
            H_piece: 拼图中心热图 [B,1,64,128]
        
        Returns:
            R_piece: 滑块模板 [B,128,16,16]
        """
        B, C, H, W = Hf.shape
        device = Hf.device
        dtype = Hf.dtype
        
        # 获取每个batch中拼图中心的位置
        H_piece_flat = H_piece.view(B, -1)  # [B, H*W]
        max_indices = torch.argmax(H_piece_flat, dim=1)  # [B]
        
        # 转换为2D坐标（浮点数）
        v_centers = (max_indices // W).float()  # 行索引
        u_centers = (max_indices % W).float()   # 列索引
        
        # ========== 使用grid_sample进行纯tensor操作 ==========
        # 将中心坐标归一化到[-1, 1]范围
        # grid_sample坐标系统：-1=左/上边界，1=右/下边界
        u_norm = (u_centers / (W - 1)) * 2.0 - 1.0  # [B]
        v_norm = (v_centers / (H - 1)) * 2.0 - 1.0  # [B]
        
        # 创建16x16的采样网格
        # 生成相对坐标，表示以16x16的区域
        half_size_norm = self.template_size / 2.0
        # 在[-8, 7]范围内生成16个点（以像素为单位）
        template_coords = torch.linspace(-half_size_norm + 0.5, half_size_norm - 0.5, 
                                       self.template_size, device=device, dtype=dtype)
        # 转换到归一化坐标
        template_coords_u = template_coords / (W / 2.0)  # 转换到[-1,1]范围
        template_coords_v = template_coords / (H / 2.0)
        
        # 创建网格
        grid_v, grid_u = torch.meshgrid(template_coords_v, template_coords_u, indexing='ij')
        
        # 为每个batch生成偏移后的网格
        # grid shape: [B, template_size, template_size, 2]
        grid = torch.zeros(B, self.template_size, self.template_size, 2, 
                          device=device, dtype=dtype)
        
        # 广播并添加中心偏移
        # 注意：grid_sample的坐标顺序是(x, y)，对应(width, height)
        grid[..., 0] = grid_u.unsqueeze(0) + u_norm.view(B, 1, 1)  # x坐标(宽度)
        grid[..., 1] = grid_v.unsqueeze(0) + v_norm.view(B, 1, 1)  # y坐标(高度)
        
        # 使用grid_sample提取模板
        # padding_mode选项：'zeros'(补零), 'border'(复制边界), 'reflection'(反射)
        if self.padding_mode == 'zeros':
            padding_mode = 'zeros'
        elif self.padding_mode == 'replicate':
            padding_mode = 'border'  # grid_sample中'border'等价于'replicate'
        else:  # reflect
            padding_mode = 'reflection'
        
        # 执行采样
        R_piece = F.grid_sample(Hf, grid, mode='bilinear', 
                               padding_mode=padding_mode, align_corners=False)
        
        return R_piece
    
    def generate_dynamic_kernel(self, R_piece: torch.Tensor) -> torch.Tensor:
        """
        Step 2: 生成动态卷积核
        
        Args:
            R_piece: 滑块模板 [B,128,16,16]
        
        Returns:
            W: 动态卷积核 [B,128,3,3]
        """
        B, C = R_piece.shape[:2]
        
        # 全局平均池化
        gap = F.adaptive_avg_pool2d(R_piece, (1, 1))  # [B, C, 1, 1]
        gap = gap.view(B, C)  # [B, C]
        
        # 通过FC层生成卷积核参数
        kernel_params = self.kernel_generator(gap)  # [B, C*k*k]
        
        # Reshape为卷积核形状
        W = kernel_params.view(B, C, self.kernel_size, self.kernel_size)  # [B, C, k, k]
        
        # L2归一化（每个通道独立归一化）
        if self.use_l2_norm:
            # 对每个通道的卷积核进行L2归一化
            W_flat = W.view(B, C, -1)  # [B, C, k*k]
            W_norm = F.normalize(W_flat, p=2, dim=2)  # 在kernel维度上归一化
            W = W_norm.view(B, C, self.kernel_size, self.kernel_size)
        
        return W
    
    def cross_correlation(self, 
                         Hf: torch.Tensor, 
                         W: torch.Tensor) -> torch.Tensor:
        """
        Step 3: 使用动态卷积核进行相关搜索
        
        Args:
            Hf: 特征图 [B,128,64,128]
            W: 动态卷积核 [B,128,3,3]
        
        Returns:
            H_corr: 相关热图 [B,1,64,128]
        """
        B, C, H, Wf = Hf.shape
        
        # 计算padding以保持尺寸
        padding = self.kernel_size // 2
        
        # 对每个batch独立进行depthwise卷积
        corr_maps = []
        for b in range(B):
            # 获取当前batch的特征和卷积核
            hf_b = Hf[b:b+1]  # [1, C, H, W]
            w_b = W[b]  # [C, k, k]
            
            # Reshape卷积核为depthwise卷积格式 [C, 1, k, k]
            w_b = w_b.unsqueeze(1)  # [C, 1, k, k]
            
            # 执行depthwise卷积（groups=C）
            corr_b = F.conv2d(hf_b, w_b, padding=padding, groups=C)  # [1, C, H, W]
            
            # 对通道维度求和得到单通道相关图
            corr_b = torch.sum(corr_b, dim=1, keepdim=True)  # [1, 1, H, W]
            
            corr_maps.append(corr_b)
        
        # 拼接所有batch
        H_corr = torch.cat(corr_maps, dim=0)  # [B, 1, H, W]
        
        # 归一化到[0,1]（使用sigmoid）
        H_corr = torch.sigmoid(H_corr)
        
        return H_corr
    
    def fusion_heatmaps(self, 
                       H_gap: torch.Tensor, 
                       H_corr: torch.Tensor) -> torch.Tensor:
        """
        Step 4: 融合原始热图和相关热图
        
        Args:
            H_gap: 原始缺口热图 [B,1,64,128]
            H_corr: 相关热图 [B,1,64,128]
        
        Returns:
            H_gap_final: 融合后的最终热图 [B,1,64,128]
        """
        # 拼接两个热图
        combined = torch.cat([H_gap, H_corr], dim=1)  # [B, 2, 64, 128]
        
        # 通过融合网络
        H_gap_final = self.fusion_net(combined)  # [B, 1, 64, 128]
        
        return H_gap_final
    
    def forward(self, 
                Hf: torch.Tensor,
                H_piece: torch.Tensor,
                H_gap: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            Hf: 特征图 [B,128,64,128]
            H_piece: 拼图中心热图 [B,1,64,128]
            H_gap: 原始缺口中心热图 [B,1,64,128]
        
        Returns:
            outputs: 包含以下键值的字典
                - 'heatmap_gap_final': 增强后的缺口热图 [B,1,64,128]
                - 'heatmap_corr': 相关热图 [B,1,64,128]
                - 'template': 提取的模板 [B,128,16,16] (用于可视化)
        """
        # Step 1: 提取滑块模板
        R_piece = self.extract_template(Hf, H_piece)
        
        # Step 2: 生成动态卷积核
        W = self.generate_dynamic_kernel(R_piece)
        
        # Step 3: 相关搜索
        H_corr = self.cross_correlation(Hf, W)
        
        # Step 4: 融合热图
        H_gap_final = self.fusion_heatmaps(H_gap, H_corr)
        
        # 返回结果
        outputs = {
            'heatmap_gap_final': H_gap_final,
            'heatmap_corr': H_corr,
            'template': R_piece  # 可用于调试和可视化
        }
        
        return outputs


def create_tamh(config: dict):
    """
    创建TAMH模块的工厂函数
    
    Args:
        config: 配置字典，应包含:
            - feature_channels: 特征通道数
            - template_size: 模板大小
            - kernel_size: 动态卷积核大小
            - fusion_channels: 融合层通道数
            - use_l2_norm: 是否使用L2归一化
            - padding_mode: padding模式
    
    Returns:
        TAMH模块实例
    """
    # 默认配置
    default_config = {
        'feature_channels': 128,
        'template_size': 16,
        'kernel_size': 3,
        'fusion_channels': 32,
        'use_l2_norm': True,
        'padding_mode': 'replicate'
    }
    
    # 合并用户配置
    final_config = {**default_config, **config}
    
    return TAMH(
        feature_channels=final_config['feature_channels'],
        template_size=final_config['template_size'],
        kernel_size=final_config['kernel_size'],
        fusion_channels=final_config['fusion_channels'],
        use_l2_norm=final_config['use_l2_norm'],
        padding_mode=final_config['padding_mode']
    )