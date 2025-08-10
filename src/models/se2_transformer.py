"""
PMN-R3-FP SE(2) Cross-Attention Transformer Module
SE(2)群等变交叉注意力变换器 - 用于滑块-缺口匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            x + pe: [B, N, D]
        """
        return x + self.pe[:, :x.size(1)]


class SE2Attention(nn.Module):
    """SE(2)等变注意力机制"""
    
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 旋转编码 (用于SE(2)等变性)
        self.rotation_embedding = nn.Parameter(torch.randn(1, n_heads, 1, self.d_k))
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, N_q, D] - 查询序列
            key: [B, N_k, D] - 键序列
            value: [B, N_k, D] - 值序列
            mask: [B, N_q, N_k] - 注意力掩码
        Returns:
            output: [B, N_q, D]
            attention: [B, n_heads, N_q, N_k]
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(B, N_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, N_q, d_k]
        K = self.w_k(key).view(B, N_k, self.n_heads, self.d_k).transpose(1, 2)    # [B, n_heads, N_k, d_k]
        V = self.w_v(value).view(B, N_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, N_k, d_k]
        
        # 添加旋转编码 (SE(2)等变性)
        Q = Q + self.rotation_embedding
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, N_q, N_k]
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力权重
        context = torch.matmul(attention, V)  # [B, n_heads, N_q, d_k]
        
        # 重塑并输出投影
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        output = self.w_o(context)
        
        return output, attention


class SE2CrossAttention(nn.Module):
    """SE(2)交叉注意力层"""
    
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.self_attn = SE2Attention(d_model, n_heads, dropout)
        self.cross_attn = SE2Attention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, memory, query_mask=None, memory_mask=None):
        """
        Args:
            query: [B, N_q, D] - 查询序列（滑块特征）
            memory: [B, N_m, D] - 记忆序列（缺口特征）
            query_mask: 查询掩码
            memory_mask: 记忆掩码
        Returns:
            output: [B, N_q, D]
        """
        # 自注意力
        residual = query
        query_norm = self.norm1(query)
        self_output, _ = self.self_attn(query_norm, query_norm, query_norm, query_mask)
        query = residual + self.dropout(self_output)
        
        # 交叉注意力
        residual = query
        query_norm = self.norm2(query)
        cross_output, cross_attention = self.cross_attn(query_norm, memory, memory, memory_mask)
        query = residual + self.dropout(cross_output)
        
        # 前馈网络
        residual = query
        query_norm = self.norm3(query)
        ff_output = self.feed_forward(query_norm)
        output = residual + ff_output
        
        return output, cross_attention


class SE2Transformer(nn.Module):
    """SE(2)变换器 - 完整的滑块-缺口匹配模块"""
    
    def __init__(self, d_model=256, n_heads=8, n_layers=3, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 特征投影
        self.piece_proj = nn.Linear(64*64, d_model)  # 滑块ROI特征投影
        self.gap_proj = nn.Linear(64*64, d_model)     # 缺口ROI特征投影
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # SE(2)交叉注意力层
        self.layers = nn.ModuleList([
            SE2CrossAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # 匹配得分预测头
        self.match_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 几何参数预测头
        self.geometry_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # (dx, dy, dθ)
        )
    
    def forward(self, piece_features, gap_features):
        """
        Args:
            piece_features: [B, N_piece, C, H, W] - 滑块ROI特征
            gap_features: [B, N_gap, C, H, W] - 缺口ROI特征
        Returns:
            Dict containing:
                - match_scores: [B, N_piece, N_gap] - 匹配得分矩阵
                - geometry: [B, N_piece, N_gap, 3] - 几何变换参数
                - attention_maps: List of attention maps
        """
        B, N_piece, C, H, W = piece_features.shape
        N_gap = gap_features.shape[1]
        
        # 展平并投影特征
        piece_flat = piece_features.view(B, N_piece, -1)  # [B, N_piece, C*H*W]
        gap_flat = gap_features.view(B, N_gap, -1)        # [B, N_gap, C*H*W]
        
        piece_tokens = self.piece_proj(piece_flat)  # [B, N_piece, d_model]
        gap_tokens = self.gap_proj(gap_flat)        # [B, N_gap, d_model]
        
        # 添加位置编码
        piece_tokens = self.pos_encoding(piece_tokens)
        gap_tokens = self.pos_encoding(gap_tokens)
        
        # 保存注意力图
        attention_maps = []
        
        # 通过SE(2)交叉注意力层
        for layer in self.layers:
            piece_tokens, cross_attn = layer(piece_tokens, gap_tokens)
            attention_maps.append(cross_attn)
        
        # 计算匹配得分和几何参数
        match_scores = []
        geometry_params = []
        
        for i in range(N_piece):
            piece_token = piece_tokens[:, i:i+1, :]  # [B, 1, d_model]
            
            # 与每个缺口计算匹配得分
            piece_expanded = piece_token.expand(-1, N_gap, -1)  # [B, N_gap, d_model]
            
            # 匹配得分
            scores = self.match_head(piece_expanded).squeeze(-1)  # [B, N_gap]
            match_scores.append(scores)
            
            # 几何参数
            geom = self.geometry_head(piece_expanded)  # [B, N_gap, 3]
            geometry_params.append(geom)
        
        match_scores = torch.stack(match_scores, dim=1)  # [B, N_piece, N_gap]
        geometry_params = torch.stack(geometry_params, dim=1)  # [B, N_piece, N_gap, 3]
        
        return {
            'match_scores': torch.sigmoid(match_scores),
            'geometry': geometry_params,
            'attention_maps': attention_maps
        }


class GeometricRefinement(nn.Module):
    """几何精修模块"""
    
    def __init__(self, d_model=256):
        super().__init__()
        
        self.refine_net = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3)  # 精修的(dx, dy, dθ)
        )
    
    def forward(self, features, initial_geometry):
        """
        Args:
            features: [B, N, D] - 特征
            initial_geometry: [B, N, 3] - 初始几何参数
        Returns:
            refined_geometry: [B, N, 3] - 精修后的几何参数
        """
        # 拼接特征和初始几何参数
        combined = torch.cat([features, initial_geometry], dim=-1)
        
        # 预测残差
        residual = self.refine_net(combined)
        
        # 添加残差
        refined_geometry = initial_geometry + residual * 0.1  # 缩放残差
        
        return refined_geometry


if __name__ == "__main__":
    # 测试代码
    
    # 测试SE(2)注意力
    se2_attn = SE2Attention(d_model=256, n_heads=8)
    query = torch.randn(2, 10, 256)
    key = torch.randn(2, 15, 256)
    value = torch.randn(2, 15, 256)
    
    output, attention = se2_attn(query, key, value)
    print("SE2 Attention Output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention shape: {attention.shape}")
    
    # 测试SE(2)变换器
    transformer = SE2Transformer(d_model=256, n_heads=8, n_layers=3)
    piece_features = torch.randn(2, 5, 32, 8, 8)  # 5个滑块
    gap_features = torch.randn(2, 10, 32, 8, 8)   # 10个缺口
    
    result = transformer(piece_features, gap_features)
    print("\nSE2 Transformer Output:")
    print(f"  Match scores shape: {result['match_scores'].shape}")
    print(f"  Geometry shape: {result['geometry'].shape}")
    print(f"  Number of attention maps: {len(result['attention_maps'])}")
    
    # 测试几何精修
    refiner = GeometricRefinement(d_model=256)
    features = torch.randn(2, 5, 256)
    initial_geom = torch.randn(2, 5, 3)
    
    refined = refiner(features, initial_geom)
    print(f"\nRefined geometry shape: {refined.shape}")