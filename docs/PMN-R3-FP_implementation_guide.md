# PMN-R3-FP 神经网络实现指南

## PuzzleMatchNet - Robust & Refined with FPN+PAN

### 文档版本
- **版本**: 1.0
- **日期**: 2025-01-10
- **架构名称**: PMN-R3-FP (PuzzleMatchNet-Robust & Refined with FPN+PAN)
- **目标任务**: 高精度滑块验证码识别与匹配

---

## 1. 架构总览

### 1.1 核心设计原则
- **输入规范**: 固定分辨率 H=256, W=512
- **张量格式**: [B, C, H, W] (批次, 通道, 高度, 宽度)
- **ROI规范**: 64×64 标准化尺寸
- **激活函数**: 所有卷积后默认接 BN+ReLU (除特别说明)

### 1.2 架构组成
1. **输入预处理层**: 多模态输入融合
2. **共享Stem**: 特征提取入口
3. **HR-Backbone**: 高分辨率并行主干网络
4. **Region支路**: FPN+PAN双向特征金字塔
5. **Shape支路**: 高分辨率直通+SDF预测
6. **Proposal生成**: 候选区域提取
7. **SE(2)-CAT**: 跨注意力变换匹配
8. **几何精修器**: 亚像素级别优化
9. **排序判别器**: 真实缺口筛选

---

## 2. 输入与预处理 (Layer 0)

### 2.1 输入通道定义

```python
class InputPreprocessor:
    """
    输入预处理器
    将RGB图像与辅助通道融合
    """
    def __init__(self):
        self.coord_conv = CoordConv2d()  # 坐标编码生成器
        
    def forward(self, rgb_input):
        """
        Args:
            rgb_input: [B, 3, 256, 512] - RGB输入图像
            
        Returns:
            X0: [B, 6, 256, 512] - 融合后的输入张量
        """
        # 主输入
        X_rgb = rgb_input  # [B, 3, 256, 512]
        
        # 坐标编码 (x,y ∈ [-1,1])
        X_coord = self.coord_conv(X_rgb)  # [B, 2, 256, 512]
        
        # Padding掩码 (黑/白padding=1, 其他=0)
        X_pad = self.generate_padding_mask(X_rgb)  # [B, 1, 256, 512]
        
        # 通道拼接
        X0 = torch.cat([X_rgb, X_coord, X_pad], dim=1)  # [B, 6, 256, 512]
        
        return X0
```

### 2.2 固定边缘先验 (仅Shape支路使用)

```python
class EdgePriorExtractor:
    """
    固定边缘先验提取器
    生成Sobel和LoG边缘特征
    """
    def __init__(self):
        self.sobel_x = self._init_sobel_x()  # Gx核
        self.sobel_y = self._init_sobel_y()  # Gy核
        self.log = self._init_log()          # LoG核
        
    def forward(self, X_rgb):
        """
        Returns:
            E_full: [B, 3, 256, 512] - 全分辨率边缘
            E_1/2:  [B, 3, 128, 256] - 1/2分辨率
            E_1/4:  [B, 3, 64, 128]  - 1/4分辨率
        """
        # Sobel边缘 (深度可分离)
        edge_sobel = self.apply_sobel(X_rgb)  # [B, 2, 256, 512]
        
        # LoG边缘
        edge_log = self.apply_log(X_rgb)  # [B, 1, 256, 512]
        
        # 拼接
        E_full = torch.cat([edge_sobel, edge_log], dim=1)  # [B, 3, 256, 512]
        
        # 多尺度下采样
        E_half = F.avg_pool2d(E_full, 2)  # [B, 3, 128, 256]
        E_quarter = F.avg_pool2d(E_half, 2)  # [B, 3, 64, 128]
        
        return E_full, E_half, E_quarter
```

---

## 3. 共享Stem (Layer 1)

```python
class SharedStem(nn.Module):
    """
    共享特征提取stem
    两级下采样到1/4分辨率
    """
    def __init__(self):
        super().__init__()
        # 第一级: 7×7卷积, stride=2
        self.conv7x7_s2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B, 6, 256, 512] → [B, 64, 128, 256]
        
        # 第二级: 3×3卷积, stride=2
        self.conv3x3_s2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B, 64, 128, 256] → [B, 64, 64, 128]
        
    def forward(self, X0):
        """
        Args:
            X0: [B, 6, 256, 512]
            
        Returns:
            stem_out: [B, 64, 64, 128] - 主输出
            F_stem: [B, 64, 128, 256] - 浅层特征(供Shape支路)
        """
        F_stem = self.conv7x7_s2(X0)  # [B, 64, 128, 256]
        stem_out = self.conv3x3_s2(F_stem)  # [B, 64, 64, 128]
        
        return stem_out, F_stem
```

---

## 4. HR-Backbone (Layer 3)

### 4.1 基础块定义

```python
class BasicBlock(nn.Module):
    """
    HRNet基础残差块
    两个3×3卷积 + 残差连接
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out
```

### 4.2 多分辨率并行Stage

```python
class HRBackbone(nn.Module):
    """
    高分辨率并行主干网络
    4个Stage, 逐步增加并行分支
    """
    def __init__(self):
        super().__init__()
        
        # Stage1: 单分支 (1/4分辨率)
        self.stage1 = self._make_stage(64, 3)  # BasicBlock(64) × 3
        
        # Transition2: 生成双分支
        self.trans2_b1 = nn.Conv2d(64, 32, 1, 1, 0)  # 1×1卷积
        self.trans2_b2 = nn.Conv2d(64, 64, 3, 2, 1)  # 3×3_s2下采样
        
        # Stage2: 双分支 (1/4 & 1/8)
        self.stage2_b1 = self._make_stage(32, 4)  # [B, 32, 64, 128]
        self.stage2_b2 = self._make_stage(64, 4)  # [B, 64, 32, 64]
        self.fuse2 = MultiBranchFusion(2)
        
        # Transition3: 新增1/16分支
        self.trans3_b3 = nn.Conv2d(64, 128, 3, 2, 1)
        
        # Stage3: 三分支 (1/4, 1/8, 1/16)
        self.stage3_b1 = self._make_stage(32, 4)   # [B, 32, 64, 128]
        self.stage3_b2 = self._make_stage(64, 4)   # [B, 64, 32, 64]
        self.stage3_b3 = self._make_stage(128, 4)  # [B, 128, 16, 32]
        self.fuse3 = MultiBranchFusion(3)
        
        # Transition4: 新增1/32分支
        self.trans4_b4 = nn.Conv2d(128, 256, 3, 2, 1)
        
        # Stage4: 四分支 (1/4, 1/8, 1/16, 1/32)
        self.stage4_b1 = self._make_stage(32, 4)   # [B, 32, 64, 128]
        self.stage4_b2 = self._make_stage(64, 4)   # [B, 64, 32, 64]
        self.stage4_b3 = self._make_stage(128, 4)  # [B, 128, 16, 32]
        self.stage4_b4 = self._make_stage(256, 4)  # [B, 256, 8, 16]
        self.fuse4 = MultiBranchFusion(4)
        
    def forward(self, stem_out):
        """
        Args:
            stem_out: [B, 64, 64, 128]
            
        Returns:
            H2: [B, 32, 64, 128]  - 1/4分辨率
            H3: [B, 64, 32, 64]   - 1/8分辨率
            H4: [B, 128, 16, 32]  - 1/16分辨率
            H5: [B, 256, 8, 16]   - 1/32分辨率
        """
        # Stage1
        s1_out = self.stage1(stem_out)  # [B, 64, 64, 128]
        
        # Transition2
        F2_1_4 = self.trans2_b1(s1_out)  # [B, 32, 64, 128]
        F2_1_8 = self.trans2_b2(s1_out)  # [B, 64, 32, 64]
        
        # Stage2
        s2_b1 = self.stage2_b1(F2_1_4)
        s2_b2 = self.stage2_b2(F2_1_8)
        F3_1_4, F3_1_8 = self.fuse2([s2_b1, s2_b2])
        
        # ... Stage3和Stage4类似处理 ...
        
        return H2, H3, H4, H5
```

---

## 5. Region/Proposal支路 (Layer 4)

### 5.1 FPN+PAN双向金字塔

```python
class RegionBranch(nn.Module):
    """
    Region分支: FPN+PAN串联架构
    用于生成多尺度区域特征
    """
    def __init__(self):
        super().__init__()
        
        # 侧向连接: 统一到128通道
        self.lateral2 = nn.Conv2d(32, 128, 1)   # H2 → P2
        self.lateral3 = nn.Conv2d(64, 128, 1)   # H3 → P3
        self.lateral4 = nn.Conv2d(128, 128, 1)  # H4 → P4
        self.lateral5 = nn.Conv2d(256, 128, 1)  # H5 → P5
        
        # Top-Down FPN路径
        self.td5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.td4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.td3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.td2 = nn.Conv2d(128, 128, 3, 1, 1)
        
        # Bottom-Up PAN路径
        self.bu2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bu3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bu4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bu5 = nn.Conv2d(128, 128, 3, 1, 1)
        
        # 上采样到全分辨率
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # [B, 128, 64, 128] → [B, 64, 128, 256]
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # [B, 64, 128, 256] → [B, 64, 256, 512]
        
    def forward(self, H2, H3, H4, H5):
        """
        Args:
            H2-H5: 来自Backbone的多尺度特征
            
        Returns:
            R_full: [B, 64, 256, 512] - 全分辨率区域特征
            N2-N5: PAN输出特征 (用于其他支路)
        """
        # 侧向连接
        P2 = self.lateral2(H2)  # [B, 128, 64, 128]
        P3 = self.lateral3(H3)  # [B, 128, 32, 64]
        P4 = self.lateral4(H4)  # [B, 128, 16, 32]
        P5 = self.lateral5(H5)  # [B, 128, 8, 16]
        
        # Top-Down FPN
        P5_td = self.td5(P5)
        P4_td = self.td4(self._upsample_add(P5_td, P4))
        P3_td = self.td3(self._upsample_add(P4_td, P3))
        P2_td = self.td2(self._upsample_add(P3_td, P2))
        
        # Bottom-Up PAN
        N2 = self.bu2(P2_td)
        N3 = self.bu3(self._downsample_add(N2, P3_td))
        N4 = self.bu4(self._downsample_add(N3, P4_td))
        N5 = self.bu5(self._downsample_add(N4, P5_td))
        
        # 上采样到全分辨率
        R_up1 = self.upsample1(N2)  # [B, 64, 128, 256]
        R_full = self.upsample2(R_up1)  # [B, 64, 256, 512]
        
        return R_full, (N2, N3, N4, N5)
```

### 5.2 分割头

```python
class SegmentationHeads(nn.Module):
    """
    区域分割头
    生成拼图、缺口、环境掩码
    """
    def __init__(self):
        super().__init__()
        self.head_puzzle = nn.Conv2d(64, 1, 1)  # M_p
        self.head_gap = nn.Conv2d(64, 1, 1)     # M_g
        self.head_env = nn.Conv2d(64, 1, 1)     # E
        
    def forward(self, R_full_gated):
        """
        Args:
            R_full_gated: [B, 64, 256, 512] - 门控后的区域特征
            
        Returns:
            M_p: [B, 1, 256, 512] - 拼图掩码
            M_g: [B, 1, 256, 512] - 缺口掩码
            E:   [B, 1, 256, 512] - 环境掩码
        """
        M_p = torch.sigmoid(self.head_puzzle(R_full_gated))
        M_g = torch.sigmoid(self.head_gap(R_full_gated))
        E = torch.sigmoid(self.head_env(R_full_gated))
        
        return M_p, M_g, E
```

---

## 6. Shape/SDF支路 (Layer 5)

### 6.1 高分辨率直通网络

```python
class ShapeBranch(nn.Module):
    """
    Shape支路: 保持高分辨率的形状预测
    包含边界热图和SDF预测
    """
    def __init__(self):
        super().__init__()
        
        # 多尺度融合到1/4
        self.fuse_1_8 = nn.Conv2d(64, 32, 1)
        self.fuse_1_16 = nn.Conv2d(128, 32, 1)
        self.fuse_1_32 = nn.Conv2d(256, 32, 1)
        
        # 边缘融合
        self.edge_fusion = nn.Conv2d(35, 64, 3, 1, 1)  # 32+3边缘
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(67, 64, 4, 2, 1),  # 64+3边缘
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 输出头
        self.boundary_head = nn.Conv2d(64, 2, 1)  # B_p, B_g
        self.sdf_head = nn.Conv2d(64, 2, 1)       # D_p, D_g
        self.center_head = nn.Conv2d(64, 2, 1)    # C_p, C_g (可选)
        
    def forward(self, H2, H3, H4, H5, E_full, E_half, E_quarter):
        """
        Args:
            H2-H5: Backbone特征
            E_*: 边缘先验特征
            
        Returns:
            S_full: [B, 64, 256, 512] - 全分辨率形状特征
            B_out:  [B, 2, 256, 512]  - 边界热图
            D_out:  [B, 2, 256, 512]  - SDF
            C_out:  [B, 2, 256, 512]  - 中心热图(可选)
        """
        # 多尺度上采样到1/4
        S2_in = H2  # 已经是1/4
        S2_in += F.interpolate(self.fuse_1_8(H3), scale_factor=2)
        S2_in += F.interpolate(self.fuse_1_16(H4), scale_factor=4)
        S2_in += F.interpolate(self.fuse_1_32(H5), scale_factor=8)
        # S2_in: [B, 32, 64, 128]
        
        # 拼接边缘先验
        S2_cat = torch.cat([S2_in, E_quarter], dim=1)  # [B, 35, 64, 128]
        S2 = self.edge_fusion(S2_cat)  # [B, 64, 64, 128]
        
        # 上采样到1/2
        S_half = self.up1(S2)  # [B, 64, 128, 256]
        S_half_cat = torch.cat([S_half, E_half], dim=1)  # [B, 67, 128, 256]
        
        # 上采样到全分辨率
        S_full = self.up2(S_half_cat)  # [B, 64, 256, 512]
        
        # 输出头
        B_out = torch.sigmoid(self.boundary_head(S_full))  # [B, 2, 256, 512]
        D_out = torch.tanh(self.sdf_head(S_full))  # [B, 2, 256, 512]
        C_out = torch.sigmoid(self.center_head(S_full))  # [B, 2, 256, 512]
        
        return S_full, B_out, D_out, C_out
```

### 6.2 纹理门控机制

```python
class TextureGate(nn.Module):
    """
    外观门控机制
    动态融合Region和Shape特征
    """
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, R_full, S_full):
        """
        Args:
            R_full: [B, 64, 256, 512] - 区域特征
            S_full: [B, 64, 256, 512] - 形状特征
            
        Returns:
            R_full_gated: [B, 64, 256, 512] - 门控后的特征
        """
        # 计算门控权重
        alpha = self.detector(R_full)  # [B, 1]
        alpha = alpha.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 加权融合
        R_full_gated = R_full * alpha + S_full * (1 - alpha)
        
        return R_full_gated
```

---

## 7. Proposal生成 (Layer 6)

```python
class ProposalGenerator(nn.Module):
    """
    候选区域生成器
    从分割和边界图中提取候选缺口
    """
    def __init__(self, max_proposals=4):
        super().__init__()
        self.max_proposals = max_proposals
        self.area_threshold = 100  # 最小面积阈值
        self.nms_threshold = 0.3   # NMS IoU阈值
        
    def forward(self, M_g, B_g, D_g, E):
        """
        Args:
            M_g: [B, 1, 256, 512] - 缺口掩码
            B_g: [B, 1, 256, 512] - 缺口边界
            D_g: [B, 1, 256, 512] - 缺口SDF
            E:   [B, 1, 256, 512] - 环境掩码
            
        Returns:
            proposals: List[Dict] - 每个包含(x,y,w,h,θ0)的候选
        """
        batch_proposals = []
        
        for b in range(M_g.size(0)):
            # 连通域分析
            regions = self._extract_connected_components(M_g[b])
            
            # 面积筛选
            valid_regions = [r for r in regions if r['area'] > self.area_threshold]
            
            # 边界细化
            for region in valid_regions:
                # 使用B_g细化边界
                refined_contour = self._refine_boundary(region, B_g[b])
                
                # 计算旋转矩形
                x, y, w, h, theta = self._fit_rotated_rect(refined_contour)
                
                # 计算置信度
                sdf_consistency = self._compute_sdf_consistency(region, D_g[b])
                env_overlap = self._compute_env_overlap(region, E[b])
                confidence = sdf_consistency * (1 - env_overlap)
                
                region.update({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'theta': theta, 'confidence': confidence
                })
            
            # NMS
            proposals = self._nms(valid_regions, self.nms_threshold)
            
            # 保留Top-K
            proposals = sorted(proposals, key=lambda x: x['confidence'], reverse=True)
            proposals = proposals[:self.max_proposals]
            
            batch_proposals.append(proposals)
        
        return batch_proposals
```

---

## 8. SE(2)-CAT匹配 (Layer 7)

### 8.1 ROI特征提取

```python
class ROIFeatureExtractor(nn.Module):
    """
    ROI特征提取器
    将拼图和缺口区域规范化到64×64
    """
    def __init__(self, roi_size=64):
        super().__init__()
        self.roi_size = roi_size
        self.roi_align = ROIAlign((roi_size, roi_size), 1.0, -1)
        
    def forward(self, B_p, D_p, B_g, D_g, proposals):
        """
        Args:
            B_p, D_p: 拼图边界和SDF
            B_g, D_g: 缺口边界和SDF
            proposals: 候选区域列表
            
        Returns:
            P_roi: [K, 2, 64, 64] - 拼图ROI特征
            G_roi: [K, 2, 64, 64] - 缺口ROI特征
        """
        P_features = []
        G_features = []
        
        for prop in proposals:
            # 提取拼图ROI (假设已知拼图位置)
            p_roi_b = self.roi_align(B_p, prop['puzzle_box'])
            p_roi_d = self.roi_align(D_p, prop['puzzle_box'])
            P_roi = torch.cat([p_roi_b, p_roi_d], dim=1)  # [1, 2, 64, 64]
            
            # 提取缺口ROI
            g_roi_b = self.roi_align(B_g, prop['gap_box'])
            g_roi_d = self.roi_align(D_g, prop['gap_box'])
            G_roi = torch.cat([g_roi_b, g_roi_d], dim=1)  # [1, 2, 64, 64]
            
            P_features.append(P_roi)
            G_features.append(G_roi)
        
        return torch.cat(P_features, dim=0), torch.cat(G_features, dim=0)
```

### 8.2 角度旋转与互相关

```python
class AngleCorrelation(nn.Module):
    """
    多角度互相关计算
    通过STN旋转拼图到不同角度并计算NCC
    """
    def __init__(self, angle_bins=9):
        super().__init__()
        self.angle_bins = angle_bins
        # 角度桶: -20, -10, -5, -2, 0, 2, 5, 10, 20度
        self.angles = torch.tensor([-20, -10, -5, -2, 0, 2, 5, 10, 20])
        self.stn = SpatialTransformer()
        
    def forward(self, P_roi, G_roi):
        """
        Args:
            P_roi: [B, 2, 64, 64] - 拼图ROI
            G_roi: [B, 2, 64, 64] - 缺口ROI
            
        Returns:
            CorrStack: [B, 9, 64, 64] - 角度相关堆叠
            coarse_params: (Δx_c, Δy_c, θ_c) - 粗对齐参数
        """
        B = P_roi.size(0)
        corr_maps = []
        
        for angle in self.angles:
            # 旋转拼图
            theta_mat = self._angle_to_matrix(angle)
            P_rotated = self.stn(P_roi, theta_mat)
            
            # 计算NCC
            corr = self._compute_ncc(P_rotated, G_roi)  # [B, 1, 64, 64]
            corr_maps.append(corr)
        
        CorrStack = torch.cat(corr_maps, dim=1)  # [B, 9, 64, 64]
        
        # Soft-argmax获取粗对齐
        max_corr, max_idx = torch.max(CorrStack.view(B, 9, -1), dim=2)
        best_angle_idx = torch.argmax(torch.mean(max_corr, dim=1))
        
        # DSNT获取平移
        best_corr = CorrStack[:, best_angle_idx]
        dx_c, dy_c = self._dsnt_argmax(best_corr)
        theta_c = self.angles[best_angle_idx]
        
        return CorrStack, (dx_c, dy_c, theta_c)
```

### 8.3 跨注意力Transformer

```python
class SE2CrossAttentionTransformer(nn.Module):
    """
    SE(2)跨注意力Transformer
    精细化姿态估计
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(10, d_model, 4, 4)  # 64×64 → 16×16
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, d_model))
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads) 
            for _ in range(n_layers)
        ])
        
        # 姿态回归头
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Δx_f, Δy_f, sin(θ), cos(θ), confidence
        )
        
    def forward(self, P_roi_stack, G_roi, R_full_gated=None):
        """
        Args:
            P_roi_stack: [B, 9, 2, 64, 64] - 多角度拼图
            G_roi: [B, 2, 64, 64] - 缺口ROI
            R_full_gated: 可选的纹理特征
            
        Returns:
            refined_pose: (Δx_f, Δy_f, Δθ_f, s, p)
        """
        B = G_roi.size(0)
        
        # 可选: 添加纹理特征
        if R_full_gated is not None:
            tex_feat = self._extract_texture(R_full_gated)  # [B, 8, 64, 64]
            P_roi_stack = torch.cat([P_roi_stack, tex_feat.unsqueeze(1).expand(-1, 9, -1, -1, -1)], dim=2)
            G_roi = torch.cat([G_roi, tex_feat], dim=1)
        
        # Patch Embedding
        # Query from Gap
        G_patches = self.patch_embed(G_roi)  # [B, 256, 16, 16]
        G_patches = G_patches.flatten(2).transpose(1, 2)  # [B, 256, 256]
        
        # Key/Value from Puzzle (多角度)
        P_patches = []
        for i in range(9):
            p = self.patch_embed(P_roi_stack[:, i])
            p = p.flatten(2).transpose(1, 2)
            P_patches.append(p)
        P_patches = torch.cat(P_patches, dim=1)  # [B, 9*256, 256]
        
        # 添加位置编码
        G_patches += self.pos_encoding
        
        # Transformer处理
        query = G_patches
        for layer in self.transformer_layers:
            query = layer(query, P_patches)  # 跨注意力
        
        # 全局池化 + 回归
        feat = torch.mean(query, dim=1)  # [B, 256]
        pose = self.pose_head(feat)  # [B, 5]
        
        # 解析输出
        dx_f = torch.tanh(pose[:, 0])  # [-1, 1]
        dy_f = torch.tanh(pose[:, 1])  # [-1, 1]
        sin_theta = pose[:, 2]
        cos_theta = pose[:, 3]
        theta_f = torch.atan2(sin_theta, cos_theta)
        confidence = torch.sigmoid(pose[:, 4])
        
        return dx_f, dy_f, theta_f, 1.0, confidence
```

---

## 9. 几何精修器 (Layer 7.4)

```python
class GeometricRefiner(nn.Module):
    """
    可微几何精修器
    使用Chamfer距离和最优传输进行亚像素优化
    """
    def __init__(self, n_points=256, sinkhorn_iters=20):
        super().__init__()
        self.n_points = n_points
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = 0.05
        
        # 精修MLP
        self.refine_mlp = nn.Sequential(
            nn.Linear(4, 64),  # energy + chamfer + normal + ot
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # δx_r, δy_r
        )
        
    def forward(self, B_p, D_g, coarse_pose, fine_pose):
        """
        Args:
            B_p: 拼图边界
            D_g: 缺口SDF
            coarse_pose: (Δx_c, Δy_c, θ_c)
            fine_pose: (Δx_f, Δy_f, Δθ_f, s)
            
        Returns:
            refined_pose: 最终精修姿态
            energy: 几何能量
        """
        # 合成变换参数
        total_dx = coarse_pose[0] + fine_pose[0]
        total_dy = coarse_pose[1] + fine_pose[1]
        total_theta = coarse_pose[2] + fine_pose[2]
        scale = fine_pose[3]
        
        # 从边界采样点集
        puzzle_points = self._sample_boundary_points(B_p, self.n_points)
        
        # 应用变换
        transformed_points = self._apply_transform(
            puzzle_points, total_dx, total_dy, total_theta, scale
        )
        
        # 在SDF上评估
        sdf_values = self._evaluate_sdf(transformed_points, D_g)
        
        # 计算Chamfer距离
        chamfer_dist = self._compute_chamfer(transformed_points, D_g)
        
        # 计算法向一致性
        normal_consistency = self._compute_normal_consistency(
            transformed_points, D_g
        )
        
        # Sinkhorn最优传输
        ot_cost = self._sinkhorn_ot(transformed_points, D_g)
        
        # 特征向量
        features = torch.stack([
            chamfer_dist,
            normal_consistency,
            ot_cost,
            torch.mean(torch.abs(sdf_values))
        ], dim=1)  # [B, 4]
        
        # MLP精修
        delta = self.refine_mlp(features)  # [B, 2]
        delta_x_r = delta[:, 0] * 0.1  # 缩放到亚像素
        delta_y_r = delta[:, 1] * 0.1
        
        # 最终姿态
        final_dx = total_dx + delta_x_r
        final_dy = total_dy + delta_y_r
        final_theta = total_theta
        
        # 能量
        energy = torch.mean(features, dim=1)
        
        return (final_dx, final_dy, final_theta, scale), energy
```

---

## 10. 排序判别器 (Layer 8)

```python
class RankingHead(nn.Module):
    """
    真实缺口判别和排序
    从多个候选中选择最佳匹配
    """
    def __init__(self, feature_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, candidate_features):
        """
        Args:
            candidate_features: List[Tensor] - 每个候选的特征
                包含: confidence, energy, IoU, SDF_overlap, area_ratio等
                
        Returns:
            best_idx: 最佳候选索引
            scores: 所有候选的得分
        """
        scores = []
        
        for feat in candidate_features:
            # 构建特征向量
            # [confidence, energy, IoU, SDF_overlap, area/perimeter, ...]
            score = self.mlp(feat)
            scores.append(score)
        
        scores = torch.cat(scores, dim=0)  # [K, 1]
        
        # 选择最高分
        best_idx = torch.argmax(scores)
        
        return best_idx, scores
```

---

## 11. 损失函数定义

```python
class PMNLoss(nn.Module):
    """
    PMN-R3-FP完整损失函数
    包含分割、边界、SDF、姿态、排序等多个组件
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.l1 = nn.L1Loss()
        self.huber = nn.HuberLoss()
        
        # 损失权重
        self.weights = {
            'seg': 1.0,      # 分割
            'boundary': 2.0,  # 边界
            'sdf': 1.5,      # SDF
            'pose': 3.0,     # 姿态
            'energy': 0.5,   # 几何能量
            'rank': 1.0,     # 排序
            'consistency': 0.1  # 一致性
        }
        
    def forward(self, predictions, targets):
        """
        计算总损失
        """
        losses = {}
        
        # 1. 分割损失 (BCE + Dice)
        seg_loss = self.bce(predictions['M_p'], targets['M_p_gt'])
        seg_loss += self.bce(predictions['M_g'], targets['M_g_gt'])
        seg_loss += self.bce(predictions['E'], targets['E_gt'])
        seg_loss += self.dice(predictions['M_p'], targets['M_p_gt'])
        seg_loss += self.dice(predictions['M_g'], targets['M_g_gt'])
        losses['seg'] = seg_loss
        
        # 2. 边界损失 (Focal)
        boundary_loss = self.focal(predictions['B_p'], targets['B_p_gt'])
        boundary_loss += self.focal(predictions['B_g'], targets['B_g_gt'])
        losses['boundary'] = boundary_loss
        
        # 3. SDF损失 (L1, 边界邻域加权)
        sdf_loss = self._weighted_sdf_loss(
            predictions['D_p'], targets['D_p_gt'],
            predictions['D_g'], targets['D_g_gt']
        )
        losses['sdf'] = sdf_loss
        
        # 4. 姿态损失 (Huber)
        pose_loss = self.huber(predictions['dx'], targets['dx_gt'])
        pose_loss += self.huber(predictions['dy'], targets['dy_gt'])
        # 角度用sin/cos回归
        pose_loss += self.huber(
            torch.sin(predictions['theta']), 
            torch.sin(targets['theta_gt'])
        )
        pose_loss += self.huber(
            torch.cos(predictions['theta']), 
            torch.cos(targets['theta_gt'])
        )
        losses['pose'] = pose_loss
        
        # 5. 几何能量损失
        losses['energy'] = predictions['energy'].mean()
        
        # 6. 排序损失 (ListNet)
        rank_loss = self._listnet_loss(
            predictions['ranks'], 
            targets['true_gap_idx']
        )
        losses['rank'] = rank_loss
        
        # 7. 一致性损失 (可选)
        if 'R_full' in predictions and 'S_full' in predictions:
            consistency_loss = self._consistency_loss(
                predictions['R_full'], 
                predictions['S_full']
            )
            losses['consistency'] = consistency_loss
        
        # 加权总和
        total_loss = sum(
            self.weights[k] * v for k, v in losses.items()
        )
        
        return total_loss, losses
    
    def _weighted_sdf_loss(self, pred_p, gt_p, pred_g, gt_g):
        """边界邻域加权的SDF损失"""
        # 边界附近(|SDF|<3px)权重更高
        weight_p = (torch.abs(gt_p) < 3).float() * 2 + 1
        weight_g = (torch.abs(gt_g) < 3).float() * 2 + 1
        
        loss_p = (weight_p * torch.abs(pred_p - gt_p)).mean()
        loss_g = (weight_g * torch.abs(pred_g - gt_g)).mean()
        
        return loss_p + loss_g
    
    def _listnet_loss(self, scores, true_idx):
        """ListNet排序损失"""
        # Softmax交叉熵
        log_probs = F.log_softmax(scores, dim=0)
        target = F.one_hot(true_idx, scores.size(0)).float()
        loss = -(target * log_probs).sum()
        return loss
    
    def _consistency_loss(self, R_full, S_full):
        """R/S分支一致性损失"""
        # 使用KL散度或L2距离
        return F.kl_div(
            F.log_softmax(R_full, dim=1),
            F.softmax(S_full, dim=1),
            reduction='batchmean'
        )
```

---

## 12. 训练配置与超参数

```python
class TrainingConfig:
    """
    PMN-R3-FP训练配置
    """
    # 模型架构
    BACKBONE = 'hrnet_w32'  # HR-like backbone
    FPN_CHANNELS = 128       # FPN/PAN统一通道数
    SHAPE_CHANNELS = 64      # Shape支路通道数
    
    # ROI配置
    ROI_SIZE = 64           # ROI规范化尺寸
    MAX_PROPOSALS = 4       # 最大候选数
    
    # 角度配置
    ANGLE_BINS = 9          # 角度桶数量
    ANGLE_RANGE = [-20, 20] # 角度范围
    
    # Transformer配置
    TRANSFORMER_DIM = 256   # 特征维度
    TRANSFORMER_HEADS = 8   # 注意力头数
    TRANSFORMER_LAYERS = 4  # 层数
    
    # 几何精修
    REFINER_POINTS = 256    # 边界采样点数
    SINKHORN_ITERS = 20     # Sinkhorn迭代次数
    SINKHORN_EPS = 0.05     # 正则化参数
    DSNT_TEMPERATURE = 0.1  # DSNT温度
    
    # 优化器
    OPTIMIZER = 'AdamW'
    BASE_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 学习率调度
    SCHEDULER = 'CosineAnnealingLR'
    T_MAX = 100
    ETA_MIN = 1e-6
    
    # 训练参数
    BATCH_SIZE = 8          # 根据GPU内存调整
    NUM_EPOCHS = 100
    GRADIENT_CLIP = 1.0
    
    # 数据增强
    AUGMENTATION = {
        'brightness': 0.2,
        'contrast': 0.2,
        'rotation': 5,       # 度
        'scale': (0.9, 1.1),
        'translation': 0.1   # 相对比例
    }
```

---

## 13. 推理流程

```python
class PMNInference:
    """
    PMN-R3-FP推理管道
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, image):
        """
        单张图像推理
        
        Args:
            image: [3, 256, 512] - 输入图像
            
        Returns:
            result: Dict - 包含最终预测的滑块位置
        """
        # 1. 预处理
        x = self.preprocess(image)
        
        # 2. 前向传播
        outputs = self.model(x)
        
        # 3. 提取候选
        proposals = outputs['proposals']
        
        # 4. 排序选择最佳
        best_idx = outputs['best_idx']
        best_proposal = proposals[best_idx]
        
        # 5. 获取最终姿态
        pose = outputs['poses'][best_idx]
        
        # 6. 转换到图像坐标
        slider_x = pose['dx'] * 512  # 缩放到原图
        slider_y = pose['dy'] * 256
        
        result = {
            'slider_position': (slider_x, slider_y),
            'gap_position': best_proposal['position'],
            'rotation': pose['theta'],
            'confidence': outputs['confidence'][best_idx],
            'all_proposals': proposals  # 用于可视化
        }
        
        return result
```

---

## 14. 部署优化建议

### 14.1 模型压缩
- **知识蒸馏**: 训练轻量学生网络
- **量化**: INT8量化减少内存占用
- **剪枝**: 移除冗余通道和连接

### 14.2 推理加速
- **ONNX导出**: 支持多平台部署
- **TensorRT优化**: GPU推理加速
- **批处理**: 充分利用并行计算

### 14.3 精度-速度权衡
- **快速模式**: 仅使用Region支路
- **精确模式**: 完整SE(2)-CAT流程
- **自适应模式**: 根据置信度动态选择

---

## 15. 常见问题与调试

### 15.1 收敛问题
- 检查学习率设置
- 验证数据标签正确性
- 逐步解冻各支路训练

### 15.2 过拟合
- 增强数据增强强度
- 引入Dropout层
- 减少模型容量

### 15.3 推理速度
- Profile各层耗时
- 优化ROI数量
- 简化Transformer层数

---

## 结语

本文档提供了PMN-R3-FP架构的完整实现指南。建议开发者：

1. **分阶段实现**: 先实现Backbone，再逐步添加各支路
2. **单元测试**: 验证每个模块的输入输出维度
3. **渐进训练**: 从简单任务开始，逐步增加复杂度
4. **监控指标**: 实时跟踪各损失分量的变化

如有疑问，请参考原始论文或联系架构设计团队。

---

**文档维护记录**
- 2025-01-10: 初始版本创建
- 架构版本: PMN-R3-FP v1.0
- 适用框架: PyTorch 2.0+