# Lite-HRNet-18 网络架构设计说明书 - 损失函数

## 0️⃣ 记号与标签生成（统一规范）

### 基础定义

- **原图大小**：`H×W = 256×512`
- **主特征图**：`H_f ∈ [B,128,64,128]`（**1/4 分辨率**）
- **真实中心（原图像素）**：
  - Gap缺口：`(x_g, y_g)`
  - Piece滑块：`(x_p, y_p)`

### 坐标映射

映射到1/4栅格坐标：

```python
(u_g, v_g) = (x_g/4, y_g/4)  # gap栅格坐标  
(u_p, v_p) = (x_p/4, y_p/4)  # piece栅格坐标
```

### 假缺口定义

来自生成器的混淆缺口：

```python
{(u_k, v_k)}_{k=1}^K, K=1..3  # 1-3个假缺口
```

### 高斯热图标签

使用栅格单位的σ生成热图：

```python
Y(i,j) = exp(−((i−v)² + (j−u)²)/(2σ²))
```

推荐参数：`σ=1.5`

### 子像素偏移标签

就地监督的连续偏移：

```python
δu = u − ⌊u⌋ ∈ [0,1)
δv = v − ⌊v⌋ ∈ [0,1)
```

训练时回归到`[−0.5, 0.5]`范围

### Padding Mask与权重掩码处理

#### 4通道输入结构

模型接收4通道输入张量 `[B, 4, H, W]`：

- **前3通道**：RGB图像（归一化到[0,1]）
- **第4通道**：Padding mask（1=有效区域，0=padding区域）

#### 权重掩码生成

```python
# 从输入第4通道提取并下采样
padding_mask = input[:, 3, :, :]  # [B, H, W]
weight_mask = AvgPool2d(k=4, s=4)(padding_mask)  # [B, H/4, W/4]
```

#### 实际实现细节

1. **数据预处理阶段**：
   - Letterbox变换生成padding mask
   - 将padding mask作为第4通道存储在图像NPY文件中
   - 不再单独保存weights.npy文件

2. **数据加载阶段**：
   - 从images.npy的第4通道提取padding mask
   - 使用平均池化下采样到1/4分辨率
   - 生成权重掩码供损失计算使用

> **核心原则**：所有1/4分辨率上的像素级损失逐像素乘权重掩码并按有效像素数归一化。

---

## 1️⃣ 热力图损失：CenterNet-style Focal（双中心）

### 作用

让`H_gap`与`H_piece`在真实中心处形成尖峰、背景更干净。

### 预测

```python
H_gap, H_piece ∈ (0,1)^[B,1,64,128]  # Sigmoid激活
```

### Focal基本式

对gap/piece各算一次再相加，推荐`α=1.5, β=4, t_pos=0.9`：

```python
L_heat(P,Y) = -1/N_pos * Σ_{i,j} {
    (1-P_ij)^α * log(P_ij)           if Y_ij ≥ t_pos  
    (1-Y_ij)^β * P_ij^α * log(1-P_ij)  otherwise
}
```

### 屏蔽归一化

替代`1/N_pos`与`(1-M_pad/4)`：

```python
L_heat = Σ_{i,j}(L_focal(P,Y)_ij * W_1/4,ij) / (Σ_{i,j}W_1/4,ij + ε)
```

---

## 2️⃣ 子像素偏移损失：Smooth-L1（正样本内）

### 作用

回归每个中心在其栅格内的连续偏移，实现亚像素定位。

### 预测张量

```python
O = [B,4,64,128]  # 四通道分别是 (du_g, dv_g, du_p, dv_p)
```

网络输出通过 `tanh × 0.5` 映射到 `[−0.5, 0.5]`

### 读取对应单点预测

按batch第b个样本：

- Gap: `d̂_x(g) = O[b,0,:,:]`, `d̂_y(g) = O[b,1,:,:]`
- Piece: `d̂_x(p) = O[b,2,:,:]`, `d̂_y(p) = O[b,3,:,:]`

### 标签

```python
(δu_g - 0.5, δv_g - 0.5)  # Gap偏移
(δu_p - 0.5, δv_p - 0.5)  # Piece偏移
```

### 损失计算

单样本Smooth-L1（Huber）：

```python
l_off(g) = SmoothL1(d̂_x(g) - d*_x(g)) + SmoothL1(d̂_y(g) - d*_y(g))
l_off(p) = SmoothL1(d̂_x(p) - d*_x(p)) + SmoothL1(d̂_y(p) - d*_y(p))
```

### 加权归一化

使用热图值作为权重：

```python
L_off = Σ(w_g * l_off(g)) / Σw_g + Σ(w_p * l_off(p)) / Σw_p
```

其中 w_g 和 w_p 分别是gap和piece的热图值

**注意**：不使用epsilon，而是通过assert检查确保权重和不为零

---

## 3️⃣ 假缺口抑制（Hard-Negative）：Margin Ranking  

### 作用

显式压制1–3个confusing_gap的次峰。

### 分数取值（任选其一，更稳）

1. **双线性采样**：

```python
s^+ = bilinear(H_gap, u_g, v_g)
s_k^- = bilinear(H_gap, u_k, v_k)
```

2. **邻域最大**：在各坐标3×3邻域取最大

且只在有效区采样（要求`W_1/4(v,u) > 0`）。

### 损失

```python
L_hn = 1/K * Σ_{k=1}^K max(0, m - s^+ + s_k^-)
```

其中`m = 0.2 ~ 0.3`

---

## 4️⃣ 角度损失（微角度，可选）

### 作用

建模0.5–1.8°的微旋，抑制"形似但角度偏"的假峰。

### 预测

```python
θ = [B,2,64,128]  # L2归一化后表示(sin̂θ, coŝθ)
```

### 标签

```python
(sinθ_g, cosθ_g)  # 来自合成器
```

### 仅在gap中心邻域监督

```python
M_ang = 1(Y_gap > 0.7)
```

### 屏蔽归一化

```python
L_ang = Σ[1 - (sin̂θ*sinθ_g + coŝθ*cosθ_g)] * M_ang * W_1/4 / Σ(M_ang*W_1/4)+ε
```

---

## 5️⃣ 滑块和背景缺口不同y轴损失

### 作用

确保模型预测的gap和piece中心位于不同的y轴位置，抑制"假同行"现象。

### 设定

热力图分辨率 $(H_s, W_s) = (64, 128)$，两张概率热力图为 $H_g, H_p \in (0,1)^{H_s \times W_s}$（与工程中 heatmap_gap/heatmap_piece 一致）。

### 0) 两张热力图 → 行分布（无需mask）

为了让行softmax的动态范围更合理，给行能量做轻微对数伸展（可选）：

**计算每行平均能量**：

$$\bar{H}_g(y) = \frac{1}{W_s} \sum_{x} H_g(y,x), \quad \bar{H}_p(y) = \frac{1}{W_s} \sum_{x} H_p(y,x)$$

**行能量处理**：

轻微对数伸展：
$$s_g(y) = \log(\bar{H}_g(y) + \varepsilon), \quad s_p(y) = \log(\bar{H}_p(y) + \varepsilon), \quad \varepsilon = 10^{-8}$$

**Softmax归一化为行分布**：

$$p_g(y) = \frac{\exp(s_g(y)/\tau)}{\sum_t \exp(s_g(t)/\tau)}, \quad p_p(y) = \frac{\exp(s_p(y)/\tau)}{\sum_t \exp(s_p(t)/\tau)}, \quad \tau = 1.0$$

### 1) 行分类损失（主对齐项）

正确的行分类损失，分别对gap和piece的目标行计算交叉熵：

$$\mathcal{L}_{\text{rowCE}} = -\log p_g(y^*) - \log p_p(y^*)$$

其中 $y^* = \lfloor Y^*/4 \rfloor$ 是真实y坐标对应的1/4分辨率行索引。

### 2) 分布约束 - 1D EMD

计算累积分布函数（CDF）：

$$F_g(y) = \sum_{t \leq y} p_g(t), \quad F_p(y) = \sum_{t \leq y} p_p(t)$$

EMD损失（Wasserstein-1距离）：

$$\mathcal{L}_{\text{EMD}} = \frac{1}{H_s} \sum_{y=0}^{H_s-1} |F_g(y) - F_p(y)|$$

### 3) 总Y轴损失

$$\mathcal{L}_y = \lambda_{\text{row}} \mathcal{L}_{\text{rowCE}} + \lambda_{\text{dist}} \mathcal{L}_{\text{EMD}}$$

**推荐权重**：
- $\lambda_{\text{row}} = 1.0$
- $\lambda_{\text{dist}} = 0.2 \sim 0.5$

## 6️⃣ Padding抑制损失（Padding BCE Loss）

### 作用

确保模型在padding区域（无效区域）不产生误检测。由于输入包含padding区域，模型可能在这些区域产生虚假响应，需要显式抑制。

### 设定

- **有效区域掩码**：$M \in \{0,1\}^{H_s \times W_s}$，其中 $M(y,x)=1$ 表示有效区域，$M(y,x)=0$ 表示padding区域
- **Padding区域掩码**：$\bar{M} = 1 - M$，选中所有padding像素
- **预测热力图**：$P_g, P_p \in (0,1)^{H_s \times W_s}$（已经过Sigmoid激活）

### 损失公式

对padding区域的每个像素，希望预测概率接近0，使用二元交叉熵（BCE）：

$$\mathcal{L}_{\text{pad-bce}} = \frac{1}{N_{\text{pad}}} \sum_{y,x} \left[ -\log(1-P_g(y,x)) - \log(1-P_p(y,x)) \right] \cdot \bar{M}(y,x)$$

其中：
- $P_g(y,x), P_p(y,x)$：模型输出的gap和piece热力图概率值
- $\bar{M}(y,x) = 1 - M(y,x)$：padding区域选择器（padding处为1，有效区域为0）
- $N_{\text{pad}} = \sum_{y,x} \bar{M}(y,x)$：padding像素总数，用于归一化

### 符号含义

- **$-\log(1-P)$**：当预测概率$P$接近0时损失小，接近1时损失大
- **$\bar{M}(y,x)$**：二值掩码，确保只在padding区域计算损失
- **$N_{\text{pad}}$归一化**：使损失值不依赖于padding区域大小

### 梯度计算

对于单个padding像素，BCE损失对预测值$P$的梯度为：

$$\frac{\partial \mathcal{L}_{\text{bce}}}{\partial P} = \frac{1}{1-P}$$

当$P \to 0$时，梯度约为1（稳定）；当$P \to 1$时，梯度趋于无穷（强烈惩罚）。

### 数值稳定性考虑

1. **避免log(0)**：对预测值进行截断
   ```python
   P_clamped = torch.clamp(P, min=eps, max=1-eps)
   loss = -log(1 - P_clamped)
   ```

2. **避免除零**：padding像素数检查
   ```python
   if N_pad < eps:
       return 0  # 没有padding区域时返回0
   ```

3. **梯度爆炸预防**：当$P$接近1时，使用梯度裁剪
   ```python
   loss = torch.clamp(loss, max=10.0)  # 防止单个像素贡献过大损失
   ```

### 与其他损失的关系

- **与Focal Loss的区别**：
  - Focal Loss处理有效区域内的正负样本平衡
  - Padding BCE专门处理无效区域的抑制
  - 两者作用域互补，不重叠

- **与Hard Negative的区别**：
  - Hard Negative抑制有效区域内的假缺口
  - Padding BCE抑制无效区域的所有响应
  - 前者是选择性抑制，后者是全面抑制

### 实现要点

```python
def padding_bce_loss(P_gap, P_piece, mask, eps=1e-8):
    # 计算padding掩码
    padding_mask = 1 - mask  # [B, 1, H, W]
    N_pad = padding_mask.sum()
    
    if N_pad < eps:
        return 0
    
    # 数值稳定性
    P_gap = torch.clamp(P_gap, min=eps, max=1-eps)
    P_piece = torch.clamp(P_piece, min=eps, max=1-eps)
    
    # BCE损失（目标为0）
    loss_gap = -torch.log(1 - P_gap) * padding_mask
    loss_piece = -torch.log(1 - P_piece) * padding_mask
    
    # 归一化
    L_pad_bce = (loss_gap.sum() + loss_piece.sum()) / N_pad
    
    return L_pad_bce
```

### 推荐权重

- $\lambda_{\text{pad}} = 0.1 \sim 0.3$：相对较小的权重，避免过度抑制影响有效区域学习

---

## 总损失组合

$$
L=w_h⋅(L_{heat}^{gap}+L_{heat}^{piece})+w_o⋅L_{off}+w_a⋅L_{ang}+w_{hn}·L_{hn}+\lambda_{row}\mathcal{L}_{\text{rowCE}}+\lambda_{dist}\mathcal{L}_{\text{EMD}}+\lambda_{pad}\mathcal{L}_{\text{pad-bce}}
$$

其中：

- `L_focal`：CenterNet风格热力图损失，α=1.5, β=4.0
- `L_offset`：子像素偏移损失，使用热图值加权
- `L_hard_negative`：Margin Ranking损失，抑制假缺口
- `L_angle`：角度损失，用于微旋转
- `L_rowCE`：行分类损失 Row-CE，确保滑块和缺口在不同y轴位置
- `L_EMD`：一维 Wasserstein-1 / EMD，抑制错位/双峰造成的"假同行"
- `L_pad_bce`：Padding BCE损失，抑制无效区域响应
- 权重：$w_h=1.0, w_o=1.0, w_a=0.5, w_{hn}=0.5, \lambda_{row}=1.0, \lambda_{dist}=0.2\sim0.5, \lambda_{pad}=0.1\sim0.3$

---

## 6️⃣ 附加解释

### CenterNet风格的热力图Focal Loss

**核心思想**：

- 正样本（中心点）：当预测值低时施加更大惩罚
- 负样本（背景）：根据到中心的距离加权，越远权重越小
- 通过`α`和`β`控制难易样本的关注度

**数学原理**：

- `(1-P)^α`项：当P接近0时损失更大，聚焦困难正样本
- `(1-Y)^β`项：距离权重，远离中心的负样本获得更小权重
- 避免简单负样本主导训练

### 子像素偏移损失（Smooth L1）

**为什么需要子像素**：

- 热力图分辨率为1/4，定位精度受限于栅格大小
- 子像素偏移补偿量化误差，实现亚像素精度
- 将离散坐标转换为连续坐标，提升定位精度至1像素级别

**Smooth L1优势**：

- 对异常值不敏感（相比L2）
- 梯度连续，训练稳定
- 在误差小时表现为L2（平滑），误差大时表现为L1（鲁棒）

### 形态学梯度

**边界提取原理**：

```python
Edge = Dilation(M) - Erosion(M)
```

- 膨胀操作扩大前景区域
- 腐蚀操作缩小前景区域
- 两者之差即为边界

**为什么先下采样**：

- 直接在高分辨率提取边界会导致细边消失
- 先下采样保证边界宽度与特征图尺度匹配

### Margin Ranking假缺口抑制

**动机**：

- 混淆缺口在视觉上与真实缺口相似
- 仅靠Focal Loss难以完全抑制假峰
- 需要显式的对比学习机制

**Margin设计**：

- `m=0.2~0.3`提供足够的决策边界
- 强制真实缺口分数比假缺口高至少m
- 类似于人脸识别中的Triplet Loss思想

---

## 7️⃣ 实现细节与最佳实践

### 调试可视化

```python
# TensorBoard监控
writer.add_scalar('Loss/heat', L_heat, step)
writer.add_scalar('Loss/offset', L_off, step)
writer.add_scalar('Loss/hard_negative', L_hn, step)
writer.add_scalar('Loss/row_ce', L_rowCE, step)
writer.add_scalar('Loss/emd', L_EMD, step)
writer.add_image('Heatmap/gap', H_gap[0], step)
writer.add_image('Heatmap/piece', H_piece[0], step)
```

---

## 8️⃣ 代码实现说明

### 数据处理流水线

#### 预处理流程

```python
# src/preprocessing/preprocessor.py
class TrainingPreprocessor:
    def preprocess(self, image, gap_center, slider_center, confusing_gaps, gap_angle):
        # 1. Letterbox变换
        image_letterboxed, transform_params = self.letterbox.apply(image)
        
        # 2. 生成padding mask
        padding_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 3. 组合4通道输入
        input_tensor = np.concatenate([
            image_channels,                    # [3, H, W]
            padding_mask[np.newaxis, :, :]    # [1, H, W]
        ], axis=0)  # [4, H, W]
        
        # 4. 生成热图和偏移标签
        # ...
        
        return {
            'input': input_tensor,  # 4通道
            'heatmaps': heatmaps,
            'offsets': offsets,
            'confusing_gaps': confusing_grids,
            'gap_angle': np.radians(gap_angle)
        }
```

#### 数据集生成

```python
# src/preprocessing/dataset_generator.py
class StreamingDatasetGenerator:
    def _flush_buffer_to_disk(self, split):
        # 保存数据文件
        np.save(image_path, self._buf_images[:batch_size])     # 4通道图像
        np.save(heatmap_path, self._buf_heatmaps[:batch_size])
        np.save(offset_path, self._buf_offsets[:batch_size])
        # 注意：权重掩码已集成在第4通道，不再单独保存
```

#### 数据加载

```python
# src/training/npy_data_loader.py
class NPYBatchDataset:
    def __getitem__(self, idx):
        # 加载4通道图像
        images = np.load(image_path)  # [B, 4, 256, 512]
        
        # 从第4通道提取权重掩码
        padding_mask = images[:, 3, :, :]  # [B, 256, 512]
        
        # 下采样到1/4分辨率
        weights = downsample_by_avgpool(padding_mask, factor=4)  # [B, 64, 128]
        
        return {
            'image': images,  # 包含padding mask的4通道
            'weight_gap': weights,
            'weight_slider': weights,
            # ...
        }
```
5️⃣ 滑块和背景缺口不同y轴损失
### 损失函数模块结构

```text
src/models/loss_calculation/
├── focal_loss.py       # CenterNet风格热力图损失
├── offset_loss.py      # 子像素偏移损失
├── hard_negative_loss.py  # 假缺口抑制损失
├── angle_loss.py       # 角度损失
├── y_axis_loss.py      # Y轴分布损失（Row-CE + EMD）
├── total_loss.py       # 总损失组合
└── config_loader.py    # 配置加载器
```

### 配置文件

```yaml
# config/loss.yaml
focal_loss:
  alpha: 1.5          # 正样本聚焦参数
  beta: 4.0           # 负样本距离加权
  pos_threshold: 0.8  # 正样本阈值
  eps: 1.0e-8        # 数值稳定性

offset_loss:
  beta: 1.0           # Smooth L1平滑参数

hard_negative_loss:
  margin: 0.2         # 真假缺口最小得分差
  score_type: bilinear  # 采样方式
  neighborhood_size: 3  # 邻域大小

angle_loss:
  enabled: true       # 是否启用角度损失
  threshold: 0.7      # 热图阈值

y_axis_loss:
  enabled: true       # 是否启用Y轴损失
  use_log_stretch: true  # 是否使用对数伸展
  temperature: 1.0    # Softmax温度参数
  eps: 1.0e-8        # 数值稳定性

total_loss:
  weights:
    heatmap: 1.0
    offset: 1.0
    hard_negative: 0.5
    angle: 0.5
    row_ce: 1.0       # Row-CE权重
    emd: 0.3          # EMD权重（0.2~0.5）
```

### 关键实现细节

1. **Focal Loss归一化**：使用N_pos（正样本数）进行归一化，避免样本不平衡
2. **Offset Loss加权**：直接使用热图值作为权重w_g，无需额外阈值
3. **Hard Negative采样**：支持双线性插值和邻域最大值两种方式
4. **工厂函数**：所有损失函数都提供工厂函数，不使用默认参数

---

## 9️⃣ 数据流与损失计算完整流程

### 训练数据流

```mermaid
graph TD
    A[原始图像 320x160] --> B[Letterbox变换]
    B --> C[图像 512x256 + Padding Mask]
    C --> D[4通道输入张量]
    D --> E[模型前向传播]
    E --> F[预测输出]
    
    G[标签生成] --> H[热图 64x128]
    G --> I[偏移 64x128]
    G --> J[混淆缺口坐标]
    G --> K[旋转角度]
    
    F --> L[损失计算]
    H --> L
    I --> L
    J --> L
    K --> L
    
    M[Padding Mask] --> N[下采样 4x]
    N --> O[权重掩码 64x128]
    O --> L
```

### 损失计算详细步骤

#### Step 1: 准备阶段

```python
# 从批次数据中提取
images = batch['image']           # [B, 4, 256, 512]
heatmap_gap_gt = batch['heatmap_gap']     # [B, 64, 128]
heatmap_slider_gt = batch['heatmap_slider']  # [B, 64, 128]
offset_gap_gt = batch['offset_gap']       # [B, 2, 64, 128]
offset_slider_gt = batch['offset_slider']    # [B, 2, 64, 128]
weight_mask = batch['weight_gap']         # [B, 64, 128]
confusing_gaps = batch['confusing_gaps']     # List[List[Tuple]]
gap_angles = batch['gap_angles']          # [B]
```

#### Step 2: 模型预测

```python
# 模型输出
outputs = model(images)  # 4通道输入
heatmap_gap_pred = outputs['heatmap_gap']     # [B, 1, 64, 128]
heatmap_slider_pred = outputs['heatmap_slider']  # [B, 1, 64, 128]
offset_gap_pred = outputs['offset_gap']       # [B, 2, 64, 128]
offset_slider_pred = outputs['offset_slider']    # [B, 2, 64, 128]
angle_pred = outputs.get('angle')          # [B, 2, 64, 128] (可选)
```

#### Step 3: 损失组件计算

```python
# 1. Focal Loss（热图损失）
L_focal_gap = focal_loss(heatmap_gap_pred, heatmap_gap_gt, weight_mask)
L_focal_slider = focal_loss(heatmap_slider_pred, heatmap_slider_gt, weight_mask)

# 2. Offset Loss（偏移损失）
L_offset_gap = offset_loss(offset_gap_pred, offset_gap_gt, heatmap_gap_gt, weight_mask)
L_offset_slider = offset_loss(offset_slider_pred, offset_slider_gt, heatmap_slider_gt, weight_mask)

# 3. Hard Negative Loss（假缺口抑制）
if confusing_gaps:
    L_hard_negative = hard_negative_loss(heatmap_gap_pred, gap_coords_gt, confusing_gaps, weight_mask)
else:
    L_hard_negative = 0

# 4. Angle Loss（角度损失）
if angle_pred is not None and gap_angles is not None:
    L_angle = angle_loss(angle_pred, gap_angles, heatmap_gap_gt, weight_mask)
else:
    L_angle = 0

# 5. Y-Axis Loss（Y轴分布损失）
L_rowCE, L_EMD = y_axis_loss(heatmap_gap_pred, heatmap_slider_pred, gap_coords_gt, slider_coords_gt)
```

#### Step 4: 总损失组合

```python
L_total = (L_focal_gap + L_focal_slider) * weights['heatmap'] + \
          (L_offset_gap + L_offset_slider) * weights['offset'] + \
          L_hard_negative * weights['hard_negative'] + \
          L_angle * weights['angle'] + \
          L_rowCE * weights['row_ce'] + \
          L_EMD * weights['emd']
```

### 关键数据转换

| 阶段 | 数据格式 | 维度 | 说明 |
|------|---------|------|------|
| 原始图像 | uint8 | [320, 160, 3] | RGB图像 |
| Letterbox后 | float32 | [256, 512, 3] | 归一化+填充 |
| 4通道输入 | float32 | [4, 256, 512] | RGB+Mask |
| 特征图 | float32 | [C, 64, 128] | 1/4分辨率 |
| 热图预测 | float32 | [1, 64, 128] | Sigmoid激活 |
| 偏移预测 | float32 | [2, 64, 128] | Tanh激活 |
| 权重掩码 | float32 | [64, 128] | 从第4通道下采样 |
