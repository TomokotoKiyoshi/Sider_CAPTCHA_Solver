# Lite-HRNet-18 网络架构设计说明书 - 损失函数

## 0️⃣ 记号与标签生成（统一规范）

### 基础定义
- **原图尺寸**：`H×W = 256×512`
- **主特征图**：`Hf ∈ [B,128,64,128]`（1/4 分辨率）
- **真实中心（原图像素）**：
  - Gap中心：`(x_g, y_g)`
  - Piece中心：`(x_p, y_p)`

### 坐标映射
从原图像素坐标映射到1/4分辨率栅格坐标：
```
(u_g, v_g) = (x_g/4, y_g/4)  # gap栅格坐标
(u_p, v_p) = (x_p/4, y_p/4)  # piece栅格坐标
```

### 混淆缺口坐标
来自数据生成器的假缺口位置：
```
{(u_k, v_k)}_{k=1}^K, K=1..3  # 1-3个混淆缺口
```

### 高斯热图标签生成
使用栅格单位的高斯分布生成热图标签：
```
Y(i,j) = exp(−((i−v)² + (j−u)²)/(2σ²))
```
推荐参数：`σ=1.5`（栅格单位）

### 子像素偏移标签
用于精确定位的亚像素偏移：
```
δu = u − ⌊u⌋ ∈ [0,1)  # 水平偏移
δv = v − ⌊v⌋ ∈ [0,1)  # 垂直偏移
```
训练时中心化到 `[−0.5, 0.5]`：
```
δu_centered = δu − 0.5
δv_centered = δv − 0.5
```

### 边界标签（可选）
正确的边界提取顺序（避免细边消失）：
```python
# 1. 先下采样掩码到1/4分辨率
M_1/4 = (AvgPool_{k=4,s=4}(M) > 0.5).float()

# 2. 在1/4分辨率上提取边界
Edge_1/4 = MorphGrad(M_1/4, kernel_size=3)
# 等价于：Edge_1/4 = dilate(M_1/4) − erode(M_1/4)
# 或：Edge_1/4 = M_1/4 ⊕ erode(M_1/4)
```

### Padding Mask与权重生成
#### Padding Mask定义
- `M_pad ∈ {0,1}`：1表示padding区域，0表示有效区域

#### 权重计算方法
**软屏蔽（推荐）**：
```python
P_1/4 = AvgPool_{k=4,s=4}(M_pad)  # 软过渡
W_1/4 = 1 − P_1/4                  # 有效权重
```

#### 权重特性
- 纯padding栅格：`P_1/4 = 1 ⇒ W_1/4 = 0`（完全屏蔽）
- 纯有效栅格：`P_1/4 = 0 ⇒ W_1/4 = 1`（完全有效）
- 边界栅格：`0 < P_1/4 < 1 ⇒ 0 < W_1/4 < 1`（部分权重）

> **核心原则**：所有1/4分辨率上的像素级损失都逐像素乘以`W_1/4`，实现精确屏蔽。

---

## 1️⃣ 热力图损失：CenterNet-style Focal Loss

### 目标
让`H_gap`和`H_piece`在真实中心形成尖锐峰值，抑制背景噪声。

### 网络输出
```
H_gap, H_piece ∈ (0,1)^[B,1,64,128]  # 通过Sigmoid激活
```

### Focal Loss定义
对每个热图分别计算，推荐超参数：`α=1.5, β=4, t_pos=0.9`

```python
def focal_loss(P, Y, alpha=1.5, beta=4, t_pos=0.9):
    pos_mask = (Y >= t_pos)
    neg_mask = ~pos_mask
    
    # 正样本损失：聚焦困难正样本
    pos_loss = -torch.log(P + 1e-7) * (1 - P)**alpha * pos_mask
    
    # 负样本损失：距离加权的背景抑制
    neg_loss = -torch.log(1 - P + 1e-7) * P**alpha * (1 - Y)**beta * neg_mask
    
    return pos_loss + neg_loss
```

### 逐像素屏蔽与归一化
```python
# 计算focal loss
L_focal_gap = focal_loss(H_gap, Y_gap)
L_focal_piece = focal_loss(H_piece, Y_piece)

# 逐像素屏蔽并归一化
L_heat_gap = torch.sum(L_focal_gap * W_1/4) / (torch.sum(W_1/4) + 1e-7)
L_heat_piece = torch.sum(L_focal_piece * W_1/4) / (torch.sum(W_1/4) + 1e-7)

# 总热图损失
L_heat = L_heat_gap + L_heat_piece
```

---

## 2️⃣ 子像素偏移损失：Smooth-L1 Loss

### 目标
精确回归中心点在栅格内的连续偏移，实现亚像素精度。

### 网络输出
```
O = [B, 4, 64, 128]  # 4通道：(du_g, dv_g, du_p, dv_p)
```
使用`tanh × 0.5`限制输出范围到`[−0.5, 0.5]`

### 标签准备
```python
# Gap偏移标签（中心化）
target_du_g = δu_g − 0.5
target_dv_g = δv_g − 0.5

# Piece偏移标签（中心化）
target_du_p = δu_p − 0.5  
target_dv_p = δv_p − 0.5
```

### 正样本掩码
只在中心附近监督偏移：
```python
M_pos_gap = (Y_gap > 0.7)    # Gap正样本区域
M_pos_piece = (Y_piece > 0.7) # Piece正样本区域
```

### 逐像素屏蔽的Smooth-L1损失
```python
def masked_smooth_l1(pred, target, pos_mask, weight_mask):
    # Smooth-L1距离
    diff = torch.abs(pred - target)
    smooth_l1 = torch.where(diff < 1, 0.5 * diff**2, diff - 0.5)
    
    # 双重掩码：正样本掩码 × padding权重
    mask = pos_mask * weight_mask
    
    # 归一化损失
    loss = torch.sum(smooth_l1 * mask) / (torch.sum(mask) + 1e-7)
    return loss

# Gap偏移损失
L_off_gap = masked_smooth_l1(
    torch.stack([du_g, dv_g], dim=-1),
    torch.stack([target_du_g, target_dv_g], dim=-1),
    M_pos_gap, W_1/4
)

# Piece偏移损失
L_off_piece = masked_smooth_l1(
    torch.stack([du_p, dv_p], dim=-1),
    torch.stack([target_du_p, target_dv_p], dim=-1),
    M_pos_piece, W_1/4
)

# 总偏移损失
L_off = L_off_gap + L_off_piece
```

---

## 3️⃣ 假缺口抑制损失：Hard-Negative Mining

### 目标
显式抑制混淆缺口位置的响应，防止假阳性峰值。

### 分数提取策略

#### 策略1：双线性插值（推荐）
```python
# 真实缺口分数（连续坐标）
s_pos = F.grid_sample(H_gap, coords_to_grid(u_g, v_g))

# 混淆缺口分数
s_neg_list = []
for u_k, v_k in confusing_gaps:
    s_k = F.grid_sample(H_gap, coords_to_grid(u_k, v_k))
    s_neg_list.append(s_k)
```

#### 有效区域筛选
```python
# 只保留落在有效区域的负样本
valid_neg_scores = []
for k, (u_k, v_k) in enumerate(confusing_gaps):
    if W_1/4[int(v_k), int(u_k)] > 0:  # 在有效区域
        valid_neg_scores.append(s_neg_list[k])
```

### 损失函数实现

#### Margin Ranking Loss
```python
def margin_ranking_loss(s_pos, s_neg_list, margin=0.25):
    losses = []
    for s_neg in s_neg_list:
        loss = torch.relu(margin - s_pos + s_neg)
        losses.append(loss)
    return torch.mean(torch.stack(losses))

L_hn = margin_ranking_loss(s_pos, valid_neg_scores, margin=0.25)
```

---

## 4️⃣ 角度损失：Cosine Loss

### 目标
建模微小角度旋转（0.5°-1.8°），提升旋转鲁棒性。

### 网络输出
```
θ = [B, 2, 64, 128]  # 2通道：(sinθ, cosθ)
```
输出经过L2归一化：`θ_norm = θ / ||θ||_2`

### 标签
```python
# 真实角度标签
target_sin = torch.sin(theta_g)
target_cos = torch.cos(theta_g)
```

### 监督区域
只在gap中心附近监督角度：
```python
M_ang = (Y_gap > 0.7)  # 角度监督掩码
```

### 逐像素屏蔽的Cosine损失
```python
def masked_cosine_loss(pred_sin, pred_cos, target_sin, target_cos, mask, weight):
    # Cosine相似度损失
    cosine_sim = pred_sin * target_sin + pred_cos * target_cos
    loss = 1 - cosine_sim  # 范围[0, 2]
    
    # 双重掩码
    full_mask = mask * weight
    
    # 归一化
    return torch.sum(loss * full_mask) / (torch.sum(full_mask) + 1e-7)

L_ang = masked_cosine_loss(
    sin_pred, cos_pred,
    target_sin, target_cos,
    M_ang, W_1/4
)
```

---

## 5️⃣ 总损失函数

### 损失组合
```python
L_total = w_h * (L_heat_gap + L_heat_piece)  # 热图损失
        + w_o * L_off                        # 偏移损失
        + w_a * L_ang                        # 角度损失
        + w_hn * L_hn                        # 假缺口抑制
```

### 权重配置
```python
loss_weights = {
    'w_h': 1.0,   # 热图损失权重（主要监督）
    'w_o': 1.0,   # 偏移损失权重（精度关键）
    'w_a': 0.2,   # 角度损失权重（辅助信息）
    'w_hn': 1.0,  # 假缺口抑制权重（抗混淆关键）
    'w_e': 0.2    # 边界损失权重（预留，未使用）
}
```

---

## 6️⃣ 实现要点与优化建议

### 数值稳定性
1. **梯度裁剪**：防止梯度爆炸
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
   ```

2. **损失缩放**：平衡不同损失项的量级
   ```python
   # 自适应损失缩放
   with torch.cuda.amp.autocast():
       loss = compute_total_loss()
   ```

3. **防止除零**：所有归一化分母添加`ε=1e-7`

### 进阶训练策略（后期实现）

1. **困难样本挖掘**：
   - 保存损失最高的10%样本
   - 每5个epoch重新训练这些困难样本

2. **标签平滑**：
   - 对热图标签应用轻微平滑：`Y_smooth = Y * 0.95 + 0.025`
   - 减少过拟合，提升泛化

### 性能优化
1. **损失计算并行化**：
   ```python
   # 并行计算各损失项
   with torch.no_grad():
       masks = prepare_all_masks()  # 预计算所有掩码
   
   losses = torch.nn.parallel.parallel_apply(
       [compute_heat_loss, compute_offset_loss, compute_hn_loss],
       inputs
   )
   ```

2. **混合精度训练**：
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       loss = model(x)
   scaler.scale(loss).backward()
   ```

3. **梯度累积**：
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = compute_loss(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

---

## 7️⃣ 调试与可视化

- `tensorborad` 可视化