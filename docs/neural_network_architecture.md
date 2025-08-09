# 滑块验证码识别神经网络架构设计文档

## 总体结构与数据流

### 输入（3路）

- **拼图**：$I_p \in \mathbb{R}^{B \times 3 \times H_p \times W_p}$
- **背景缺口图**：$I_b \in \mathbb{R}^{B \times 3 \times 160 \times 320}$
- **合成图（拼图+背景）**：$I_c \in \mathbb{R}^{B \times 3 \times 160 \times 320}$

### 处理流程

主干（共享）→ FPN（取 P2, stride=4）→ 方向分解（K通道 Softmax）

得到：
- $P2_p, P2_b, P2_c \in \mathbb{R}^{B \times 128 \times (\cdot) \times (\cdot)}$
- $O_p, O_b, O_c \in \mathbb{R}^{B \times K \times (\cdot) \times (\cdot)}$

### 阶段A：在合成图定位拼图（取 $y^*$）

1. **ORI保向2D相关**：$O_p$ vs $O_c$（方向同向 $k \leftrightarrow k$）→ 2D热力图 $S_{comp}(u,v)$
2. 取 $(u_0, v_0)$ 与小窗亚像素精修（原图 ZNCC + Soft-Argmax）→ $(x_p^*, y_p^*)$
3. 若背景与合成几何一致：$y_b^* = y_p^*$；否则做线性映射 $y_b^* = a \cdot y_p^* + b$（可学习/校准）

### 阶段B：在背景图沿 $y_b^*$ 做横向匹配，定位缺口

1. **ORI互补1D相关**：$O_p$ vs $O_b$（方向互补 $k \leftrightarrow k+K/2$）→ 1D曲线 $S_{bg}(s)$（横向位移）
2. Top-K候选 $\{s_i\}$ → 位姿回归头（$\Delta\theta_i, \log s_i, \text{OriScore}_i$）
3. 原图亚像素（仿射预配准 + ZNCC + Soft-Argmax）→ $(\delta x_i, \delta y_i), \text{CorrPeak}_i$
4. 方向加权Chamfer（$\text{DCD}_i$）与峰尖锐度重排 → 取 $i^*$ 输出 $(x_g^*, y_g^*)$

## 1. 共享骨干 + FPN（在P2上做相关）

骨干：ResNet-18（两支/三支共享权重），把maxpool替换为BlurPool(Anti-Aliasing)；FPN只取P2(stride=4)作为相关尺度。

| 模块 | 卷积/步幅/通道 | 合成/背景：输入→输出 | 拼图：输入→输出 |
|------|--------------|---------------------|-----------------|
| stem.conv1 | 7×7, s=2, out=64 | [B,3,160,320]→[B,64,80,160] | [B,3,H_p,W_p]→[B,64,⌈H_p/2⌉,⌈W_p/2⌉] |
| stem.blur | BlurPool s=2 | [B,64,80,160]→[B,64,40,80] | →[B,64,⌈H_p/4⌉,⌈W_p/4⌉] |
| layer1 | Basic×2, s=1, out=64 | [B,64,40,80]→[B,64,40,80] | →[B,64,h_f,w_f] |
| layer2 | Basic×2, s=2, out=128 | [B,64,40,80]→[B,128,20,40] | →[B,128,⌈h_f/2⌉,⌈w_f/2⌉] |
| FPN-P2 | 自顶向下融合到C2 + 3×3 conv → 128 | [B,128,40,80] | [B,128,h_f,w_f] |

后续相关均在P2进行（stride=4）；$h_f = w_f = \lceil H_p/4 \rceil \in [10,15]$。

## 2. 方向分解头（Orientation Head, ORI）

目的：把特征按方向K个通道分解并经Softmax归一，便于做"同向"与"互补方向"相关。

| 层 | 操作 | 合成/背景：输入→输出 | 拼图：输入→输出 |
|----|------|-------------------|----------------|
| ori.conv | 3×3, 128→128, BN+ReLU | [B,128,40,80]→[B,128,40,80] | [B,128,h_f,w_f]→[B,128,h_f,w_f] |
| ori.head | 1×1, 128→K + Softmax(通道维) | [B,K,40,80] 记 $O_c, O_b$ | [B,K,h_f,w_f] 记 $O_p$ |

Softmax使每位置K通道为概率分布；K取24（15°/bin）可以；若旋转假缺口多为 10-15°，上调到 **K=32（11.25°/bin）** 会更稳。

**温度参数建议**：ORI的通道Softmax建议加入温度 $\tau \in [0.7, 1.0]$（或轻微 label smoothing），避免早期过度锐化导致梯度稀疏。

## 3. 阶段A：在合成图定位拼图（取 $y^*$）

### A.1 方向同向2D相关（在P2）

对每个平移 $(u,v)$：

$$S_{comp}(u,v) = \sum_{y=0}^{h_f-1} \sum_{x=0}^{w_f-1} \sum_{k=0}^{K-1} O_p(k,y,x) \cdot O_c(k, y+v, x+u)$$

输出形状：
- $L_x^{(c)} = 80 - w_f + 1 \in [66,71]$
- $L_y^{(c)} = 40 - h_f + 1 \in [26,31]$
- $S_{comp} \in \mathbb{R}^{B \times L_y^{(c)} \times L_x^{(c)}}$

候选：对 $S_{comp}$ 做Softmax得 $\hat{P}_{uv}$，取argmax $(u_0, v_0)$。

### A.2 合成图亚像素精修（原图）

- **ROI**：以 $(x_0, y_0) = (4u_0, 4v_0)$ 为中心，从合成图裁 $R_c \in \mathbb{R}^{B \times 1 \times (H_p+2r_y) \times (W_p+2r_x)}$，拼图灰度 $R_p \in \mathbb{R}^{B \times 1 \times H_p \times W_p}$。
- **ZNCC窗口**：$(dx, dy) \in [-r_x,r_x] \times [-r_y,r_y]$，搜索离散格为 $(2r_y+1) \times (2r_x+1)$
- **合成图亚像素搜索**：$r_x = r_y = 4 \Rightarrow 9 \times 9$ 格点
- **Soft-Argmax**（温度 $T=0.04$）→ $(\delta x_p, \delta y_p)$
- **输出**：$(x_p^*, y_p^*) = (x_0 + \delta x_p, y_0 + \delta y_p)$

若几何一致：令 $y_b^* = y_p^*$。  
若不一致：加入一个Y-Align线性层（可学习或标定）：$y_b = a \cdot y_p + b$，其中 $a,b$ 由少量标定样本估计，或用合成/背景垂直投影相关自动回归。

## 4. 阶段B：在背景图沿 $y_b^*$ 做横向匹配

### B.1 垂直裁剪对齐到拼图高度（P2）

在 $O_b$ 与 $P2_b$ 上，围绕 $y_b^*/4$（取整）裁出高度 $h_f$ 的带（允许±1行冗余，取分数较优者）。

记 $\tilde{O}_b \in \mathbb{R}^{B \times K \times h_f \times 80}$，$\tilde{P2}_b \in \mathbb{R}^{B \times 128 \times h_f \times 80}$。

### B.2 方向互补1D相关（在P2）

只沿x平移，方向做180°互补匹配：

$$S_{bg}(t) = \sum_{y=0}^{h_f-1} \sum_{x=0}^{w_f-1} \sum_{k=0}^{K-1} O_p(k,y,x) \cdot \tilde{O}_b((k+\frac{K}{2}) \mod K, y, x+t)$$

输出形状：
- $L_x^{(b)} = 80 - w_f + 1 \in [66,71]$
- $S_{bg} \in \mathbb{R}^{B \times L_x^{(b)}}$

取Top-K（如10）候选位移 $\{t_i\}$（横向位移候选，P2上的格点索引），并计算每个候选的峰尖锐度（二阶差分负值）$\text{Sharpness}_i$。

### B.3 候选位姿回归头（Angle-Scale Head, ASH）

#### 候选特征拼接（在P2）

- **背景候选补丁**：$\tilde{P2}_{b,i} = \tilde{P2}_b[:,:,:,t_i:t_i+w_f] \in [B,128,h_f,w_f]$
- **拼图补丁**：$P2_p \in [B,128,h_f,w_f]$
- **方向补丁**：$\tilde{O}_{b,i} \in [B,K,h_f,w_f], O_p \in [B,K,h_f,w_f]$
- **1×1降维**：128→32，得 $B'_i, P'_p \in [B,32,h_f,w_f]$
- **拼接张量**：$T_i = [O_p, \tilde{O}_{b,i}, P'_p, B'_i] \in [B,(2K+64),h_f,w_f]$

#### 回归/评分分支（对每候选复用，共享权重）

| 层 | 操作 | 输入→输出 |
|----|------|----------|
| as.conv1 | 3×3, (2K+64)→128, BN+ReLU | [B,2K+64,h_f,w_f]→[B,128,h_f,w_f] |
| as.conv2 | 3×3, 128→128, BN+ReLU | [B,128,h_f,w_f]→[B,128,h_f,w_f] |
| GAP | 全局平均池化 | [B,128,h_f,w_f]→[B,128] |
| fc1 | FC 128→64, ReLU | [B,128]→[B,64] |
| fc_pose | FC 64→3 | [B,3] → $(\cos\Delta\theta_i, \sin\Delta\theta_i, \log \sigma_i)$ |
| fc_ori | FC 64→1 + Sigmoid | [B,1] → $\text{OriScore}_i \in (0,1)$ |

### B.4 原图亚像素精修头（FCH）

#### ROI与仿射预配准

- 以像素 $x_i = 4t_i, y_b^*$ 为中心（ROI中心），从背景原图取 $R_{b,i} \in \mathbb{R}^{B \times 1 \times (H_p+8) \times (W_p+16)}$；拼图灰度 $R_p \in \mathbb{R}^{B \times 1 \times H_p \times W_p}$。
- 用 $(\Delta\theta_i, \sigma_i)$ 对 $R_{b,i}$ 做仿射grid_sample预对齐 → $\tilde{R}_{b,i}$。

#### ZNCC + Soft-Argmax（亚像素）

- 窗口 $(dy,dx) \in [-r_y,r_y] \times [-r_x,r_x]$，搜索离散格为 $(2r_y+1) \times (2r_x+1)$
- **背景亚像素搜索**：$r_x = 4, r_y = 2 \Rightarrow 5 \times 9$ 格点，相关图 $C_i \in [B,5,9]$
- Soft-Argmax（温度 $T=0.04$）→ $(\delta x_i, \delta y_i)$；并取 $\text{CorrPeak}_i = \max C_i$。

### B.5 定向-Chamfer与综合打分

由Sobel得边界与方向 $\theta$；把拼图边界按候选位姿映射到背景，计算：

$$d(p) = \min_{q \in \partial G}(\|p-q\|_2 + \lambda_\theta[1-\cos(\theta_p+\pi-\theta_q)]), \quad \text{DCD}_i = \frac{1}{|B|}\sum_{p \in B} d(p)$$

综合分数（用于重排序/是否接受）：

$$\text{Score}_i = \alpha \cdot \text{CorrPeak}_i + \beta \cdot \text{OriScore}_i - \gamma \cdot \text{DCD}_i + \delta \cdot \text{Sharpness}_i$$

选 $i^* = \arg\max_i \text{Score}_i$（推理时用的最优候选），输出最终缺口坐标：

$$x_g^* = 4t_{i^*} + \delta x_{i^*}, \quad y_g^* = y_b^* + \delta y_{i^*}$$

## 5. 损失函数

设合成图中拼图真值平移 $(u^*, v^*) = (\lfloor x_p^*/4 \rfloor, \lfloor y_p^*/4 \rfloor)$，残差 $\epsilon_p^x = x_p^* - 4u^*$，$\epsilon_p^y = y_p^* - 4v^*$。
背景侧横向索引 $t^* = \lfloor x_g^*/4 \rfloor$，残差 $\epsilon_g^x = x_g^* - 4t^*$，允许行残差 $\epsilon_g^y \in [-2,2]$。

**训练时oracle候选选择**：$i^\dagger = \arg\min_i |t_i - t^*|$（或 $|t_i - t^*| \leq 1$ 的最近者）。注意：损失函数 $L_\theta, L_\sigma, L_{subpx-bg}, L_{DCD}$ 等只应作用在包含真值的oracle候选 $i^\dagger$ 上，而不是推理时用的 $i^* = \arg\max \text{Score}_i$。

### A. 合成图定位（阶段A）

#### 2D相关热力图监督（Focal-CE）

在 $S_{comp} \in \mathbb{R}^{L_y^{(c)} \times L_x^{(c)}}$ 上放置二维高斯软标签 $Y_{comp}(u,v)$（$\sigma_x = \sigma_y = 1$），对Softmax概率 $\hat{P}_{uv}$：

$$L_{comp} = -\sum_{u,v} \alpha(1-\hat{P}_{uv})^\gamma Y_{comp}(u,v) \log \hat{P}_{uv}, \quad \alpha = 0.25, \gamma = 2$$

#### 亚像素回归（合成图）

$$L_{subpx-comp} = \text{Huber}(\delta x_p - \epsilon_p^x) + \text{Huber}(\delta y_p - \epsilon_p^y)$$

### B. 背景缺口定位（阶段B）

#### 1D横向相关监督（Focal-CE）

在 $S_{bg} \in \mathbb{R}^{L_x^{(b)}}$ 上放1D高斯标签 $y_{bg}(t)$（$\sigma=1$，相当于stride=4空间的1格），Softmax概率 $\hat{p}_t$：

$$L_{bg} = -\sum_t \alpha(1-\hat{p}_t)^\gamma y_{bg}(t) \log \hat{p}_t$$

注：若收敛慢，可先用 $\sigma=1.5$ 预热3-5 epoch，再切回 $\sigma=1$。

#### 位姿回归（角度+尺度，仅对oracle候选 $i^\dagger$）

$$L_\theta = \text{Huber}((\cos\Delta\theta_{i^\dagger}, \sin\Delta\theta_{i^\dagger}), (\cos\Delta\theta^*, \sin\Delta\theta^*))$$
$$L_\sigma = \text{Huber}(\log \sigma_{i^\dagger}, \log \sigma^*)$$

正样本 $|\Delta\theta^*| \leq 2°$，$\sigma^* \approx 1$。

#### 方向一致性间隔（区分旋转假缺口）

对每批样本内选择1-2个"同y轴、旋转10~30°"hard negative：优先选择"同一 $y$ + 旋转 10-30° + 位移相近（$|t_i - t^*| \leq 2$）"的候选，这类负样本最具迷惑性，学习收益最大。

$$L_{ori-margin} = \max(0, m - (\text{OriScore}_{pos} - \text{OriScore}_{neg})), \quad m = 0.2$$

#### 亚像素回归（背景，仅对oracle候选 $i^\dagger$）

$$L_{subpx-bg} = \text{Huber}(\delta x_{i^\dagger} - \epsilon_g^x) + \lambda_y \cdot \text{Huber}(\delta y_{i^\dagger} - \epsilon_g^y), \quad \lambda_y = 0.5$$

#### 定向-Chamfer几何一致性（仅对oracle候选 $i^\dagger$）

$$L_{DCD} = \text{DCD}_{i^\dagger}$$

实现上对背景边缘概率图做高斯金字塔平滑，用软最小值近似min保持可微。

#### 峰尖锐度正则（可选）

$$L_{sharp} = -\Delta^2 S_{bg}(t^*)$$

实现时可用一维二阶差分 $-\Delta^2 S$ 或拉普拉斯近似，数值更稳。

### C. 总损失

$$L = \lambda_1 L_{comp} + \lambda_2 L_{subpx-comp} + \lambda_3 L_{bg} + \lambda_4 L_\theta + \lambda_5 L_\sigma + \lambda_6 L_{ori-margin} + \lambda_7 L_{subpx-bg} + \lambda_8 L_{DCD} + \lambda_9 L_{sharp}$$

默认权重：$(\lambda_1, ..., \lambda_9) = (1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.3, 0.1)$。

根据验证集上PCK@2px与平均欧氏误差调参；若几何一致性已很强，可降低 $\lambda_8$。

## 6. 尺寸速查（三种拼图尺寸示例）

| 拼图原尺寸 | $h_f = w_f$ | $S_{comp}$ 尺寸 $L_y^{(c)} \times L_x^{(c)}$ | $S_{bg}$ 长度 $L_x^{(b)}$ |
|-----------|------------|-------------------------------------|------------------------|
| 40×40 | 10 | 31 × 71 | 71 |
| 50×50 | 13 | 28 × 68 | 68 |
| 60×60 | 15 | 26 × 66 | 66 |

## 7. 关键实现要点总结

### 符号澄清
- $t_i$：横向位移候选（P2上的格点索引）
- $\sigma_i$：尺度回归值（ASH头输出，训练用 $\log \sigma_i$）
- $i^\dagger$：训练时的oracle候选（包含真值）
- $i^*$：推理时的最优候选（最高评分）

### 小优化建议
- **Y对齐**：若合成图与背景图几何并非完全一致，实现上建议先用垂直投影视差最小化粗估 $a,b$，再在少量标定样本上做线性回归微调
- **掩码化ZNCC**：若拼图曾做过填充，务必对0填充区做掩码，避免均值/方差被拉偏
- **一致性**：ROIAlign/grid_sample的align_corners、坐标原点定义（像素中心 vs 左上角）在训练与推理保持一致，否则会出现系统性 1-2 px 偏差