# Step0 - 数据预处理与坐标映射指导文档

## 1. 概述

本文档描述 Lite-HRNet-18 网络的预处理流程，采用**等比缩放+居中填充（letterbox）**方法，确保几何不变性，实现<2px的精确定位。

### 1.1 核心原则
- **几何保持**：使用letterbox避免拉伸，保持角度和形状不变
- **精确映射**：完整的坐标正反变换体系
- **统一规范**：左上角原点，像素单位

## 2. 尺寸处理：Letterbox方法

### 2.1 等比缩放计算
```python
def calculate_letterbox_params(W_in, H_in, W_t=512, H_t=256):
    """
    计算letterbox参数
    
    Args:
        W_in, H_in: 原始图像尺寸
        W_t, H_t: 目标尺寸 (512×256)
    
    Returns:
        s: 缩放比例
        W', H': 缩放后尺寸
        pad_left, pad_right, pad_top, pad_bottom: padding值
    """
    # 计算等比缩放因子
    s = min(W_t / W_in, H_t / H_in)
    
    # 缩放后尺寸
    W_prime = round(W_in * s)
    H_prime = round(H_in * s)
    
    # 计算padding
    pad_x = W_t - W_prime
    pad_y = H_t - H_prime
    
    # 居中padding
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    
    return s, W_prime, H_prime, pad_left, pad_right, pad_top, pad_bottom
```

### 2.2 图像处理实现
```python
def letterbox_image(image, target_size=(512, 256), fill_value=255):
    """
    对图像进行letterbox处理
    
    Args:
        image: PIL.Image 或 numpy array
        target_size: (W, H) = (512, 256)
        fill_value: padding填充值（255为白色）
    
    Returns:
        image_letterboxed: 处理后的图像
        transform_params: 变换参数字典
    """
    W_in, H_in = image.size if hasattr(image, 'size') else image.shape[1::-1]
    W_t, H_t = target_size
    
    # 计算letterbox参数
    s, W_prime, H_prime, pad_left, pad_right, pad_top, pad_bottom = \
        calculate_letterbox_params(W_in, H_in, W_t, H_t)
    
    # 等比缩放（双线性插值，align_corners=False）
    if hasattr(image, 'resize'):  # PIL Image
        image_resized = image.resize((W_prime, H_prime), Image.BILINEAR)
    else:  # numpy array
        import cv2
        image_resized = cv2.resize(image, (W_prime, H_prime), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # 创建目标画布并居中放置
    if hasattr(image, 'resize'):  # PIL
        canvas = Image.new('RGB', target_size, fill_value)
        canvas.paste(image_resized, (pad_left, pad_top))
    else:  # numpy
        canvas = np.full((H_t, W_t, 3), fill_value, dtype=np.uint8)
        canvas[pad_top:pad_top+H_prime, pad_left:pad_left+W_prime] = image_resized
    
    # 记录变换参数
    transform_params = {
        'scale': s,
        'pad_left': pad_left,
        'pad_top': pad_top,
        'pad_right': pad_right,
        'pad_bottom': pad_bottom,
        'resized_size': (W_prime, H_prime),
        'original_size': (W_in, H_in)
    }
    
    return canvas, transform_params
```

### 2.3 Padding Mask生成
```python
def create_padding_mask(transform_params, target_size=(512, 256)):
    """
    生成padding mask作为第4通道
    padding区域=1，原图区域=0
    """
    W_t, H_t = target_size
    mask = np.ones((H_t, W_t), dtype=np.float32)
    
    # 原图区域设为0
    pad_left = transform_params['pad_left']
    pad_top = transform_params['pad_top']
    W_prime, H_prime = transform_params['resized_size']
    
    mask[pad_top:pad_top+H_prime, pad_left:pad_left+W_prime] = 0
    
    return mask
```

## 3. 坐标映射

### 3.1 原坐标 → 网络输入坐标（像素域）

```python
def transform_coords_to_input(coords, transform_params):
    """
    将原始坐标映射到网络输入空间
    
    Args:
        coords: 各种格式的坐标
        transform_params: letterbox变换参数
    
    Returns:
        transformed_coords: 变换后的坐标
    """
    s = transform_params['scale']
    pad_left = transform_params['pad_left']
    pad_top = transform_params['pad_top']
    
    def transform_point(x, y):
        """点/关键点变换"""
        x_prime = s * x + pad_left
        y_prime = s * y + pad_top
        return x_prime, y_prime
    
    def transform_bbox(x1, y1, x2, y2):
        """轴对齐框变换"""
        x1_prime, y1_prime = transform_point(x1, y1)
        x2_prime, y2_prime = transform_point(x2, y2)
        return x1_prime, y1_prime, x2_prime, y2_prime
    
    def transform_rotated_bbox(cx, cy, w, h, theta):
        """旋转框变换"""
        cx_prime, cy_prime = transform_point(cx, cy)
        w_prime = s * w
        h_prime = s * h
        # 等比缩放下角度不变！
        theta_prime = theta
        return cx_prime, cy_prime, w_prime, h_prime, theta_prime
    
    def transform_polygon(points):
        """多边形变换"""
        transformed = []
        for x, y in points:
            x_prime, y_prime = transform_point(x, y)
            transformed.append((x_prime, y_prime))
        return transformed
    
    # 根据输入格式选择变换
    if isinstance(coords, tuple) and len(coords) == 2:
        return transform_point(*coords)
    elif isinstance(coords, dict):
        if 'cx' in coords:  # 旋转框
            return transform_rotated_bbox(
                coords['cx'], coords['cy'], 
                coords['w'], coords['h'], 
                coords.get('theta', 0)
            )
        elif 'x1' in coords:  # AABB
            return transform_bbox(
                coords['x1'], coords['y1'],
                coords['x2'], coords['y2']
            )
    elif isinstance(coords, list):  # 多边形
        return transform_polygon(coords)
    
    return coords
```

### 3.2 像素坐标 ↔ 1/4栅格坐标

```python
def pixel_to_grid(x_prime, y_prime, downsample=4):
    """
    像素坐标 → 1/4分辨率栅格坐标
    用于生成标签
    """
    u = x_prime / downsample  # 列（浮点）
    v = y_prime / downsample  # 行（浮点）
    
    # 分离整数部分和小数部分
    u_int = int(u)
    v_int = int(v)
    
    # 子像素偏移
    delta_u = u - u_int  # ∈[0,1)
    delta_v = v - v_int  # ∈[0,1)
    
    # 中心化到[-0.5, 0.5]用于训练
    offset_u = delta_u - 0.5
    offset_v = delta_v - 0.5
    
    return (u_int, v_int), (offset_u, offset_v)

def grid_to_pixel(grid_coords, offsets=None, downsample=4, method='argmax'):
    """
    栅格坐标 → 像素坐标
    用于推理恢复
    """
    if method == 'argmax':
        # 方法A: Argmax解码
        j_star, i_star = grid_coords  # 峰值格
        if offsets is not None:
            du, dv = offsets  # ∈[-0.5, 0.5]
            x_prime = 4 * (j_star + 0.5 + du)
            y_prime = 4 * (i_star + 0.5 + dv)
        else:
            x_prime = 4 * (j_star + 0.5)
            y_prime = 4 * (i_star + 0.5)
    
    elif method == 'soft_argmax':
        # 方法B: Soft-argmax/DSNT
        u_star, v_star = grid_coords  # 连续坐标
        x_prime = 4 * u_star
        y_prime = 4 * v_star
    
    return x_prime, y_prime
```

### 3.3 反变换（预测→原图坐标）

```python
def transform_coords_to_original(coords, transform_params):
    """
    将网络预测坐标还原到原始图像空间
    
    Args:
        coords: 网络输出的坐标
        transform_params: letterbox变换参数
    
    Returns:
        original_coords: 原始坐标
    """
    s = transform_params['scale']
    pad_left = transform_params['pad_left']
    pad_top = transform_params['pad_top']
    
    def inverse_point(x_prime, y_prime):
        """点的逆变换"""
        x = (x_prime - pad_left) / s
        y = (y_prime - pad_top) / s
        return x, y
    
    def inverse_rotated_bbox(cx_prime, cy_prime, w_prime, h_prime, theta_prime):
        """旋转框的逆变换"""
        cx, cy = inverse_point(cx_prime, cy_prime)
        w = w_prime / s
        h = h_prime / s
        # 等比缩放下角度不变
        theta = theta_prime
        return cx, cy, w, h, theta
    
    # 根据输入格式选择逆变换
    if isinstance(coords, tuple) and len(coords) == 2:
        return inverse_point(*coords)
    elif len(coords) == 5:  # 旋转框
        return inverse_rotated_bbox(*coords)
    
    return coords
```

## 4. 完整预处理流程

### 4.1 训练阶段
```python
class TrainingPreprocessor:
    def __init__(self, target_size=(512, 256), augment=True):
        self.target_size = target_size
        self.augment = augment
    
    def __call__(self, image_path, bgx, bgy, sdx, sdy, confusing_gaps=None):
        """
        训练预处理完整流程
        """
        # 1. 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 2. Letterbox处理
        image_letterboxed, transform_params = letterbox_image(
            image, self.target_size
        )
        
        # 3. 生成padding mask
        padding_mask = create_padding_mask(transform_params, self.target_size)
        
        # 4. 组合4通道输入 [RGB + padding_mask]
        image_array = np.array(image_letterboxed) / 255.0  # 归一化
        input_tensor = np.concatenate([
            image_array.transpose(2, 0, 1),  # HWC → CHW
            padding_mask[np.newaxis, :, :]   # 添加通道维度
        ], axis=0)  # [4, 256, 512]
        
        # 5. 坐标变换
        gap_x, gap_y = transform_coords_to_input((bgx, bgy), transform_params)
        piece_x, piece_y = transform_coords_to_input((sdx, sdy), transform_params)
        
        # 6. 生成1/4分辨率标签
        gap_grid, gap_offset = pixel_to_grid(gap_x, gap_y)
        piece_grid, piece_offset = pixel_to_grid(piece_x, piece_y)
        
        # 7. 生成热图和偏移标签
        labels = self.generate_labels(
            gap_grid, gap_offset,
            piece_grid, piece_offset,
            confusing_gaps, transform_params
        )
        
        # 8. 生成权重掩码（1/4分辨率）
        weight_mask = 1 - F.avg_pool2d(
            torch.from_numpy(padding_mask).unsqueeze(0).unsqueeze(0),
            kernel_size=4, stride=4
        ).squeeze().numpy()
        
        return {
            'input': input_tensor,
            'labels': labels,
            'weight_mask': weight_mask,
            'transform_params': transform_params
        }
    
    def generate_labels(self, gap_grid, gap_offset, piece_grid, piece_offset,
                       confusing_gaps, transform_params):
        """生成训练标签"""
        # 生成高斯热图（σ=1.5）
        gap_heatmap = self.generate_gaussian_heatmap(gap_grid, sigma=1.5)
        piece_heatmap = self.generate_gaussian_heatmap(piece_grid, sigma=1.5)
        
        # 生成偏移标签（已中心化到[-0.5, 0.5]）
        offset_map = np.zeros((4, 64, 128), dtype=np.float32)
        gy, gx = gap_grid
        py, px = piece_grid
        
        if 0 <= gx < 128 and 0 <= gy < 64:
            offset_map[0, gy, gx] = gap_offset[0]   # gap_u
            offset_map[1, gy, gx] = gap_offset[1]   # gap_v
        
        if 0 <= px < 128 and 0 <= py < 64:
            offset_map[2, py, px] = piece_offset[0]  # piece_u
            offset_map[3, py, px] = piece_offset[1]  # piece_v
        
        # 处理混淆缺口
        confusing_coords = []
        if confusing_gaps:
            for cx, cy in confusing_gaps:
                cx_prime, cy_prime = transform_coords_to_input(
                    (cx, cy), transform_params
                )
                conf_grid, _ = pixel_to_grid(cx_prime, cy_prime)
                confusing_coords.append(conf_grid)
        
        return {
            'heatmaps': np.stack([gap_heatmap, piece_heatmap]),
            'offsets': offset_map,
            'confusing_gaps': confusing_coords
        }
    
    def generate_gaussian_heatmap(self, center, size=(64, 128), sigma=1.5):
        """生成高斯热图"""
        H, W = size
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        cy, cx = center
        for i in range(max(0, cy-3*int(sigma)), min(H, cy+3*int(sigma)+1)):
            for j in range(max(0, cx-3*int(sigma)), min(W, cx+3*int(sigma)+1)):
                dist_sq = (i - cy)**2 + (j - cx)**2
                heatmap[i, j] = np.exp(-dist_sq / (2 * sigma**2))
        
        return heatmap
```

### 4.2 推理阶段
```python
class InferencePreprocessor:
    def __init__(self, target_size=(512, 256)):
        self.target_size = target_size
    
    def preprocess(self, image):
        """推理预处理"""
        # Letterbox处理
        image_letterboxed, transform_params = letterbox_image(
            image, self.target_size
        )
        
        # 生成padding mask
        padding_mask = create_padding_mask(transform_params)
        
        # 组合4通道输入
        image_array = np.array(image_letterboxed) / 255.0
        input_tensor = np.concatenate([
            image_array.transpose(2, 0, 1),
            padding_mask[np.newaxis, :, :]
        ], axis=0)
        
        return input_tensor, transform_params
    
    def postprocess(self, predictions, transform_params):
        """
        将网络预测映射回原始坐标
        
        Args:
            predictions: {
                'gap': (grid_coords, offset),
                'piece': (grid_coords, offset)
            }
            transform_params: letterbox变换参数
        """
        results = {}
        
        for key in ['gap', 'piece']:
            grid_coords, offset = predictions[key]
            
            # 1/4栅格 → 像素坐标
            x_prime, y_prime = grid_to_pixel(
                grid_coords, offset, method='argmax'
            )
            
            # 像素坐标 → 原始坐标
            x, y = transform_coords_to_original(
                (x_prime, y_prime), transform_params
            )
            
            results[key] = (x, y)
        
        return results
```

## 5. 重要说明

### 5.1 为什么用Letterbox？
1. **几何不变性**：等比缩放保持形状和角度不变（θ' = θ）
2. **精度保证**：避免拉伸导致的定位误差
3. **统一输入**：固定尺寸[B,4,256,512]便于批处理

### 5.2 Padding Mask的作用
- **训练时**：损失函数乘以`(1 - M_pad/4)`权重，屏蔽padding区域
- **推理时**：可用于后处理，过滤padding区域的假阳性

### 5.3 坐标精度验证
```python
def verify_coordinate_precision():
    """验证坐标映射精度"""
    # 原始坐标
    original = (120.5, 70.3)
    
    # 模拟letterbox参数（320×160 → 512×256）
    transform_params = {
        'scale': 1.6,
        'pad_left': 0,
        'pad_top': 0
    }
    
    # 正变换
    x_prime, y_prime = transform_coords_to_input(original, transform_params)
    
    # 映射到1/4栅格
    grid, offset = pixel_to_grid(x_prime, y_prime)
    
    # 恢复到像素
    x_recovered_prime, y_recovered_prime = grid_to_pixel(grid, offset)
    
    # 逆变换
    x_recovered, y_recovered = transform_coords_to_original(
        (x_recovered_prime, y_recovered_prime), transform_params
    )
    
    # 验证误差
    error_x = abs(x_recovered - original[0])
    error_y = abs(y_recovered - original[1])
    
    print(f"原始: {original}")
    print(f"恢复: ({x_recovered:.2f}, {y_recovered:.2f})")
    print(f"误差: ({error_x:.3f}px, {error_y:.3f}px)")
    
    assert error_x < 0.5 and error_y < 0.5, "精度不满足要求"
```

## 6. 与网络架构的对应关系

| 组件 | 规格 | 说明 |
|------|------|------|
| 网络输入 | [B, 4, 256, 512] | RGB(3) + padding_mask(1) |
| 主特征Hf | [B, 128, 64, 128] | 1/4分辨率特征图 |
| 热图标签 | [B, 2, 64, 128] | gap和piece的高斯热图 |
| 偏移标签 | [B, 4, 64, 128] | 4通道子像素偏移 |
| 权重掩码 | [B, 1, 64, 128] | 屏蔽padding区域 |

