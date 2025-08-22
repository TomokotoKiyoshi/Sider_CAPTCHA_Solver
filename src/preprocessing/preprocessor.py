# -*- coding: utf-8 -*-
"""
核心预处理模块
实现Letterbox变换、坐标映射和4通道输入生成
"""
from typing import Tuple, Dict, Optional, Union, List, Any
import numpy as np
import cv2
from PIL import Image
# 移除PyTorch以解决内存泄漏
# import torch
# import torch.nn.functional as F


class LetterboxTransform:
    """
    Letterbox变换类
    实现等比缩放+居中填充，保持几何不变性
    """
    
    def __init__(self, target_size: Tuple[int, int], fill_value: int):
        """
        初始化Letterbox变换
        
        Args:
            target_size: 目标尺寸 (W, H)，必须提供
            fill_value: padding填充值，必须提供
        """
        if target_size is None:
            raise ValueError("target_size must be provided, please load from config file")
        if fill_value is None:
            raise ValueError("fill_value must be provided, please load from config file")
        
        self.target_size = target_size
        self.fill_value = fill_value
    
    def calculate_params(self, w_in: int, h_in: int) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        """
        计算letterbox参数
        
        Args:
            w_in: 原始图像宽度
            h_in: 原始图像高度
        
        Returns:
            变换参数字典，包含缩放比例、缩放后尺寸、padding值等
        """
        w_t, h_t = self.target_size
        
        # 计算等比缩放因子
        scale = min(w_t / w_in, h_t / h_in)
        
        # 缩放后尺寸
        w_prime = round(w_in * scale)
        h_prime = round(h_in * scale)
        
        # 计算padding
        pad_x = w_t - w_prime
        pad_y = h_t - h_prime
        
        # 居中padding
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        
        return {
            'scale': scale,
            'resized_size': (w_prime, h_prime),
            'pad_left': pad_left,
            'pad_right': pad_right,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'original_size': (w_in, h_in),
            'target_size': self.target_size
        }
    
    def apply(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, Dict]:
        """
        对图像应用letterbox变换
        
        Args:
            image: 输入图像（PIL.Image或numpy array）
        
        Returns:
            (变换后的图像, 变换参数)
        """
        # 获取原始尺寸
        if isinstance(image, Image.Image):
            w_in, h_in = image.size
            image = np.array(image)
        else:
            h_in, w_in = image.shape[:2]
        
        # 计算变换参数
        params = self.calculate_params(w_in, h_in)
        
        # 等比缩放
        w_prime, h_prime = params['resized_size']
        image_resized = cv2.resize(image, (w_prime, h_prime), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标画布并居中放置
        w_t, h_t = self.target_size
        if len(image.shape) == 3:
            canvas = np.full((h_t, w_t, image.shape[2]), self.fill_value, dtype=np.uint8)
        else:
            canvas = np.full((h_t, w_t), self.fill_value, dtype=np.uint8)
        
        pad_left = params['pad_left']
        pad_top = params['pad_top']
        canvas[pad_top:pad_top+h_prime, pad_left:pad_left+w_prime] = image_resized
        
        return canvas, params
    
    def create_padding_mask(self, params: Dict) -> np.ndarray:
        """
        生成padding mask作为第4通道
        
        Args:
            params: letterbox变换参数
        
        Returns:
            padding mask (H, W)，padding区域=0，原图区域=1（根据PREPROCESSING_OUTPUT.md）
        """
        w_t, h_t = params['target_size']
        # 初始化为0（padding区域）
        mask = np.zeros((h_t, w_t), dtype=np.float32)
        
        # 原图区域设为1（有效区域）
        pad_left = params['pad_left']
        pad_top = params['pad_top']
        w_prime, h_prime = params['resized_size']
        
        mask[pad_top:pad_top+h_prime, pad_left:pad_left+w_prime] = 1
        
        return mask
    
    def downsample_padding_mask(self, mask: np.ndarray, downsample: int = 4) -> np.ndarray:
        """
        使用平均池化下采样padding mask
        P_1/4 = AvgPool_{k=4,s=4}(M_pad)
        
        Args:
            mask: padding mask (H, W)
            downsample: 下采样率（默认4）
        
        Returns:
            下采样后的mask (H/4, W/4)
        """
        h, w = mask.shape
        h_out = h // downsample
        w_out = w // downsample
        
        # 使用平均池化：每个4x4区域计算平均值
        mask_downsampled = np.zeros((h_out, w_out), dtype=np.float32)
        
        for i in range(h_out):
            for j in range(w_out):
                # 提取4x4窗口
                window = mask[i*downsample:(i+1)*downsample, 
                             j*downsample:(j+1)*downsample]
                # 计算平均值
                mask_downsampled[i, j] = np.mean(window)
        
        return mask_downsampled


class CoordinateTransform:
    """
    坐标变换类
    处理原始坐标、网络输入坐标、1/4栅格坐标之间的映射
    """
    
    def __init__(self, downsample: int):
        """
        初始化坐标变换
        
        Args:
            downsample: 下采样率，必须提供
        """
        if downsample is None:
            raise ValueError("downsample must be provided, please load from config file")
        
        self.downsample = downsample
    
    def original_to_input(self, coords: Union[Tuple[float, float], np.ndarray], 
                         params: Dict) -> Union[Tuple[float, float], np.ndarray]:
        """
        原始坐标 → 网络输入坐标
        
        Args:
            coords: 原始坐标 (x, y) 或坐标数组
            params: letterbox变换参数
        
        Returns:
            网络输入空间的坐标
        """
        scale = params['scale']
        pad_left = params['pad_left']
        pad_top = params['pad_top']
        
        if isinstance(coords, tuple):
            x, y = coords
            x_prime = scale * x + pad_left
            y_prime = scale * y + pad_top
            return (x_prime, y_prime)
        else:
            # 处理numpy数组 - 优化：避免双重复制
            coords_prime = coords.astype(np.float32, copy=True)
            coords_prime[..., 0] = scale * coords_prime[..., 0] + pad_left
            coords_prime[..., 1] = scale * coords_prime[..., 1] + pad_top
            return coords_prime
    
    def input_to_original(self, coords: Union[Tuple[float, float], np.ndarray],
                         params: Dict) -> Union[Tuple[float, float], np.ndarray]:
        """
        网络输入坐标 → 原始坐标（逆变换）
        
        Args:
            coords: 网络输入坐标 (x', y') 或坐标数组
            params: letterbox变换参数
        
        Returns:
            原始图像空间的坐标
        """
        scale = params['scale']
        pad_left = params['pad_left']
        pad_top = params['pad_top']
        
        if isinstance(coords, tuple):
            x_prime, y_prime = coords
            x = (x_prime - pad_left) / scale
            y = (y_prime - pad_top) / scale
            return (x, y)
        else:
            # 优化：避免双重复制
            coords_orig = coords.astype(np.float32, copy=True)
            coords_orig[..., 0] = (coords_orig[..., 0] - pad_left) / scale
            coords_orig[..., 1] = (coords_orig[..., 1] - pad_top) / scale
            return coords_orig
    
    def pixel_to_grid(self, x: float, y: float) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        像素坐标 → 1/4栅格坐标
        
        Args:
            x, y: 像素坐标
        
        Returns:
            ((栅格索引), (子像素偏移))
        """
        u = x / self.downsample  # 列（浮点）
        v = y / self.downsample  # 行（浮点）
        
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
    
    def grid_to_pixel(self, grid_coords: Tuple[int, int], 
                     offsets: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        栅格坐标 → 像素坐标
        
        Args:
            grid_coords: 栅格索引 (j, i)
            offsets: 子像素偏移 (du, dv)，范围[-0.5, 0.5]
        
        Returns:
            像素坐标 (x, y)
        """
        j, i = grid_coords
        
        if offsets is not None:
            du, dv = offsets
            x = self.downsample * (j + 0.5 + du)
            y = self.downsample * (i + 0.5 + dv)
        else:
            x = self.downsample * (j + 0.5)
            y = self.downsample * (i + 0.5)
        
        return (x, y)


class TrainingPreprocessor:
    """
    训练预处理器
    生成2通道输入和完整标签
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练预处理器
        
        Args:
            config: 配置字典，必须包含preprocessing相关配置
        """
        if config is None:
            raise ValueError("Config dictionary must be provided")
        
        # 从配置中提取参数
        try:
            letterbox_cfg = config['preprocessing']['letterbox']
            coord_cfg = config['preprocessing']['coordinate']
            heatmap_cfg = config['preprocessing']['heatmap']
            
            target_size = tuple(letterbox_cfg['target_size'])
            fill_value = letterbox_cfg['fill_value']
            downsample = coord_cfg['downsample']
            sigma = heatmap_cfg['sigma']
        except KeyError as e:
            raise ValueError(f"Config missing required key: {e}")
        
        # 提取量化配置（如果存在）
        quantization_cfg = config['preprocessing'].get('quantization', {})
        self.quantization_enabled = quantization_cfg.get('enabled', False)
        self.bits_to_keep = quantization_cfg.get('bits_to_keep', 8)
        
        # 计算量化掩码
        if self.quantization_enabled and self.bits_to_keep < 8:
            # 生成掩码：保留高N位，清除低(8-N)位
            # 例如：bits_to_keep=4 -> mask=0b11110000 (240)
            self.quantization_mask = ((1 << self.bits_to_keep) - 1) << (8 - self.bits_to_keep)
        else:
            self.quantization_mask = 0xFF  # 不量化，保留所有位
        
        self.letterbox = LetterboxTransform(target_size, fill_value)
        self.coord_transform = CoordinateTransform(downsample)
        self.target_size = target_size
        self.downsample = downsample
        self.sigma = sigma
        
        # 计算1/4分辨率的尺寸
        self.grid_size = (target_size[1] // downsample, target_size[0] // downsample)  # (H, W)
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, str],
                  gap_center: Tuple[int, int],
                  slider_center: Tuple[int, int],
                  confusing_gaps: Optional[List[Tuple[int, int]]] = None,
                  gap_angle: float = 0.0) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        完整的训练预处理流程
        
        Args:
            image: 输入图像或图像路径
            gap_center: 缺口中心坐标 (x, y)
            slider_center: 滑块中心坐标 (x, y)
            confusing_gaps: 混淆缺口坐标列表
            gap_angle: 缺口旋转角度（度）
        
        Returns:
            预处理结果字典，包含：
            - input: 2通道输入张量 [2, H, W]（第2通道是padding mask）
            - heatmaps: 热图标签 [2, H/4, W/4]
            - offsets: 偏移标签 [4, H/4, W/4]
            - confusing_gaps: 混淆缺口栅格坐标
            - gap_angle: 缺口角度（弧度）
            - transform_params: 变换参数
            注：权重掩码可从第4通道下采样获得，无需单独生成
        """
        # 1. 加载图像并转换为灰度图
        if isinstance(image, str):
            image = cv2.imread(image)
            # 转换为灰度图
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            # 如果是RGB图像，转换为灰度图
            if len(image.shape) == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image
        else:
            # 如果已经是numpy数组
            if len(image.shape) == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image
        
        # 应用量化（如果启用）
        if self.quantization_enabled:
            grayscale = grayscale & self.quantization_mask
        
        # 2. Letterbox处理（灰度图）
        image_letterboxed, transform_params = self.letterbox.apply(grayscale)
        
        # 3. 生成padding mask
        padding_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 4. 组合2通道输入 - 灰度图 + padding mask
        image_normalized = image_letterboxed.astype(np.float32) / 255.0
        input_tensor = np.stack([
            image_normalized,                      # [H, W] - 灰度图
            padding_mask                           # [H, W] - padding mask
        ], axis=0)  # [2, H, W]
        
        # 5. 坐标变换
        gap_input = self.coord_transform.original_to_input(gap_center, transform_params)
        slider_input = self.coord_transform.original_to_input(slider_center, transform_params)
        
        # 6. 生成1/4分辨率标签
        gap_grid, gap_offset = self.coord_transform.pixel_to_grid(*gap_input)
        slider_grid, slider_offset = self.coord_transform.pixel_to_grid(*slider_input)
        
        # 7. 生成热图
        gap_heatmap = self._generate_gaussian_heatmap(gap_grid, self.grid_size, self.sigma)
        slider_heatmap = self._generate_gaussian_heatmap(slider_grid, self.grid_size, self.sigma)
        heatmaps = np.stack([gap_heatmap, slider_heatmap], axis=0)  # [2, H/4, W/4]
        
        # 8. 生成偏移标签
        offsets = self._generate_offset_map(gap_grid, gap_offset, slider_grid, slider_offset)
        
        # 9. 处理混淆缺口
        confusing_grids = []
        if confusing_gaps:
            for conf_gap in confusing_gaps:
                conf_input = self.coord_transform.original_to_input(conf_gap, transform_params)
                conf_grid, _ = self.coord_transform.pixel_to_grid(*conf_input)
                confusing_grids.append(conf_grid)
        
        # 优化：数据已经是float32，不需要再转换
        return {
            'input': input_tensor,  # 已经是float32
            'heatmaps': heatmaps,   # 已经是float32
            'offsets': offsets,     # 已经是float32
            'gap_grid': gap_grid,  # 添加网格坐标
            'gap_offset': gap_offset,  # 添加偏移
            'slider_grid': slider_grid,  # 添加滑块网格
            'slider_offset': slider_offset,  # 添加滑块偏移
            'confusing_gaps': confusing_grids,
            'gap_angle': np.radians(gap_angle),  # 转换为弧度
            'transform_params': transform_params
        }
    
    def _generate_gaussian_heatmap(self, center: Tuple[int, int], 
                                  size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        生成高斯热图
        
        Args:
            center: 中心栅格坐标 (列, 行)
            size: 热图尺寸 (H, W)
            sigma: 高斯标准差（栅格单位）
        
        Returns:
            高斯热图 [H, W]
        """
        h, w = size
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        cx, cy = center  # cx=列(x), cy=行(y)
        if not (0 <= cx < w and 0 <= cy < h):
            print(f"Warning: Center out of bounds in heatmap: col={cx}, row={cy}, size=(h={h}, w={w})")
            return heatmap  # 中心在图像外
        
        # 计算高斯分布
        radius = int(3 * sigma)
        for i in range(max(0, cy - radius), min(h, cy + radius + 1)):
            for j in range(max(0, cx - radius), min(w, cx + radius + 1)):
                dist_sq = (i - cy) ** 2 + (j - cx) ** 2
                heatmap[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
        
        return heatmap
    
    def _generate_offset_map(self, gap_grid: Tuple[int, int], gap_offset: Tuple[float, float],
                            slider_grid: Tuple[int, int], slider_offset: Tuple[float, float]) -> np.ndarray:
        """
        生成偏移标签
        
        Args:
            gap_grid: 缺口栅格坐标 (列, 行)
            gap_offset: 缺口子像素偏移 (du, dv)
            slider_grid: 滑块栅格坐标 (列, 行)
            slider_offset: 滑块子像素偏移 (du, dv)
        
        Returns:
            偏移标签 [4, H/4, W/4]
        """
        h, w = self.grid_size
        offset_map = np.zeros((4, h, w), dtype=np.float32)
        
        # Gap偏移 - 修复：gap_grid是(列,行)格式
        gap_col, gap_row = gap_grid  # 列(x), 行(y)
        if 0 <= gap_col < w and 0 <= gap_row < h:
            offset_map[0, gap_row, gap_col] = gap_offset[0]   # gap_du (x方向偏移)
            offset_map[1, gap_row, gap_col] = gap_offset[1]   # gap_dv (y方向偏移)
        else:
            print(f"Warning: Gap position out of bounds: col={gap_col}, row={gap_row}, grid_size=(h={h}, w={w})")
        
        # Slider偏移 - 修复：slider_grid是(列,行)格式
        slider_col, slider_row = slider_grid  # 列(x), 行(y)
        if 0 <= slider_col < w and 0 <= slider_row < h:
            offset_map[2, slider_row, slider_col] = slider_offset[0]  # slider_du (x方向偏移)
            offset_map[3, slider_row, slider_col] = slider_offset[1]  # slider_dv (y方向偏移)
        else:
            print(f"Warning: Slider position out of bounds: col={slider_col}, row={slider_row}, grid_size=(h={h}, w={w})")
        
        return offset_map


class InferencePreprocessor:
    """
    推理预处理器
    处理推理时的图像预处理和坐标恢复
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理预处理器
        
        Args:
            config: 配置字典，必须包含preprocessing相关配置
        """
        if config is None:
            raise ValueError("Config dictionary must be provided")
        
        # 从配置中提取参数
        try:
            letterbox_cfg = config['preprocessing']['letterbox']
            coord_cfg = config['preprocessing']['coordinate']
            
            target_size = tuple(letterbox_cfg['target_size'])
            fill_value = letterbox_cfg['fill_value']
            downsample = coord_cfg['downsample']
        except KeyError as e:
            raise ValueError(f"Config missing required key: {e}")
        
        self.letterbox = LetterboxTransform(target_size, fill_value)
        self.coord_transform = CoordinateTransform(downsample)
        self.target_size = target_size
        self.downsample = downsample
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, str]) -> Tuple[np.ndarray, Dict]:
        """
        推理预处理
        
        Args:
            image: 输入图像或图像路径
        
        Returns:
            (2通道输入张量, 变换参数)
        """
        # 加载图像并转换为灰度图
        if isinstance(image, str):
            image = cv2.imread(image)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image
        else:
            if len(image.shape) == 3:
                grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                grayscale = image
        
        # 应用量化（如果启用）
        if self.quantization_enabled:
            grayscale = grayscale & self.quantization_mask
        
        # Letterbox处理（灰度图）
        image_letterboxed, transform_params = self.letterbox.apply(grayscale)
        
        # 生成padding mask
        padding_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 组合2通道输入
        image_normalized = image_letterboxed.astype(np.float32) / 255.0
        input_tensor = np.stack([
            image_normalized,
            padding_mask
        ], axis=0)
        
        # 推理时返回NumPy数组，避免在多进程中创建PyTorch张量
        return input_tensor.astype(np.float32), transform_params
    
    
    def verify_precision(self, original_coord: Tuple[float, float],
                        transform_params: Dict) -> Dict[str, float]:
        """
        验证坐标映射精度
        
        Args:
            original_coord: 原始坐标
            transform_params: 变换参数
        
        Returns:
            精度验证结果
        """
        # 正变换
        input_coord = self.coord_transform.original_to_input(original_coord, transform_params)
        
        # 映射到栅格
        grid_coord, offset = self.coord_transform.pixel_to_grid(*input_coord)
        
        # 恢复到像素
        recovered_input = self.coord_transform.grid_to_pixel(grid_coord, offset)
        
        # 逆变换
        recovered_original = self.coord_transform.input_to_original(recovered_input, transform_params)
        
        # 计算误差
        error_x = abs(recovered_original[0] - original_coord[0])
        error_y = abs(recovered_original[1] - original_coord[1])
        
        return {
            'original': original_coord,
            'recovered': recovered_original,
            'error_x': error_x,
            'error_y': error_y,
            'total_error': np.sqrt(error_x**2 + error_y**2)
        }