# -*- coding: utf-8 -*-
"""
核心预处理模块
实现Letterbox变换、坐标映射和4通道输入生成
"""
from typing import Tuple, Dict, Optional, Union, List, Any
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F


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
            padding mask (H, W)，padding区域=1，原图区域=0
        """
        w_t, h_t = params['target_size']
        mask = np.ones((h_t, w_t), dtype=np.float32)
        
        # 原图区域设为0
        pad_left = params['pad_left']
        pad_top = params['pad_top']
        w_prime, h_prime = params['resized_size']
        
        mask[pad_top:pad_top+h_prime, pad_left:pad_left+w_prime] = 0
        
        return mask


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
            # 处理numpy数组
            coords_prime = coords.copy().astype(np.float32)
            coords_prime[..., 0] = scale * coords[..., 0] + pad_left
            coords_prime[..., 1] = scale * coords[..., 1] + pad_top
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
            coords_orig = coords.copy().astype(np.float32)
            coords_orig[..., 0] = (coords[..., 0] - pad_left) / scale
            coords_orig[..., 1] = (coords[..., 1] - pad_top) / scale
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
    生成4通道输入和完整标签
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
                  gap_angle: float = 0.0) -> Dict[str, Union[np.ndarray, torch.Tensor, Dict]]:
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
            - input: 4通道输入张量 [4, H, W]
            - heatmaps: 热图标签 [2, H/4, W/4]
            - offsets: 偏移标签 [4, H/4, W/4]
            - weight_mask: 权重掩码 [H/4, W/4]
            - confusing_gaps: 混淆缺口栅格坐标
            - gap_angle: 缺口角度（弧度）
            - transform_params: 变换参数
        """
        # 1. 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # 2. Letterbox处理
        image_letterboxed, transform_params = self.letterbox.apply(image)
        
        # 3. 生成padding mask
        padding_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 4. 组合4通道输入
        image_normalized = image_letterboxed.astype(np.float32) / 255.0
        input_tensor = np.concatenate([
            image_normalized.transpose(2, 0, 1),  # [3, H, W]
            padding_mask[np.newaxis, :, :]        # [1, H, W]
        ], axis=0)  # [4, H, W]
        
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
        
        # 9. 生成权重掩码（1/4分辨率）
        weight_mask = self._generate_weight_mask(padding_mask)
        
        # 10. 处理混淆缺口
        confusing_grids = []
        if confusing_gaps:
            for conf_gap in confusing_gaps:
                conf_input = self.coord_transform.original_to_input(conf_gap, transform_params)
                conf_grid, _ = self.coord_transform.pixel_to_grid(*conf_input)
                confusing_grids.append(conf_grid)
        
        return {
            'input': torch.from_numpy(input_tensor).float(),
            'heatmaps': torch.from_numpy(heatmaps).float(),
            'offsets': torch.from_numpy(offsets).float(),
            'weight_mask': torch.from_numpy(weight_mask).float(),
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
    
    def _generate_weight_mask(self, padding_mask: np.ndarray) -> np.ndarray:
        """
        生成1/4分辨率的权重掩码
        
        Args:
            padding_mask: 原始分辨率的padding mask
        
        Returns:
            权重掩码 [H/4, W/4]，有效区域=1，padding区域=0
        """
        # 使用平均池化下采样
        mask_tensor = torch.from_numpy(padding_mask).unsqueeze(0).unsqueeze(0).float()
        mask_downsampled = F.avg_pool2d(mask_tensor, kernel_size=self.downsample, 
                                       stride=self.downsample)
        
        # 转换：padding mask是1表示padding，权重掩码需要相反
        weight_mask = 1 - mask_downsampled.squeeze().numpy()
        
        return weight_mask


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
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, str]) -> Tuple[torch.Tensor, Dict]:
        """
        推理预处理
        
        Args:
            image: 输入图像或图像路径
        
        Returns:
            (4通道输入张量, 变换参数)
        """
        # 加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Letterbox处理
        image_letterboxed, transform_params = self.letterbox.apply(image)
        
        # 生成padding mask
        padding_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 组合4通道输入
        image_normalized = image_letterboxed.astype(np.float32) / 255.0
        input_tensor = np.concatenate([
            image_normalized.transpose(2, 0, 1),
            padding_mask[np.newaxis, :, :]
        ], axis=0)
        
        return torch.from_numpy(input_tensor).float(), transform_params
    
    def postprocess(self, predictions: Dict[str, Tuple[Tuple[int, int], Tuple[float, float]]],
                   transform_params: Dict) -> Dict[str, Tuple[float, float]]:
        """
        将网络预测映射回原始坐标
        
        Args:
            predictions: 网络预测，格式为 {
                'gap': ((grid_y, grid_x), (offset_v, offset_u)),
                'slider': ((grid_y, grid_x), (offset_v, offset_u))
            }
            transform_params: letterbox变换参数
        
        Returns:
            原始坐标，格式为 {'gap': (x, y), 'slider': (x, y)}
        """
        results = {}
        
        for key in ['gap', 'slider']:
            if key not in predictions:
                continue
            
            grid_coords, offsets = predictions[key]
            # 注意：grid_coords是(y, x)格式，需要转换
            grid_y, grid_x = grid_coords
            offset_v, offset_u = offsets if offsets else (0, 0)
            
            # 1/4栅格 → 像素坐标
            x_prime, y_prime = self.coord_transform.grid_to_pixel(
                (grid_x, grid_y), (offset_u, offset_v)
            )
            
            # 像素坐标 → 原始坐标
            x, y = self.coord_transform.input_to_original(
                (x_prime, y_prime), transform_params
            )
            
            results[key] = (x, y)
        
        return results
    
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