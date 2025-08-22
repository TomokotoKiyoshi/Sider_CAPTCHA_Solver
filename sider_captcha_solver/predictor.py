"""高层预测器API，封装底层模型调用"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
from tqdm import tqdm
import time

# 导入src中的模型代码
import sys
package_dir = Path(__file__).parent.parent
src_path = package_dir / 'src'
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.models.lite_hrnet_18_fpn import create_lite_hrnet_18_fpn
from src.preprocessing.preprocessor import LetterboxTransform, CoordinateTransform
from src.config.config_loader import ConfigLoader


class CaptchaPredictor:
    """滑块验证码预测器
    
    提供高层API用于滑块验证码识别，自动处理模型加载、图像预处理和结果后处理。
    """
    
    def __init__(self, 
                 model_path: str = 'best',
                 device: str = 'auto',
                 hm_threshold: float = 0.1):
        """初始化预测器
        
        Args:
            model_path: 模型路径或预设名称('best'/'v1.1.0')
            device: 运行设备 ('auto'/'cuda'/'cpu')
            hm_threshold: 热力图阈值，用于筛选有效检测
        """
        self.device = self._setup_device(device)
        self.threshold = hm_threshold
        
        # 加载配置
        self.config_loader = ConfigLoader()
        self.model_config = self.config_loader.get_config('model_config')
        
        # 初始化预处理器 - 使用正确的 512x256 尺寸
        self.letterbox = LetterboxTransform(
            target_size=(512, 256),  # 宽x高
            fill_value=255  # 白色填充
        )
        self.coord_transform = CoordinateTransform(downsample=4)
        
        # 加载模型
        self.model = create_lite_hrnet_18_fpn()
        
        # 加载权重
        weights_path = self._get_model_path(model_path)
        self._load_weights(weights_path)
        
        # 将模型移到指定设备
        self.model.to(self.device)
        self.model.eval()
        
        # 验证设备并显示详细信息
        if str(self.device) == 'cuda':
            # 确认模型在GPU上
            if next(self.model.parameters()).is_cuda:
                print(f"✅ 模型加载成功 | 设备: GPU ({torch.cuda.get_device_name(0)})")
                # 显示GPU内存使用
                allocated = torch.cuda.memory_allocated() / 1024**2
                print(f"   GPU内存使用: {allocated:.1f} MB")
            else:
                print(f"⚠️ 模型加载成功但未在GPU上 | 设备: {self.device}")
        else:
            print(f"✅ 模型加载成功 | 设备: CPU")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置运行设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _get_model_path(self, model_path: str) -> Path:
        """获取模型路径，支持本地路径和预设模型名
        
        优先使用 best_model_weights.pth（更小，推理专用）
        如果不存在，则使用 best_model.pth（完整检查点）
        """
        # 如果是本地路径且存在
        if Path(model_path).exists():
            return Path(model_path)
        
        # 尝试在src/checkpoints中查找
        checkpoints_dir = Path(__file__).parent.parent / 'src' / 'checkpoints'
        if checkpoints_dir.exists():
            # 查找best模型 - 默认使用1.1.0版本
            if model_path == 'best':
                # 优先查找1.1.0/best_model_weights.pth（推理专用，文件更小）
                weights_path = checkpoints_dir / '1.1.0' / 'best_model_weights.pth'
                if weights_path.exists():
                    print(f"   使用推理权重: {weights_path.name} (12.96 MB)")
                    return weights_path
                
                # 其次查找1.1.0/best_model.pth（完整检查点）
                best_path = checkpoints_dir / '1.1.0' / 'best_model.pth'
                if best_path.exists():
                    print(f"   使用完整检查点: {best_path.name} (50.75 MB)")
                    return best_path
                
                # 否则查找任意版本的权重文件
                for checkpoint_path in checkpoints_dir.glob('**/best_model_weights.pth'):
                    print(f"   使用推理权重: {checkpoint_path.name}")
                    return checkpoint_path
                for checkpoint_path in checkpoints_dir.glob('**/best_model.pth'):
                    print(f"   使用完整检查点: {checkpoint_path.name}")
                    return checkpoint_path
            
            # 查找指定版本 (支持 "1.1.0" 或 "v1.1.0" 格式)
            version = model_path.lstrip('v')
            
            # 优先尝试weights版本
            weights_path = checkpoints_dir / version / "best_model_weights.pth"
            if weights_path.exists():
                print(f"   使用推理权重: {weights_path.name}")
                return weights_path
            
            # 否则使用完整版本
            full_path = checkpoints_dir / version / "best_model.pth"
            if full_path.exists():
                print(f"   使用完整检查点: {full_path.name}")
                return full_path
        
        raise ValueError(f"找不到模型: {model_path}")
    
    def _load_weights(self, weights_path: Path):
        """加载模型权重"""
        # 直接加载到目标设备以提高效率
        # 如果是GPU，直接加载到GPU；否则加载到CPU
        if str(self.device) == 'cuda' and torch.cuda.is_available():
            checkpoint = torch.load(weights_path, map_location=self.device)
        else:
            checkpoint = torch.load(weights_path, map_location='cpu')
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
    
    def predict(self, image_path: Union[str, Path, np.ndarray]) -> Dict:
        """预测单张图片
        
        Args:
            image_path: 图片路径或numpy数组
        
        Returns:
            Dict: 包含以下字段的字典:
                - success: 是否成功检测
                - sliding_distance: 滑动距离(像素)
                - gap_x, gap_y: 缺口中心坐标
                - slider_x, slider_y: 滑块中心坐标
                - gap_confidence: 缺口置信度
                - slider_confidence: 滑块置信度
                - confidence: 综合置信度
                - processing_time_ms: 处理时间(毫秒)
        """
        start_time = time.time()
        
        try:
            # 预处理
            image_tensor = self._preprocess(image_path)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                predictions = self.model.decode_predictions(outputs, input_images=image_tensor)
            
            # 后处理
            result = self._format_result(predictions)
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def predict_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> List[Dict]:
        """批量预测多张图片 - 支持真正的批量GPU推理
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小（根据GPU内存调整）
        
        Returns:
            List[Dict]: 预测结果列表
        """
        results = []
        num_images = len(image_paths)
        
        # 分批处理
        for batch_start in tqdm(range(0, num_images, batch_size), desc="批量推理"):
            batch_end = min(batch_start + batch_size, num_images)
            batch_paths = image_paths[batch_start:batch_end]
            
            # 批量预处理
            batch_tensors = []
            batch_params = []
            batch_original_sizes = []
            valid_indices = []
            
            for idx, image_path in enumerate(batch_paths):
                try:
                    # 预处理单张图片
                    image_tensor, transform_params, original_size = self._preprocess_with_params(image_path)
                    batch_tensors.append(image_tensor)
                    batch_params.append(transform_params)
                    batch_original_sizes.append(original_size)
                    valid_indices.append(idx)
                except Exception as e:
                    # 记录失败的图片
                    results.append({
                        'success': False,
                        'error': str(e),
                        'image_path': str(image_path)
                    })
            
            if batch_tensors:
                # 合并成批次tensor
                batch_tensor = torch.cat(batch_tensors, dim=0)  # [B, 2, 256, 512]
                
                # 批量推理
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    predictions = self.model.decode_predictions(outputs, input_images=batch_tensor)
                
                # 批量后处理
                for i, idx in enumerate(valid_indices):
                    try:
                        result = self._format_batch_result(
                            predictions, i, batch_params[i], batch_original_sizes[i]
                        )
                        result['success'] = True
                        result['image_path'] = str(batch_paths[idx])
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e),
                            'image_path': str(batch_paths[idx])
                        })
        
        return results
    
    def predict_batch_folder(self, folder_path: Union[str, Path], 
                            batch_size: int = 32,
                            extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[Dict]:
        """批量预测整个文件夹的图片
        
        Args:
            folder_path: 文件夹路径
            batch_size: 批处理大小
            extensions: 支持的图片扩展名
        
        Returns:
            List[Dict]: 预测结果列表
        """
        folder_path = Path(folder_path)
        
        # 收集所有图片文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder_path.glob(f'*{ext}'))
            image_paths.extend(folder_path.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(list(set(image_paths)))  # 去重并排序
        
        if not image_paths:
            print(f"⚠️ 文件夹 {folder_path} 中没有找到图片")
            return []
        
        print(f"📁 找到 {len(image_paths)} 张图片")
        print(f"🚀 使用批大小 {batch_size} 在 {self.device} 上进行推理")
        
        if str(self.device) == 'cuda':
            # 显示GPU内存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 GPU内存: {allocated:.1f}/{total_memory:.1f} GB")
        
        # 批量推理
        start_time = time.time()
        results = self.predict_batch(image_paths, batch_size)
        total_time = time.time() - start_time
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        print(f"✅ 成功处理: {success_count}/{len(image_paths)} 张图片")
        print(f"⏱️ 总耗时: {total_time:.2f}秒 ({len(image_paths)/total_time:.1f} 张/秒)")
        
        if str(self.device) == 'cuda':
            # 显示最终GPU内存使用
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 最终GPU内存使用: {allocated:.1f} GB")
        
        return results
    
    def _preprocess_with_params(self, image: Union[str, Path, np.ndarray]) -> tuple:
        """图像预处理 - 返回tensor和变换参数（用于批处理）"""
        # 读取图像
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 保存原始尺寸
        original_size = image.shape[:2]  # (H, W)
        
        # 应用Letterbox变换
        image_letterboxed, transform_params = self.letterbox.apply(image)
        
        # 生成valid mask
        valid_mask = self.letterbox.create_padding_mask(transform_params)
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(image_letterboxed).float().permute(2, 0, 1) / 255.0
        valid_mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0)
        
        # 转换为灰度图
        gray_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        gray_tensor = gray_tensor.unsqueeze(0)
        
        # 组合输入
        image_tensor = torch.cat([gray_tensor, valid_mask_tensor], dim=0)
        
        # 添加batch维度并移到设备
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, transform_params, original_size
    
    def _preprocess(self, image: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """图像预处理 - 使用正确的Letterbox变换"""
        # 读取图像
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 保存原始尺寸用于后处理
        self.original_size = image.shape[:2]  # (H, W)
        
        # 应用Letterbox变换 - 等比缩放 + 居中填充
        image_letterboxed, self.transform_params = self.letterbox.apply(image)
        
        # 生成valid mask (valid区域=1, padding区域=0)
        valid_mask = self.letterbox.create_padding_mask(self.transform_params)
        
        # 转换为tensor并归一化 [0, 255] -> [0, 1]
        image_tensor = torch.from_numpy(image_letterboxed).float().permute(2, 0, 1) / 255.0
        valid_mask_tensor = torch.from_numpy(valid_mask).float().unsqueeze(0)
        
        # 转换为灰度图
        gray_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        gray_tensor = gray_tensor.unsqueeze(0)  # 添加通道维度
        
        # 组合输入 [Grayscale(1) + Valid_Mask(1)]  
        # 最后一个通道: 1=有效区域, 0=padding区域
        image_tensor = torch.cat([gray_tensor, valid_mask_tensor], dim=0)
        
        # 添加batch维度 [1, 2, 256, 512]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _format_batch_result(self, predictions: Dict, batch_idx: int, 
                            transform_params: Dict, original_size: tuple) -> Dict:
        """格式化批量推理的单个结果"""
        # 提取指定批次索引的结果
        gap_x_net = predictions['gap_coords'][batch_idx, 0].item()
        gap_y_net = predictions['gap_coords'][batch_idx, 1].item()
        slider_x_net = predictions['slider_coords'][batch_idx, 0].item()
        slider_y_net = predictions['slider_coords'][batch_idx, 1].item()
        
        # 映射回原始图像空间
        gap_x, gap_y = self.coord_transform.input_to_original(
            (gap_x_net, gap_y_net), transform_params
        )
        slider_x, slider_y = self.coord_transform.input_to_original(
            (slider_x_net, slider_y_net), transform_params
        )
        
        # 提取置信度
        gap_conf = predictions['gap_score'][batch_idx].item()
        slider_conf = predictions['slider_score'][batch_idx].item()
        
        return {
            'sliding_distance': gap_x - slider_x,
            'gap_x': gap_x,
            'gap_y': gap_y,
            'slider_x': slider_x,
            'slider_y': slider_y,
            'gap_confidence': gap_conf,
            'slider_confidence': slider_conf,
            'confidence': (gap_conf + slider_conf) / 2,
            'details': {
                'gap_coords': [gap_x, gap_y],
                'slider_coords': [slider_x, slider_y],
                'model_version': '1.1.0',
                'device_used': str(self.device),
                'original_size': original_size,
                'network_coords': {
                    'gap': [gap_x_net, gap_y_net],
                    'slider': [slider_x_net, slider_y_net]
                }
            }
        }
    
    def _format_result(self, predictions: Dict) -> Dict:
        """格式化输出结果 - 直接使用模型解码的坐标"""
        # 注意：模型已经在decode_predictions中完成了所有处理
        # 包括padding mask屏蔽和坐标clamp到[0,512]x[0,256]
        # 这里只需要提取并映射回原始图像空间
        
        # 提取网络输出的坐标（在512x256空间中）
        gap_x_net = predictions['gap_coords'][0, 0].item()
        gap_y_net = predictions['gap_coords'][0, 1].item()
        slider_x_net = predictions['slider_coords'][0, 0].item()
        slider_y_net = predictions['slider_coords'][0, 1].item()
        
        # 将坐标从网络空间映射回原始图像空间
        gap_x, gap_y = self.coord_transform.input_to_original(
            (gap_x_net, gap_y_net), self.transform_params
        )
        slider_x, slider_y = self.coord_transform.input_to_original(
            (slider_x_net, slider_y_net), self.transform_params
        )
        
        # 提取置信度
        gap_conf = predictions['gap_score'][0].item()
        slider_conf = predictions['slider_score'][0].item()
        
        return {
            'sliding_distance': gap_x - slider_x,
            'gap_x': gap_x,
            'gap_y': gap_y,
            'slider_x': slider_x,
            'slider_y': slider_y,
            'gap_confidence': gap_conf,
            'slider_confidence': slider_conf,
            'confidence': (gap_conf + slider_conf) / 2,
            'details': {
                'gap_coords': [gap_x, gap_y],
                'slider_coords': [slider_x, slider_y],
                'model_version': '1.1.0',
                'device_used': str(self.device),
                'original_size': self.original_size,
                'network_coords': {
                    'gap': [gap_x_net, gap_y_net],
                    'slider': [slider_x_net, slider_y_net]
                }
            }
        }
    
    def visualize_prediction(self, 
                           image_path: Union[str, Path],
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
        """可视化预测结果
        
        Args:
            image_path: 输入图片路径
            save_path: 保存路径（可选）
            show: 是否显示图片
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # 获取预测结果
        result = self.predict(image_path)
        if not result['success']:
            print(f"预测失败: {result.get('error', 'Unknown error')}")
            return
        
        # 读取原图
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(image)
        
        # 绘制缺口位置（红色）
        gap_rect = patches.Circle((result['gap_x'], result['gap_y']), 
                                 20, linewidth=2, edgecolor='red', 
                                 facecolor='none', label='Gap')
        ax.add_patch(gap_rect)
        
        # 绘制滑块位置（绿色）
        slider_rect = patches.Circle((result['slider_x'], result['slider_y']), 
                                    20, linewidth=2, edgecolor='green', 
                                    facecolor='none', label='Slider')
        ax.add_patch(slider_rect)
        
        # 绘制滑动距离线（蓝色）
        ax.arrow(result['slider_x'], result['slider_y'],
                result['sliding_distance'], 0,
                head_width=10, head_length=10, fc='blue', ec='blue',
                alpha=0.7, label=f"Distance: {result['sliding_distance']:.1f}px")
        
        # 添加标题和图例
        ax.set_title(f"滑动距离: {result['sliding_distance']:.1f}px | "
                    f"置信度: {result['confidence']:.3f}")
        ax.legend()
        ax.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ 可视化已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_heatmaps(self,
                         image_path: Union[str, Path],
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
        """可视化热力图
        
        Args:
            image_path: 输入图片路径
            save_path: 保存路径（可选）
            show: 是否显示图片
        """
        import matplotlib.pyplot as plt
        
        # 预处理图像
        image_tensor = self._preprocess(image_path)
        
        # 获取热力图
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 读取原图
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # 提取热力图
        gap_heatmap = outputs['heatmap_gap'][0, 0].cpu().numpy()
        slider_heatmap = outputs['heatmap_slider'][0, 0].cpu().numpy()
        
        # 上采样热力图到原图尺寸
        h, w = image.shape[:2]
        gap_heatmap = cv2.resize(gap_heatmap, (w, h))
        slider_heatmap = cv2.resize(slider_heatmap, (w, h))
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(image)
        axes[0].set_title("原图")
        axes[0].axis('off')
        
        # 缺口热力图
        im1 = axes[1].imshow(gap_heatmap, cmap='hot', alpha=0.8)
        axes[1].imshow(image, alpha=0.3)
        axes[1].set_title("缺口热力图")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # 滑块热力图
        im2 = axes[2].imshow(slider_heatmap, cmap='hot', alpha=0.8)
        axes[2].imshow(image, alpha=0.3)
        axes[2].set_title("滑块热力图")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ 热力图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()