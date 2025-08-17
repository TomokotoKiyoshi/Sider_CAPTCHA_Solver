"""
滑块验证码识别GUI应用
支持图片浏览、模型推理、结果可视化
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
from pathlib import Path
import threading
import time
from typing import Optional, List, Dict, Tuple
import sys
import json
from dataclasses import dataclass
from queue import Queue
import logging

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入预测器
from sider_captcha_solver.predictor import CaptchaPredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据类"""
    success: bool
    sliding_distance: float = 0
    gap_x: float = 0
    gap_y: float = 0
    slider_x: float = 0
    slider_y: float = 0
    confidence: float = 0
    processing_time_ms: float = 0
    error: str = ""


class ImageProcessor:
    """图像预处理器 - 遵循Step0_Preprocessing_Guide.md"""
    
    @staticmethod
    def letterbox(image: np.ndarray, target_size: Tuple[int, int] = (512, 256), 
                  fill_value: int = 255) -> Tuple[np.ndarray, Dict]:
        """
        Letterbox变换：等比缩放+居中填充
        
        Args:
            image: 输入图像 (H, W, 3)
            target_size: 目标尺寸 (W, H) = (512, 256)
            fill_value: 填充值（255为白色）
        
        Returns:
            处理后的图像和变换参数
        """
        H_in, W_in = image.shape[:2]
        W_t, H_t = target_size
        
        # 计算缩放比例
        scale = min(W_t / W_in, H_t / H_in)
        
        # 缩放后尺寸
        W_prime = int(W_in * scale)
        H_prime = int(H_in * scale)
        
        # 等比缩放
        image_resized = cv2.resize(image, (W_prime, H_prime), interpolation=cv2.INTER_LINEAR)
        
        # 计算padding
        pad_x = W_t - W_prime
        pad_y = H_t - H_prime
        
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        
        # 创建画布并居中放置
        canvas = np.full((H_t, W_t, 3), fill_value, dtype=np.uint8)
        canvas[pad_top:pad_top+H_prime, pad_left:pad_left+W_prime] = image_resized
        
        # 变换参数
        transform_params = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'resized_size': (W_prime, H_prime),
            'original_size': (W_in, H_in)
        }
        
        return canvas, transform_params


class CaptchaGUI:
    """滑块验证码识别GUI主类"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🔍 滑块验证码智能识别系统 v1.1.0")
        self.root.geometry("1400x800")
        
        # 设置应用图标和样式
        self.setup_styles()
        
        # 初始化变量
        self.current_dir = None
        self.image_files: List[Path] = []
        self.current_index = 0
        self.current_image = None
        self.current_result: Optional[PredictionResult] = None
        self.predictor: Optional[CaptchaPredictor] = None
        self.processing = False
        self.cache: Dict[int, PredictionResult] = {}
        self.current_model_path = None
        self.default_model_dir = Path("D:/Hacker/Sider_CAPTCHA_Solver/src/checkpoints/1.1.0")
        
        # 创建UI组件
        self.create_widgets()
        self.bind_shortcuts()
        
        # 启动时提示
        self.show_welcome()
        
        # 自动加载默认模型目录中的第一个模型
        self.auto_load_default_model()
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义颜色
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#007acc',
            'success': '#4caf50',
            'error': '#f44336',
            'warning': '#ff9800',
            'gap': '#ff4444',
            'slider': '#44ff44',
            'path': '#4444ff'
        }
        
        # 配置样式
        style.configure('Title.TLabel', font=('Microsoft YaHei', 14, 'bold'))
        style.configure('Info.TLabel', font=('Consolas', 10))
        style.configure('Status.TLabel', font=('Microsoft YaHei', 10))
    
    def create_widgets(self):
        """创建UI组件"""
        # 顶部工具栏
        self.create_toolbar()
        
        # 主内容区
        self.create_main_area()
        
        # 底部状态栏
        self.create_statusbar()
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 目录选择按钮
        ttk.Button(
            toolbar, 
            text="📁 选择captchas目录",
            command=lambda: self.load_directory('data/captchas')
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            toolbar,
            text="📁 选择real_captchas目录", 
            command=lambda: self.load_directory('data/real_captchas/annotated')
        ).pack(side=tk.LEFT, padx=5)
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 模型选择按钮
        ttk.Button(
            toolbar,
            text="🤖 选择模型",
            command=self.select_model
        ).pack(side=tk.LEFT, padx=5)
        
        # 当前模型标签
        self.current_model_label = ttk.Label(toolbar, text="模型: 未选择", style='Info.TLabel')
        self.current_model_label.pack(side=tk.LEFT, padx=5)
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 当前图片信息
        self.current_file_label = ttk.Label(toolbar, text="未加载图片", style='Info.TLabel')
        self.current_file_label.pack(side=tk.LEFT, padx=10)
        
        # 模型状态
        self.model_status_label = ttk.Label(toolbar, text="⏳ 请选择模型", style='Status.TLabel')
        self.model_status_label.pack(side=tk.RIGHT, padx=10)
    
    def create_main_area(self):
        """创建主显示区域"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：图片列表
        self.create_image_list(main_frame)
        
        # 中间：图片显示
        self.create_image_display(main_frame)
        
        # 右侧：结果面板
        self.create_result_panel(main_frame)
    
    def create_image_list(self, parent):
        """创建图片列表"""
        list_frame = ttk.LabelFrame(parent, text="图片列表", width=200)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 列表框和滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg='#2b2b2b',
            fg='white',
            selectbackground='#007acc',
            font=('Consolas', 9),
            width=25
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        
        # 绑定选择事件
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
    
    def create_image_display(self, parent):
        """创建图片显示区域"""
        display_frame = ttk.LabelFrame(parent, text="图像预览")
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas用于显示图片
        self.canvas = tk.Canvas(display_frame, bg='#2b2b2b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件（用于缩放等）
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
    
    def create_result_panel(self, parent):
        """创建结果面板"""
        result_frame = ttk.LabelFrame(parent, text="预测结果", width=300)
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        result_frame.pack_propagate(False)
        
        # 结果文本框
        self.result_text = tk.Text(
            result_frame,
            bg='#2b2b2b',
            fg='#00ff00',
            font=('Consolas', 11),
            wrap=tk.WORD,
            width=35,
            height=30
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 操作按钮
        button_frame = ttk.Frame(result_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.predict_button = ttk.Button(
            button_frame,
            text="🚀 开始推理 (Enter)",
            command=self.predict_current,
            state=tk.DISABLED
        )
        self.predict_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            button_frame,
            text="💾 导出结果",
            command=self.export_results
        ).pack(fill=tk.X, pady=2)
    
    def create_statusbar(self):
        """创建状态栏"""
        statusbar = ttk.Frame(self.root)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 进度条
        self.progress = ttk.Progressbar(
            statusbar,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side=tk.LEFT, padx=10)
        
        # 状态信息
        self.status_label = ttk.Label(
            statusbar,
            text="就绪 | 使用方向键切换图片，Enter键推理",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # GPU/CPU指示器
        self.device_label = ttk.Label(
            statusbar,
            text="设备: 检测中...",
            style='Status.TLabel'
        )
        self.device_label.pack(side=tk.RIGHT, padx=10)
    
    def bind_shortcuts(self):
        """绑定键盘快捷键"""
        self.root.bind('<Left>', lambda e: self.navigate_image(-1))
        self.root.bind('<Right>', lambda e: self.navigate_image(1))
        self.root.bind('<Return>', lambda e: self.predict_current())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('<Control-o>', lambda e: self.open_custom_directory())
        self.root.bind('<Control-m>', lambda e: self.select_model())
        self.root.bind('<F5>', lambda e: self.refresh_current())
        self.root.bind('<space>', lambda e: self.toggle_visualization())
    
    def select_model(self):
        """打开模型选择对话框"""
        # 设置初始目录
        initial_dir = self.default_model_dir if self.default_model_dir.exists() else Path.cwd()
        
        model_path = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=initial_dir,
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        
        if model_path:
            self.load_model(model_path)
    
    def auto_load_default_model(self):
        """自动加载默认目录中的模型"""
        if self.default_model_dir.exists():
            # 查找目录中的pth文件
            pth_files = list(self.default_model_dir.glob("*.pth"))
            if pth_files:
                # 优先加载best_model.pth，否则加载第一个
                best_model = self.default_model_dir / "best_model.pth"
                if best_model.exists():
                    self.load_model(str(best_model))
                else:
                    self.load_model(str(pth_files[0]))
    
    def load_model(self, model_path: str):
        """加载指定的模型文件"""
        self.current_model_path = Path(model_path)
        self.model_status_label.config(text="⏳ 正在加载模型...")
        self.predict_button.config(state=tk.DISABLED)
        
        # 清空缓存（因为换了模型）
        self.cache.clear()
        self.update_image_list()
        
        # 异步加载模型
        def load():
            try:
                self.predictor = CaptchaPredictor(model_path=model_path, device='auto')
                device = str(self.predictor.device)
                model_name = self.current_model_path.name
                self.root.after(0, lambda: self.on_model_loaded(device, model_name))
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def init_model_async(self):
        """异步初始化模型（保留兼容性）"""
        if self.current_model_path:
            self.load_model(str(self.current_model_path))
    
    def on_model_loaded(self, device: str, model_name: str = "默认模型"):
        """模型加载完成回调"""
        self.model_status_label.config(text=f"✅ 模型就绪")
        self.current_model_label.config(text=f"模型: {model_name}")
        self.device_label.config(text=f"设备: {device.upper()}")
        self.predict_button.config(state=tk.NORMAL)
        self.update_status(f"模型 {model_name} 加载成功，可以开始推理")
    
    def on_model_error(self, error: str):
        """模型加载错误回调"""
        self.model_status_label.config(text="❌ 模型加载失败")
        messagebox.showerror("模型加载失败", f"无法加载模型：{error}")
    
    def load_directory(self, default_path: str):
        """加载图片目录"""
        # 构建完整路径
        full_path = Path(self.root.winfo_toplevel().winfo_pathname(self.root.winfo_id())).parent / default_path
        
        if not full_path.exists():
            # 如果默认路径不存在，打开文件对话框
            directory = filedialog.askdirectory(title="选择图片目录")
            if not directory:
                return
            full_path = Path(directory)
        
        self.current_dir = full_path
        
        # 查找所有图片文件
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        self.image_files = sorted([
            f for f in full_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not self.image_files:
            messagebox.showwarning("警告", "所选目录中没有找到图片文件")
            return
        
        # 更新列表
        self.update_image_list()
        
        # 加载第一张图片
        self.current_index = 0
        self.load_image(0)
        
        self.update_status(f"已加载 {len(self.image_files)} 张图片")
    
    def update_image_list(self):
        """更新图片列表显示"""
        self.image_listbox.delete(0, tk.END)
        
        for i, file in enumerate(self.image_files):
            # 显示文件名和缓存状态
            status = "✅" if i in self.cache else "  "
            self.image_listbox.insert(tk.END, f"{status} {file.name}")
        
        if self.image_files:
            self.image_listbox.selection_set(0)
    
    def on_image_select(self, event):
        """列表选择事件处理"""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_image(index)
    
    def load_image(self, index: int):
        """加载并显示图片"""
        if not 0 <= index < len(self.image_files):
            return
        
        self.current_index = index
        image_path = self.image_files[index]
        
        # 更新列表选择
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.see(index)
        
        # 读取图片
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("无法读取图片")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image
            
            # 显示图片
            self.display_image(image)
            
            # 更新信息
            self.current_file_label.config(
                text=f"[{index+1}/{len(self.image_files)}] {image_path.name}"
            )
            
            # 如果有缓存的结果，显示它
            if index in self.cache:
                self.display_result(self.cache[index])
            else:
                self.clear_result()
                
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{e}")
    
    def display_image(self, image: np.ndarray, with_annotations: bool = False):
        """在Canvas上显示图片"""
        # 获取Canvas尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, lambda: self.display_image(image, with_annotations))
            return
        
        # 计算缩放比例以适应Canvas
        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 如果需要标注且有结果，添加标注
        if with_annotations and self.current_result and self.current_result.success:
            image_resized = self.add_annotations(image_resized, scale)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清除Canvas并显示新图片
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)
    
    def add_annotations(self, image: np.ndarray, scale: float) -> np.ndarray:
        """在图片上添加预测标注"""
        if not self.current_result or not self.current_result.success:
            return image
        
        result = self.current_result
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        # 缩放坐标
        gap_x = int(result.gap_x * scale)
        gap_y = int(result.gap_y * scale)
        slider_x = int(result.slider_x * scale)
        slider_y = int(result.slider_y * scale)
        
        # 绘制缺口（红色圆圈）
        radius = 20
        draw.ellipse(
            [gap_x - radius, gap_y - radius, gap_x + radius, gap_y + radius],
            outline='red', width=3
        )
        draw.text((gap_x + radius + 5, gap_y - 10), "Gap", fill='red')
        
        # 绘制滑块（绿色圆圈）
        draw.ellipse(
            [slider_x - radius, slider_y - radius, slider_x + radius, slider_y + radius],
            outline='lime', width=3
        )
        draw.text((slider_x + radius + 5, slider_y - 10), "Slider", fill='lime')
        
        # 绘制滑动路径（蓝色箭头）
        draw.line([slider_x, slider_y, gap_x, slider_y], fill='blue', width=2)
        # 箭头头部
        arrow_size = 10
        draw.polygon([
            (gap_x, slider_y),
            (gap_x - arrow_size, slider_y - arrow_size//2),
            (gap_x - arrow_size, slider_y + arrow_size//2)
        ], fill='blue')
        
        # 显示滑动距离
        distance_text = f"{result.sliding_distance:.1f}px"
        text_x = (slider_x + gap_x) // 2
        text_y = slider_y - 30
        draw.text((text_x, text_y), distance_text, fill='yellow')
        
        return np.array(img_pil)
    
    def navigate_image(self, direction: int):
        """导航到上一张/下一张图片"""
        if not self.image_files:
            return
        
        new_index = (self.current_index + direction) % len(self.image_files)
        self.load_image(new_index)
    
    def predict_current(self):
        """对当前图片进行推理"""
        if not self.predictor or not self.current_image is not None or self.processing:
            return
        
        if self.current_index in self.cache:
            # 如果已有缓存，直接显示
            self.display_result(self.cache[self.current_index])
            return
        
        self.processing = True
        self.progress.start(10)
        self.update_status("正在推理...")
        self.predict_button.config(state=tk.DISABLED)
        
        def predict():
            try:
                # 执行推理
                image_path = self.image_files[self.current_index]
                result_dict = self.predictor.predict(str(image_path))
                
                # 转换为结果对象
                if result_dict['success']:
                    result = PredictionResult(
                        success=True,
                        sliding_distance=result_dict['sliding_distance'],
                        gap_x=result_dict['gap_x'],
                        gap_y=result_dict['gap_y'],
                        slider_x=result_dict['slider_x'],
                        slider_y=result_dict['slider_y'],
                        confidence=result_dict['confidence'],
                        processing_time_ms=result_dict['processing_time_ms']
                    )
                else:
                    result = PredictionResult(
                        success=False,
                        error=result_dict.get('error', 'Unknown error')
                    )
                
                self.root.after(0, lambda: self.on_prediction_complete(result))
                
            except Exception as e:
                result = PredictionResult(success=False, error=str(e))
                self.root.after(0, lambda: self.on_prediction_complete(result))
        
        thread = threading.Thread(target=predict, daemon=True)
        thread.start()
    
    def on_prediction_complete(self, result: PredictionResult):
        """推理完成回调"""
        self.processing = False
        self.progress.stop()
        self.predict_button.config(state=tk.NORMAL)
        
        # 缓存结果
        self.cache[self.current_index] = result
        self.current_result = result
        
        # 显示结果
        self.display_result(result)
        
        # 更新图片列表的状态标记
        self.update_image_list()
        
        # 重新显示图片（带标注）
        self.display_image(self.current_image, with_annotations=True)
        
        if result.success:
            self.update_status(f"推理完成 | 耗时: {result.processing_time_ms:.1f}ms")
        else:
            self.update_status(f"推理失败: {result.error}")
    
    def display_result(self, result: PredictionResult):
        """显示预测结果"""
        self.current_result = result
        self.result_text.delete(1.0, tk.END)
        
        if result.success:
            text = f"""
╔════════════════════════════════╗
║      🎯 预测结果               ║
╚════════════════════════════════╝

📏 滑动距离: {result.sliding_distance:.2f} px

📍 缺口位置:
   X: {result.gap_x:.2f}
   Y: {result.gap_y:.2f}

📍 滑块位置:
   X: {result.slider_x:.2f}  
   Y: {result.slider_y:.2f}

📊 置信度: {result.confidence:.3f}
⏱️ 处理时间: {result.processing_time_ms:.1f} ms

═══════════════════════════════════
            """
            self.result_text.insert(1.0, text)
            
            # 高亮显示重要数值
            self.result_text.tag_add("distance", "4.14", "4.21")
            self.result_text.tag_config("distance", foreground="#00ff00", font=('Consolas', 12, 'bold'))
        else:
            text = f"""
╔════════════════════════════════╗
║      ❌ 预测失败               ║
╚════════════════════════════════╝

错误信息:
{result.error}
            """
            self.result_text.insert(1.0, text)
            self.result_text.tag_add("error", "5.0", "end")
            self.result_text.tag_config("error", foreground="#ff4444")
    
    def clear_result(self):
        """清除结果显示"""
        self.current_result = None
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "等待推理...\n\n按 Enter 键开始")
    
    def toggle_visualization(self):
        """切换可视化显示"""
        if self.current_image is not None:
            show_annotations = not (self.current_result is not None)
            self.display_image(self.current_image, with_annotations=show_annotations)
    
    def refresh_current(self):
        """刷新当前显示"""
        if self.current_index in self.cache:
            del self.cache[self.current_index]
        self.load_image(self.current_index)
    
    def export_results(self):
        """导出预测结果"""
        if not self.cache:
            messagebox.showinfo("提示", "没有可导出的结果")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            results = {}
            for idx, result in self.cache.items():
                if result.success:
                    results[self.image_files[idx].name] = {
                        'sliding_distance': result.sliding_distance,
                        'gap_x': result.gap_x,
                        'gap_y': result.gap_y,
                        'slider_x': result.slider_x,
                        'slider_y': result.slider_y,
                        'confidence': result.confidence
                    }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("成功", f"结果已导出到：{filename}")
    
    def open_custom_directory(self):
        """打开自定义目录"""
        directory = filedialog.askdirectory(title="选择图片目录")
        if directory:
            self.load_directory(directory)
    
    def on_mouse_wheel(self, event):
        """鼠标滚轮缩放（预留功能）"""
        # 可以实现图片缩放功能
        pass
    
    def update_status(self, message: str):
        """更新状态栏信息"""
        self.status_label.config(text=message)
    
    def show_welcome(self):
        """显示欢迎信息"""
        welcome_text = """
╔══════════════════════════════════════╗
║   🔍 滑块验证码智能识别系统          ║
║      Powered by Lite-HRNet-18        ║
╚══════════════════════════════════════╝

📌 快捷键说明:
  • ← / → : 切换上一张/下一张图片
  • Enter : 执行模型推理
  • Space : 切换标注显示
  • F5 : 刷新当前图片
  • Ctrl+O : 打开自定义目录
  • Ctrl+M : 选择模型文件
  • Esc : 退出程序

📂 请选择图片目录开始使用...
        """
        self.result_text.insert(1.0, welcome_text)


def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置DPI感知（Windows）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = CaptchaGUI(root)
    
    # 居中显示窗口
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()