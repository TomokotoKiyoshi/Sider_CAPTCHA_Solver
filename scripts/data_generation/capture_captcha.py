"""
验证码抓取脚本
支持点击指定坐标和截取指定区域的验证码图片
"""

import os
import time
import json
from datetime import datetime
from typing import Tuple, Optional
import pyautogui
import keyboard
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import messagebox
import threading

class CaptchaCapturer:
    def __init__(self, save_dir: str = "data/real_captchas/merged/site2"):
        """
        初始化验证码抓取器
        
        Args:
            save_dir: 保存截图的目录
        """
        self.save_dir = save_dir
        self.click_coord = None
        self.capture_region = None
        self.is_running = False
        self.capture_count = 0
        self.wait_after_click = 2.0  # 默认等待时间
        self.auto_retry_on_black = True  # 黑屏时自动重试
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 禁用pyautogui的安全特性
        pyautogui.FAILSAFE = False
        
    def set_click_coordinate(self):
        """设置点击坐标（空格键确认）"""
        print("\n" + "="*50)
        print("设置点击坐标")
        print("="*50)
        print("请将鼠标移动到需要点击的位置")
        print("按下【空格键】确认坐标...")
        print("按 ESC 取消设置")
        
        while True:
            if keyboard.is_pressed('esc'):
                print("已取消设置")
                return False
                
            # 实时显示鼠标位置
            x, y = pyautogui.position()
            print(f"\r当前鼠标位置: ({x}, {y})", end="")
            
            # 检测空格键
            if keyboard.is_pressed('space'):
                self.click_coord = (x, y)
                print(f"\n✓ 点击坐标已设置: {self.click_coord}")
                time.sleep(0.5)  # 防止重复检测
                return True
                
            time.sleep(0.01)
    
    def set_capture_region(self):
        """设置截图区域（空格键确认两个角点）"""
        print("\n" + "="*50)
        print("设置截图区域")
        print("="*50)
        print("请将鼠标移动到截图区域的【左上角】")
        print("按下【空格键】确认第一个点...")
        print("按 ESC 取消设置")
        
        start_pos = None
        
        while True:
            if keyboard.is_pressed('esc'):
                print("已取消设置")
                return False
            
            x, y = pyautogui.position()
            
            if start_pos is None:
                # 等待第一个点
                print(f"\r左上角位置: ({x}, {y})", end="")
                
                if keyboard.is_pressed('space'):
                    start_pos = (x, y)
                    print(f"\n✓ 左上角已设置: {start_pos}")
                    print("\n请将鼠标移动到截图区域的【右下角】")
                    print("按下【空格键】确认第二个点...")
                    time.sleep(0.5)  # 防止重复检测
            else:
                # 等待第二个点
                width = abs(x - start_pos[0])
                height = abs(y - start_pos[1])
                print(f"\r右下角位置: ({x}, {y}) | 区域大小: {width}×{height}", end="")
                
                if keyboard.is_pressed('space'):
                    end_pos = (x, y)
                    
                    # 计算区域
                    left = min(start_pos[0], end_pos[0])
                    top = min(start_pos[1], end_pos[1])
                    width = abs(end_pos[0] - start_pos[0])
                    height = abs(end_pos[1] - start_pos[1])
                    
                    if width > 10 and height > 10:  # 最小区域限制
                        self.capture_region = (left, top, width, height)
                        print(f"\n✓ 截图区域已设置: 左上角({left}, {top}), 宽度:{width}, 高度:{height}")
                        
                        # 显示预览
                        self._show_region_preview()
                        time.sleep(0.5)
                        return True
                    else:
                        print("\n选择的区域太小，请重新选择")
                        start_pos = None
                        print("\n请将鼠标移动到截图区域的【左上角】")
                        print("按下【空格键】确认第一个点...")
                        time.sleep(0.5)
            
            time.sleep(0.01)
    
    def _show_region_preview(self):
        """显示选中区域的预览"""
        if not self.capture_region:
            return
            
        try:
            # 截取预览图
            screenshot = pyautogui.screenshot(region=self.capture_region)
            
            # 在图片上添加边框
            draw = ImageDraw.Draw(screenshot)
            draw.rectangle(
                [(0, 0), (screenshot.width-1, screenshot.height-1)],
                outline='red',
                width=2
            )
            
            # 保存预览图
            preview_path = os.path.join(self.save_dir, "_preview.png")
            screenshot.save(preview_path)
            print(f"预览图已保存: {preview_path}")
            
        except Exception as e:
            print(f"生成预览时出错: {e}")
    
    def capture_single(self, wait_time: float = None):
        """执行单次抓取"""
        if not self.click_coord or not self.capture_region:
            print("错误: 请先设置点击坐标和截图区域")
            return False
        
        if wait_time is None:
            wait_time = self.wait_after_click
        
        try:
            # 1. 点击指定坐标
            print(f"点击坐标: {self.click_coord}")
            pyautogui.click(self.click_coord[0], self.click_coord[1])
            
            # 2. 等待页面加载（可调整）
            print(f"等待 {wait_time} 秒...")
            time.sleep(wait_time)
            
            # 3. 截取指定区域
            screenshot = pyautogui.screenshot(region=self.capture_region)
            
            # 4. 检查是否为黑屏（所有像素都是黑色或接近黑色）
            # 转换为灰度图检查
            gray_img = screenshot.convert('L')
            pixels = list(gray_img.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            
            if avg_brightness < 10:  # 平均亮度小于10认为是黑屏
                print(f"⚠️ 检测到黑屏（平均亮度: {avg_brightness:.1f}），可能需要增加等待时间")
                # 可以选择重试
                if self.auto_retry_on_black:
                    print("自动重试中...")
                    time.sleep(2)  # 额外等待
                    screenshot = pyautogui.screenshot(region=self.capture_region)
                    gray_img = screenshot.convert('L')
                    pixels = list(gray_img.getdata())
                    avg_brightness = sum(pixels) / len(pixels)
                    print(f"重试后平均亮度: {avg_brightness:.1f}")
            
            # 5. 生成文件名并保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captcha_{timestamp}_{self.capture_count:04d}.png"
            filepath = os.path.join(self.save_dir, filename)
            
            screenshot.save(filepath)
            self.capture_count += 1
            
            print(f"✓ 已保存: {filename} (亮度: {avg_brightness:.1f})")
            return True
            
        except Exception as e:
            print(f"抓取失败: {e}")
            return False
    
    def run_continuous(self, interval: float = 3.0):
        """连续运行抓取"""
        print("\n" + "="*50)
        print("开始连续抓取模式")
        print(f"抓取间隔: {interval}秒")
        print("按 'q' 停止抓取")
        print("="*50)
        
        self.is_running = True
        
        while self.is_running:
            if keyboard.is_pressed('q'):
                print("\n停止抓取")
                self.is_running = False
                break
            
            # 执行抓取
            if self.capture_single():
                print(f"等待 {interval} 秒后继续...")
                time.sleep(interval)
            else:
                print("抓取失败，停止运行")
                break
        
        print(f"\n总共抓取了 {self.capture_count} 个验证码")
    
    def save_config(self):
        """保存配置到文件"""
        config = {
            'click_coord': self.click_coord,
            'capture_region': self.capture_region,
            'save_dir': self.save_dir
        }
        
        config_path = os.path.join(self.save_dir, "_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"配置已保存: {config_path}")
    
    def load_config(self, config_path: str = None):
        """从文件加载配置"""
        if config_path is None:
            config_path = os.path.join(self.save_dir, "_config.json")
        
        if not os.path.exists(config_path):
            print("配置文件不存在")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.click_coord = tuple(config['click_coord']) if config['click_coord'] else None
            self.capture_region = tuple(config['capture_region']) if config['capture_region'] else None
            
            print("配置已加载:")
            print(f"  点击坐标: {self.click_coord}")
            print(f"  截图区域: {self.capture_region}")
            return True
            
        except Exception as e:
            print(f"加载配置失败: {e}")
            return False

def main():
    """主函数 - 交互式界面"""
    capturer = CaptchaCapturer()
    
    while True:
        print("\n" + "="*60)
        print("验证码抓取工具")
        print("="*60)
        print("1. 设置点击坐标（空格键确认）")
        print("2. 设置截图区域（空格键确认两个角点）") 
        print("3. 执行单次抓取")
        print("4. 开始连续抓取")
        print("5. 保存当前配置")
        print("6. 加载之前的配置")
        print("7. 查看当前设置")
        print("8. 调整等待时间（当前: {:.1f}秒）".format(capturer.wait_after_click))
        print("9. 测试截图（不点击，仅截图）")
        print("0. 退出")
        print("="*60)
        
        choice = input("请选择操作 (0-9): ").strip()
        
        if choice == '1':
            capturer.set_click_coordinate()
            
        elif choice == '2':
            capturer.set_capture_region()
            
        elif choice == '3':
            if capturer.click_coord and capturer.capture_region:
                capturer.capture_single()
            else:
                print("请先完成设置（选项1和2）")
                
        elif choice == '4':
            if capturer.click_coord and capturer.capture_region:
                try:
                    interval = float(input("请输入抓取间隔（秒，默认3）: ") or "3")
                    capturer.run_continuous(interval)
                except ValueError:
                    print("无效的输入")
            else:
                print("请先完成设置（选项1和2）")
                
        elif choice == '5':
            if capturer.click_coord and capturer.capture_region:
                capturer.save_config()
            else:
                print("没有可保存的配置")
                
        elif choice == '6':
            capturer.load_config()
            
        elif choice == '7':
            print("\n当前设置:")
            print(f"  点击坐标: {capturer.click_coord or '未设置'}")
            print(f"  截图区域: {capturer.capture_region or '未设置'}")
            print(f"  保存目录: {capturer.save_dir}")
            print(f"  已抓取数量: {capturer.capture_count}")
            print(f"  等待时间: {capturer.wait_after_click}秒")
            print(f"  黑屏自动重试: {'开启' if capturer.auto_retry_on_black else '关闭'}")
            
        elif choice == '8':
            try:
                new_wait = float(input(f"请输入新的等待时间（秒，当前{capturer.wait_after_click}）: "))
                if 0 < new_wait <= 10:
                    capturer.wait_after_click = new_wait
                    print(f"✓ 等待时间已设置为: {new_wait}秒")
                else:
                    print("等待时间应在0-10秒之间")
            except ValueError:
                print("无效的输入")
        
        elif choice == '9':
            # 测试截图功能
            if capturer.capture_region:
                try:
                    print("正在截图...")
                    screenshot = pyautogui.screenshot(region=capturer.capture_region)
                    
                    # 检查亮度
                    gray_img = screenshot.convert('L')
                    pixels = list(gray_img.getdata())
                    avg_brightness = sum(pixels) / len(pixels)
                    
                    # 保存测试图
                    test_path = os.path.join(capturer.save_dir, "_test.png")
                    screenshot.save(test_path)
                    
                    print(f"✓ 测试截图已保存: {test_path}")
                    print(f"  图像尺寸: {screenshot.width}×{screenshot.height}")
                    print(f"  平均亮度: {avg_brightness:.1f}")
                    
                    if avg_brightness < 10:
                        print("  ⚠️ 警告: 图像可能是黑屏！")
                    elif avg_brightness < 50:
                        print("  ⚠️ 提示: 图像较暗")
                    else:
                        print("  ✓ 图像亮度正常")
                        
                except Exception as e:
                    print(f"测试失败: {e}")
            else:
                print("请先设置截图区域（选项2）")
            
        elif choice == '0':
            print("退出程序")
            break
            
        else:
            print("无效的选择")

if __name__ == "__main__":
    print("欢迎使用验证码抓取工具！")
    print("提示: 请确保目标网页已经打开并可见")
    main()