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
from pynput import keyboard
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import messagebox
import threading
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll
import numpy as np

class CaptchaCapturer:
    def __init__(self, save_dir: str = None):
        """
        初始化验证码抓取器
        
        Args:
            save_dir: 保存截图的目录
        """
        self.save_dir = save_dir if save_dir else "data/real_captchas/merged/site1"
        self.click_coord = None
        self.capture_region = None
        self.is_running = False
        self.capture_count = 0
        self.wait_after_click = 2.0  # 默认等待时间
        self.auto_retry_on_black = True  # 黑屏时自动重试
        self.background_mode = False  # 后台模式
        self.target_window = None  # 目标窗口句柄
        
        # 键盘监听器相关
        self.keyboard_listener = None
        self.space_pressed = False
        self.esc_pressed = False
        self.q_pressed = False
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 禁用pyautogui的安全特性
        pyautogui.FAILSAFE = False
    
    def on_press(self, key):
        """键盘按下事件处理"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True
            elif key == keyboard.Key.esc:
                self.esc_pressed = True
            elif hasattr(key, 'char') and key.char == 'q':
                self.q_pressed = True
        except:
            pass
    
    def on_release(self, key):
        """键盘释放事件处理"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = False
            elif key == keyboard.Key.esc:
                self.esc_pressed = False
            elif hasattr(key, 'char') and key.char == 'q':
                self.q_pressed = False
        except:
            pass
    
    def start_keyboard_listener(self):
        """启动键盘监听器"""
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.keyboard_listener.start()
    
    def stop_keyboard_listener(self):
        """停止键盘监听器"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
    def set_click_coordinate(self):
        """设置点击坐标（空格键确认）"""
        print("\n" + "="*50)
        print("设置点击坐标")
        print("="*50)
        print("请将鼠标移动到需要点击的位置")
        print("按下【空格键】确认坐标...")
        print("按 ESC 取消设置")
        print("\n等待按键...")  # 不再实时显示坐标
        
        # 启动键盘监听
        self.start_keyboard_listener()
        self.space_pressed = False
        self.esc_pressed = False
        
        try:
            while True:
                if self.esc_pressed:
                    print("已取消设置")
                    return False
                
                # 检测空格键
                if self.space_pressed:
                    # 只在按下空格时获取鼠标位置
                    x, y = pyautogui.position()
                    self.click_coord = (x, y)
                    print(f"✓ 点击坐标已设置: {self.click_coord}")
                    time.sleep(0.5)  # 防止重复检测
                    return True
                    
                time.sleep(0.05)  # 降低CPU占用
        finally:
            self.stop_keyboard_listener()
    
    def set_capture_region(self):
        """设置截图区域（空格键确认两个角点）"""
        print("\n" + "="*50)
        print("设置截图区域")
        print("="*50)
        print("请将鼠标移动到截图区域的【左上角】")
        print("按下【空格键】确认第一个点...")
        print("按 ESC 取消设置")
        print("\n等待第一个点...")  # 不再实时显示
        
        # 启动键盘监听
        self.start_keyboard_listener()
        self.space_pressed = False
        self.esc_pressed = False
        
        start_pos = None
        
        try:
            while True:
                if self.esc_pressed:
                    print("已取消设置")
                    return False
                
                if start_pos is None:
                    # 等待第一个点
                    if self.space_pressed:
                        x, y = pyautogui.position()  # 只在按键时获取
                        start_pos = (x, y)
                        print(f"✓ 左上角已设置: {start_pos}")
                        print("\n请将鼠标移动到截图区域的【右下角】")
                        print("按下【空格键】确认第二个点...")
                        print("\n等待第二个点...")
                        time.sleep(0.5)  # 防止重复检测
                        self.space_pressed = False  # 重置状态
                else:
                    # 等待第二个点
                    if self.space_pressed:
                        x, y = pyautogui.position()  # 只在按键时获取
                        end_pos = (x, y)
                        
                        # 计算区域
                        left = min(start_pos[0], end_pos[0])
                        top = min(start_pos[1], end_pos[1])
                        width = abs(end_pos[0] - start_pos[0])
                        height = abs(end_pos[1] - start_pos[1])
                        
                        if width > 10 and height > 10:  # 最小区域限制
                            self.capture_region = (left, top, width, height)
                            print(f"✓ 右下角已设置: {end_pos}")
                            print(f"✓ 截图区域已设置: 左上角({left}, {top}), 宽度:{width}, 高度:{height}")
                            
                            # 显示预览
                            self._show_region_preview()
                            time.sleep(0.5)
                            return True
                        else:
                            print("选择的区域太小，请重新选择")
                            start_pos = None
                            print("\n请将鼠标移动到截图区域的【左上角】")
                            print("按下【空格键】确认第一个点...")
                            print("\n等待第一个点...")
                            time.sleep(0.5)
                            self.space_pressed = False
                
                time.sleep(0.05)  # 降低CPU占用
        finally:
            self.stop_keyboard_listener()
    
    def select_target_window(self):
        """选择目标窗口用于后台截图"""
        print("\n" + "="*50)
        print("选择目标窗口")
        print("="*50)
        
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        # 过滤掉系统窗口
        filtered_windows = [(hwnd, title) for hwnd, title in windows 
                          if title and not title.startswith('Default IME') 
                          and not title.startswith('MSCTFIME')]
        
        print("可用窗口列表：")
        for i, (hwnd, title) in enumerate(filtered_windows[:20], 1):  # 只显示前20个
            print(f"{i}. {title[:60]}")
        
        while True:
            choice = input(f"\n请选择窗口 (1-{min(20, len(filtered_windows))}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < min(20, len(filtered_windows)):
                    self.target_window = filtered_windows[idx][0]
                    print(f"✓ 已选择窗口: {filtered_windows[idx][1]}")
                    return True
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入数字")
    
    def capture_window_screenshot(self, hwnd, region=None):
        """使用Windows API截取窗口截图（不影响鼠标键盘）"""
        try:
            # 获取窗口位置和大小
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            # 获取窗口设备上下文
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # 创建位图对象
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # 使用PrintWindow API截图（可以后台截图）
            windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
            
            # 转换为PIL Image
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )
            
            # 释放资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            # 如果指定了区域，裁剪图片
            if region:
                x, y, w, h = region
                # 转换为相对于窗口的坐标
                x_rel = x - left
                y_rel = y - top
                img = img.crop((x_rel, y_rel, x_rel + w, y_rel + h))
            
            return img
            
        except Exception as e:
            print(f"窗口截图失败: {e}")
            return None
    
    def send_background_click(self, hwnd, x, y):
        """发送后台点击消息（不移动鼠标）"""
        try:
            # 获取窗口位置
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            
            # 转换为相对于窗口的坐标
            x_rel = x - left
            y_rel = y - top
            
            # 构造lParam参数
            lParam = win32api.MAKELONG(x_rel, y_rel)
            
            # 发送鼠标消息
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            time.sleep(0.05)
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
            
            return True
        except Exception as e:
            print(f"发送点击失败: {e}")
            return False
    
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
            # 根据模式选择不同的操作方式
            if self.background_mode and self.target_window:
                # 后台模式：使用Windows API
                print(f"[后台模式] 发送点击到坐标: {self.click_coord}")
                self.send_background_click(self.target_window, 
                                         self.click_coord[0], 
                                         self.click_coord[1])
                
                # 等待页面加载
                print(f"等待 {wait_time} 秒...")
                time.sleep(wait_time)
                
                # 后台截图
                screenshot = self.capture_window_screenshot(self.target_window, 
                                                           self.capture_region)
                if screenshot is None:
                    print("后台截图失败，尝试使用常规方式")
                    screenshot = pyautogui.screenshot(region=self.capture_region)
            else:
                # 常规模式：使用pyautogui
                print(f"点击坐标: {self.click_coord}")
                pyautogui.click(self.click_coord[0], self.click_coord[1])
                
                # 等待页面加载
                print(f"等待 {wait_time} 秒...")
                time.sleep(wait_time)
                
                # 截取指定区域
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
        if self.background_mode:
            print("模式: 后台模式（不影响鼠标键盘）")
        else:
            print("模式: 常规模式（控制鼠标键盘）")
        print("按 'q' 停止抓取")
        print("="*50)
        
        self.is_running = True
        
        # 启动键盘监听
        self.start_keyboard_listener()
        self.q_pressed = False
        
        try:
            while self.is_running:
                if self.q_pressed:
                    print("\n停止抓取")
                    self.is_running = False
                    break
                
                # 执行抓取
                if self.capture_single():
                    print(f"等待 {interval} 秒后继续...")
                    
                    # 在等待期间检测停止键
                    for _ in range(int(interval * 10)):
                        if self.q_pressed:
                            print("\n停止抓取")
                            self.is_running = False
                            break
                        time.sleep(0.1)
                else:
                    print("抓取失败，停止运行")
                    break
        finally:
            self.stop_keyboard_listener()
        
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
    # 首先让用户选择站点
    print("\n" + "="*60)
    print("请选择要保存到的站点文件夹:")
    print("="*60)
    
    # 列出现有的站点文件夹
    base_dir = "data/real_captchas/merged"
    existing_sites = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)):
                existing_sites.append(item)
    
    # 显示选项
    for i, site in enumerate(sorted(existing_sites), 1):
        print(f"{i}. {site}")
    
    print(f"{len(existing_sites) + 1}. 创建新站点")
    print("="*60)
    
    # 获取用户选择
    while True:
        choice = input(f"请选择 (1-{len(existing_sites) + 1}): ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(existing_sites):
                selected_site = sorted(existing_sites)[choice_num - 1]
                save_dir = os.path.join(base_dir, selected_site)
                break
            elif choice_num == len(existing_sites) + 1:
                # 创建新站点
                new_site = input("请输入新站点名称 (例如: site3): ").strip()
                if new_site and not os.path.exists(os.path.join(base_dir, new_site)):
                    save_dir = os.path.join(base_dir, new_site)
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"✓ 已创建新站点文件夹: {new_site}")
                    break
                else:
                    print("站点名称无效或已存在")
            else:
                print("无效的选择")
        except ValueError:
            print("请输入数字")
    
    # 创建抓取器实例
    capturer = CaptchaCapturer(save_dir=save_dir)
    print(f"\n当前保存目录: {save_dir}")
    
    while True:
        print("\n" + "="*60)
        print("验证码抓取工具")
        print(f"当前站点: {os.path.basename(save_dir)}")
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
        print("B. 切换后台模式（当前: {}）".format("开启" if capturer.background_mode else "关闭"))
        print("W. 选择目标窗口（后台模式用）")
        print("S. 切换站点")
        print("0. 退出")
        print("="*60)
        
        choice = input("请选择操作 (0-9/B/W/S): ").strip().upper()
        
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
                # 后台模式检查
                if capturer.background_mode and not capturer.target_window:
                    print("后台模式需要先选择目标窗口（选项W）")
                else:
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
            print(f"  后台模式: {'开启' if capturer.background_mode else '关闭'}")
            if capturer.background_mode and capturer.target_window:
                try:
                    window_title = win32gui.GetWindowText(capturer.target_window)
                    print(f"  目标窗口: {window_title}")
                except:
                    print(f"  目标窗口: [已设置]")
            
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
        
        elif choice == 'B':
            # 切换后台模式
            capturer.background_mode = not capturer.background_mode
            if capturer.background_mode:
                print("✓ 后台模式已开启")
                print("  说明: 后台模式不会控制你的鼠标和键盘")
                print("  注意: 需要先选择目标窗口（选项W）")
                if not capturer.target_window:
                    print("  ⚠️ 请使用选项W选择目标窗口")
            else:
                print("✓ 后台模式已关闭")
                print("  说明: 使用常规模式，会控制鼠标和键盘")
        
        elif choice == 'W':
            # 选择目标窗口
            capturer.select_target_window()
            
        elif choice == 'S':
            # 切换站点
            print("\n" + "="*60)
            print("切换站点:")
            print("="*60)
            
            # 列出现有的站点文件夹
            base_dir = "data/real_captchas/merged"
            existing_sites = []
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    if os.path.isdir(os.path.join(base_dir, item)):
                        existing_sites.append(item)
            
            # 显示选项
            for i, site in enumerate(sorted(existing_sites), 1):
                current = " (当前)" if os.path.join(base_dir, site) == capturer.save_dir else ""
                print(f"{i}. {site}{current}")
            
            print(f"{len(existing_sites) + 1}. 创建新站点")
            print("0. 取消")
            print("="*60)
            
            # 获取用户选择
            site_choice = input(f"请选择 (0-{len(existing_sites) + 1}): ").strip()
            try:
                site_choice_num = int(site_choice)
                if site_choice_num == 0:
                    print("取消切换")
                elif 1 <= site_choice_num <= len(existing_sites):
                    selected_site = sorted(existing_sites)[site_choice_num - 1]
                    new_save_dir = os.path.join(base_dir, selected_site)
                    capturer.save_dir = new_save_dir
                    capturer.capture_count = 0  # 重置计数
                    print(f"✓ 已切换到站点: {selected_site}")
                    print(f"   保存目录: {new_save_dir}")
                    
                    # 尝试加载该站点的配置
                    if capturer.load_config():
                        print("   已自动加载该站点的配置")
                elif site_choice_num == len(existing_sites) + 1:
                    # 创建新站点
                    new_site = input("请输入新站点名称 (例如: site3): ").strip()
                    if new_site and not os.path.exists(os.path.join(base_dir, new_site)):
                        new_save_dir = os.path.join(base_dir, new_site)
                        os.makedirs(new_save_dir, exist_ok=True)
                        capturer.save_dir = new_save_dir
                        capturer.capture_count = 0
                        capturer.click_coord = None
                        capturer.capture_region = None
                        print(f"✓ 已创建并切换到新站点: {new_site}")
                        print(f"   保存目录: {new_save_dir}")
                    else:
                        print("站点名称无效或已存在")
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入数字")
            
        elif choice == '0':
            print("退出程序")
            break
            
        else:
            print("无效的选择")

if __name__ == "__main__":
    print("欢迎使用验证码抓取工具！")
    print("提示: 请确保目标网页已经打开并可见")
    main()