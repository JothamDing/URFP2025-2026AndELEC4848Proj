"""
USB Camera Dataset Capture Tool
用于采集数据集的USB摄像头拍照软件
支持手动调整曝光、白平衡等参数
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime
import threading
import json
import time
import re


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数据集采集工具 - USB Camera Capture")
        self.root.geometry("1200x800")
        
        # 相机相关变量
        self.cap = None
        self.camera_index = 0
        self.is_running = False
        self.current_frame = None
        
        # 保存路径
        self.save_path = os.path.join(os.path.dirname(__file__), "captured_images")
        self.image_counter = 0
        self.prefix = "img"
        
        # 相机参数默认值
        self.params = {
            'exposure': -6,           # 曝光值 (负数，范围 -9 到 -1)
            'gain': 0,                # 增益
            'brightness': 128,        # 亮度 (0-255)
            'contrast': 128,          # 对比度 (0-255)
            'saturation': 128,        # 饱和度 (0-255)
            'white_balance': 4500,    # 白平衡色温 (2800-6500K)
            'sharpness': 128,         # 锐度 (0-255)
            'gamma': 100,             # Gamma值
        }
        
        self.setup_ui()
        self.load_settings()
        self.scan_existing_images()  # 扫描已有图片，设置正确的起始编号
        
    def scan_existing_images(self):
        """扫描保存文件夹中已有的图片，找到最大编号"""
        save_dir = self.path_var.get()
        if not os.path.exists(save_dir):
            self.image_counter = 0
            self.count_var.set("0")
            return
        
        max_batch = -1
        # 匹配文件名格式: XXXX-Y.jpg
        pattern = re.compile(r'^(\d{4})-\d\.jpg$', re.IGNORECASE)
        
        for filename in os.listdir(save_dir):
            match = pattern.match(filename)
            if match:
                batch_num = int(match.group(1))
                if batch_num > max_batch:
                    max_batch = batch_num
        
        # 下一个编号从 max_batch + 1 开始
        self.image_counter = max_batch + 1
        self.count_var.set(str(self.image_counter))
        self.status_var.set(f"扫描到已有图片，下一批从 {self.image_counter:04d} 开始")
        
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：视频预览区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 视频显示标签
        self.video_label = ttk.Label(left_frame, text="点击'开启相机'开始预览")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(left_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
        
        # 右侧：控制面板
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_frame.pack_propagate(False)
        
        # === 相机选择区域 ===
        camera_frame = ttk.LabelFrame(right_frame, text="相机选择", padding="5")
        camera_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(camera_frame, text="相机索引:").pack(side=tk.LEFT)
        self.camera_index_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_index_var, 
                                     values=["0", "1", "2", "3"], width=5)
        camera_combo.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(camera_frame, text="开启相机", command=self.toggle_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # === 曝光控制区域 ===
        exposure_frame = ttk.LabelFrame(right_frame, text="曝光控制 (禁用自动曝光)", padding="5")
        exposure_frame.pack(fill=tk.X, pady=5)
        
        # 曝光值
        ttk.Label(exposure_frame, text="曝光值:").grid(row=0, column=0, sticky=tk.W)
        self.exposure_var = tk.DoubleVar(value=self.params['exposure'])
        self.exposure_scale = ttk.Scale(exposure_frame, from_=-9, to=-1, 
                                         variable=self.exposure_var, orient=tk.HORIZONTAL,
                                         command=lambda x: self.update_camera_param('exposure'))
        self.exposure_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.exposure_label = ttk.Label(exposure_frame, text=str(self.params['exposure']), width=5)
        self.exposure_label.grid(row=0, column=2)
        
        # 增益
        ttk.Label(exposure_frame, text="增益:").grid(row=1, column=0, sticky=tk.W)
        self.gain_var = tk.DoubleVar(value=self.params['gain'])
        self.gain_scale = ttk.Scale(exposure_frame, from_=0, to=255, 
                                     variable=self.gain_var, orient=tk.HORIZONTAL,
                                     command=lambda x: self.update_camera_param('gain'))
        self.gain_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.gain_label = ttk.Label(exposure_frame, text=str(self.params['gain']))
        self.gain_label.grid(row=1, column=2)
        
        # 亮度
        ttk.Label(exposure_frame, text="亮度:").grid(row=2, column=0, sticky=tk.W)
        self.brightness_var = tk.DoubleVar(value=self.params['brightness'])
        self.brightness_scale = ttk.Scale(exposure_frame, from_=0, to=255, 
                                           variable=self.brightness_var, orient=tk.HORIZONTAL,
                                           command=lambda x: self.update_camera_param('brightness'))
        self.brightness_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.brightness_label = ttk.Label(exposure_frame, text=str(self.params['brightness']))
        self.brightness_label.grid(row=2, column=2)
        
        exposure_frame.columnconfigure(1, weight=1)
        
        # === 白平衡控制区域 ===
        wb_frame = ttk.LabelFrame(right_frame, text="白平衡控制 (禁用自动白平衡)", padding="5")
        wb_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(wb_frame, text="色温 (K):").grid(row=0, column=0, sticky=tk.W)
        self.wb_var = tk.DoubleVar(value=self.params['white_balance'])
        self.wb_scale = ttk.Scale(wb_frame, from_=2800, to=6500, 
                                   variable=self.wb_var, orient=tk.HORIZONTAL,
                                   command=lambda x: self.update_camera_param('white_balance'))
        self.wb_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.wb_label = ttk.Label(wb_frame, text=str(self.params['white_balance']))
        self.wb_label.grid(row=0, column=2)
        
        wb_frame.columnconfigure(1, weight=1)
        
        # === 图像调整区域 ===
        image_frame = ttk.LabelFrame(right_frame, text="图像调整", padding="5")
        image_frame.pack(fill=tk.X, pady=5)
        
        # 对比度
        ttk.Label(image_frame, text="对比度:").grid(row=0, column=0, sticky=tk.W)
        self.contrast_var = tk.DoubleVar(value=self.params['contrast'])
        self.contrast_scale = ttk.Scale(image_frame, from_=0, to=255, 
                                         variable=self.contrast_var, orient=tk.HORIZONTAL,
                                         command=lambda x: self.update_camera_param('contrast'))
        self.contrast_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.contrast_label = ttk.Label(image_frame, text=str(self.params['contrast']))
        self.contrast_label.grid(row=0, column=2)
        
        # 饱和度
        ttk.Label(image_frame, text="饱和度:").grid(row=1, column=0, sticky=tk.W)
        self.saturation_var = tk.DoubleVar(value=self.params['saturation'])
        self.saturation_scale = ttk.Scale(image_frame, from_=0, to=255, 
                                           variable=self.saturation_var, orient=tk.HORIZONTAL,
                                           command=lambda x: self.update_camera_param('saturation'))
        self.saturation_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.saturation_label = ttk.Label(image_frame, text=str(self.params['saturation']))
        self.saturation_label.grid(row=1, column=2)
        
        # 锐度
        ttk.Label(image_frame, text="锐度:").grid(row=2, column=0, sticky=tk.W)
        self.sharpness_var = tk.DoubleVar(value=self.params['sharpness'])
        self.sharpness_scale = ttk.Scale(image_frame, from_=0, to=255, 
                                          variable=self.sharpness_var, orient=tk.HORIZONTAL,
                                          command=lambda x: self.update_camera_param('sharpness'))
        self.sharpness_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.sharpness_label = ttk.Label(image_frame, text=str(self.params['sharpness']))
        self.sharpness_label.grid(row=2, column=2)
        
        # Gamma
        ttk.Label(image_frame, text="Gamma:").grid(row=3, column=0, sticky=tk.W)
        self.gamma_var = tk.DoubleVar(value=self.params['gamma'])
        self.gamma_scale = ttk.Scale(image_frame, from_=50, to=200, 
                                      variable=self.gamma_var, orient=tk.HORIZONTAL,
                                      command=lambda x: self.update_camera_param('gamma'))
        self.gamma_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        self.gamma_label = ttk.Label(image_frame, text=str(self.params['gamma']))
        self.gamma_label.grid(row=3, column=2)
        
        image_frame.columnconfigure(1, weight=1)
        
        # === 保存设置区域 ===
        save_frame = ttk.LabelFrame(right_frame, text="保存设置", padding="5")
        save_frame.pack(fill=tk.X, pady=5)
        
        # 保存路径
        ttk.Label(save_frame, text="保存路径:").pack(anchor=tk.W)
        path_frame = ttk.Frame(save_frame)
        path_frame.pack(fill=tk.X)
        
        self.path_var = tk.StringVar(value=self.save_path)
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="浏览", command=self.browse_path).pack(side=tk.LEFT, padx=2)
        
        # 文件前缀
        prefix_frame = ttk.Frame(save_frame)
        prefix_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prefix_frame, text="文件前缀:").pack(side=tk.LEFT)
        self.prefix_var = tk.StringVar(value=self.prefix)
        ttk.Entry(prefix_frame, textvariable=self.prefix_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # 图片计数
        count_frame = ttk.Frame(save_frame)
        count_frame.pack(fill=tk.X)
        ttk.Label(count_frame, text="已拍摄:").pack(side=tk.LEFT)
        self.count_var = tk.StringVar(value="0")
        ttk.Label(count_frame, textvariable=self.count_var).pack(side=tk.LEFT, padx=5)
        ttk.Button(count_frame, text="重置计数", command=self.reset_counter).pack(side=tk.LEFT, padx=5)
        
        # === 拍照按钮 ===
        capture_frame = ttk.Frame(right_frame)
        capture_frame.pack(fill=tk.X, pady=10)
        
        self.capture_btn = ttk.Button(capture_frame, text="📷 拍照3张 (S)", 
                                       command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, ipady=10)
        
        # === 设置保存/加载 ===
        settings_frame = ttk.Frame(right_frame)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(settings_frame, text="保存参数", command=self.save_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(settings_frame, text="加载参数", command=self.load_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(settings_frame, text="重置默认", command=self.reset_params).pack(side=tk.LEFT, padx=2)
        
        # 绑定快捷键
        self.root.bind('s', lambda e: self.capture_image())
        self.root.bind('S', lambda e: self.capture_image())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
    def toggle_camera(self):
        """开启/关闭相机"""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
            
    def start_camera(self):
        """开启相机"""
        try:
            self.camera_index = int(self.camera_index_var.get())
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                messagebox.showerror("错误", f"无法打开相机 {self.camera_index}")
                return
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # 禁用自动曝光和自动白平衡
            self.disable_auto_settings()
            
            # 应用当前参数
            self.apply_all_params()
            
            self.is_running = True
            self.start_btn.config(text="关闭相机")
            self.capture_btn.config(state=tk.NORMAL)
            self.status_var.set(f"相机 {self.camera_index} 已开启")
            
            # 开始视频更新线程
            self.update_video()
            
        except Exception as e:
            messagebox.showerror("错误", f"开启相机失败: {str(e)}")
            
    def stop_camera(self):
        """关闭相机"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="开启相机")
        self.capture_btn.config(state=tk.DISABLED)
        self.video_label.config(image='', text="点击'开启相机'开始预览")
        self.status_var.set("相机已关闭")
        
    def disable_auto_settings(self):
        """禁用自动曝光和自动白平衡"""
        if self.cap:
            # 禁用自动曝光 (0 = 手动模式)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = 手动模式 (DirectShow)
            
            # 禁用自动白平衡
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            
            # 某些相机使用不同的属性
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 禁用自动对焦
            
            self.status_var.set("已禁用自动曝光和自动白平衡")
            
    def apply_all_params(self):
        """应用所有相机参数"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.params['exposure'])
        self.cap.set(cv2.CAP_PROP_GAIN, self.params['gain'])
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.params['brightness'])
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.params['contrast'])
        self.cap.set(cv2.CAP_PROP_SATURATION, self.params['saturation'])
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.params['white_balance'])
        self.cap.set(cv2.CAP_PROP_SHARPNESS, self.params['sharpness'])
        self.cap.set(cv2.CAP_PROP_GAMMA, self.params['gamma'])
        
    def update_camera_param(self, param_name):
        """更新单个相机参数"""
        var_map = {
            'exposure': self.exposure_var,
            'gain': self.gain_var,
            'brightness': self.brightness_var,
            'contrast': self.contrast_var,
            'saturation': self.saturation_var,
            'white_balance': self.wb_var,
            'sharpness': self.sharpness_var,
            'gamma': self.gamma_var,
        }
        
        label_map = {
            'exposure': self.exposure_label,
            'gain': self.gain_label,
            'brightness': self.brightness_label,
            'contrast': self.contrast_label,
            'saturation': self.saturation_label,
            'white_balance': self.wb_label,
            'sharpness': self.sharpness_label,
            'gamma': self.gamma_label,
        }
        
        prop_map = {
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'white_balance': cv2.CAP_PROP_WB_TEMPERATURE,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'gamma': cv2.CAP_PROP_GAMMA,
        }
        
        value = var_map[param_name].get()
        int_value = int(round(value))
        self.params[param_name] = int_value
        label_map[param_name].config(text=str(int_value))
        
        if self.cap:
            # 对于曝光值，确保使用整数值
            self.cap.set(prop_map[param_name], int_value)
            
    def update_video(self):
        """更新视频预览"""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # 转换为RGB用于显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整大小以适应窗口
                height, width = frame_rgb.shape[:2]
                max_width = 800
                max_height = 600
                
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                
                # 在图像上显示信息
                info_text = f"Exp:{self.params['exposure']} Gain:{self.params['gain']} WB:{self.params['white_balance']}K"
                cv2.putText(frame_resized, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 转换为PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
            # 继续更新
            self.root.after(30, self.update_video)
            
    def capture_single_image(self, exposure_value, batch_index, image_index):
        """拍摄单张图片并保存"""
        save_dir = self.path_var.get()
        
        # 设置曝光值
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        
        # 等待相机稳定并读取新帧
        time.sleep(0.15)
        for _ in range(3):  # 丢弃几帧让曝光稳定
            self.cap.read()
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        # 生成文件名: XXXX-Y.jpg
        filename = f"{batch_index:04d}-{image_index}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # 保存图像
        cv2.imwrite(filepath, frame)
        
        return filename
    
    def capture_image(self):
        """拍照保存 - 拍摄3张不同曝光的图片"""
        if not self.is_running or self.cap is None:
            return
            
        # 确保保存目录存在
        save_dir = self.path_var.get()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 获取当前曝光值
        current_exposure = int(self.params['exposure'])
        
        # 计算三个曝光值 (当前, -1, +1)，限制在有效范围内
        exposure_values = [
            current_exposure,                    # 0: 当前曝光
            max(-9, current_exposure - 1),       # 1: 降低曝光
            min(-1, current_exposure + 1),       # 2: 提高曝光
        ]
        
        self.status_var.set("正在拍摄多曝光图片...")
        self.root.update()
        
        saved_files = []
        for idx, exp_value in enumerate(exposure_values):
            filename = self.capture_single_image(exp_value, self.image_counter, idx)
            if filename:
                saved_files.append(filename)
        
        # 恢复原始曝光值
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
        
        self.image_counter += 1
        self.count_var.set(str(self.image_counter))
        self.status_var.set(f"已保存: {self.image_counter-1:04d}-0/1/2.jpg")
            
    def browse_path(self):
        """选择保存路径"""
        path = filedialog.askdirectory(initialdir=self.save_path)
        if path:
            self.path_var.set(path)
            self.save_path = path
            self.scan_existing_images()  # 重新扫描新路径中的图片
            
    def reset_counter(self):
        """重置计数器 - 重新扫描文件夹"""
        self.scan_existing_images()
        
    def save_settings(self):
        """保存参数设置到文件"""
        settings = {
            'params': self.params,
            'save_path': self.path_var.get(),
            'prefix': self.prefix_var.get()
        }
        settings_file = os.path.join(os.path.dirname(__file__), 'camera_settings.json')
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        self.status_var.set("参数已保存")
        
    def load_settings(self):
        """从文件加载参数设置"""
        settings_file = os.path.join(os.path.dirname(__file__), 'camera_settings.json')
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self.params = settings.get('params', self.params)
                self.path_var.set(settings.get('save_path', self.save_path))
                self.prefix_var.set(settings.get('prefix', self.prefix))
                
                # 更新UI
                self.exposure_var.set(self.params['exposure'])
                self.gain_var.set(self.params['gain'])
                self.brightness_var.set(self.params['brightness'])
                self.contrast_var.set(self.params['contrast'])
                self.saturation_var.set(self.params['saturation'])
                self.wb_var.set(self.params['white_balance'])
                self.sharpness_var.set(self.params['sharpness'])
                self.gamma_var.set(self.params['gamma'])
                
                # 更新标签
                self.exposure_label.config(text=str(int(self.params['exposure'])))
                self.gain_label.config(text=str(int(self.params['gain'])))
                self.brightness_label.config(text=str(int(self.params['brightness'])))
                self.contrast_label.config(text=str(int(self.params['contrast'])))
                self.saturation_label.config(text=str(int(self.params['saturation'])))
                self.wb_label.config(text=str(int(self.params['white_balance'])))
                self.sharpness_label.config(text=str(int(self.params['sharpness'])))
                self.gamma_label.config(text=str(int(self.params['gamma'])))
                
                if self.cap:
                    self.apply_all_params()
                    
                self.status_var.set("参数已加载")
            except Exception as e:
                print(f"加载设置失败: {e}")
                
    def reset_params(self):
        """重置为默认参数"""
        self.params = {
            'exposure': -6,
            'gain': 0,
            'brightness': 128,
            'contrast': 128,
            'saturation': 128,
            'white_balance': 4500,
            'sharpness': 128,
            'gamma': 100,
        }
        
        # 更新UI
        self.exposure_var.set(self.params['exposure'])
        self.gain_var.set(self.params['gain'])
        self.brightness_var.set(self.params['brightness'])
        self.contrast_var.set(self.params['contrast'])
        self.saturation_var.set(self.params['saturation'])
        self.wb_var.set(self.params['white_balance'])
        self.sharpness_var.set(self.params['sharpness'])
        self.gamma_var.set(self.params['gamma'])
        
        for label, value in [
            (self.exposure_label, self.params['exposure']),
            (self.gain_label, self.params['gain']),
            (self.brightness_label, self.params['brightness']),
            (self.contrast_label, self.params['contrast']),
            (self.saturation_label, self.params['saturation']),
            (self.wb_label, self.params['white_balance']),
            (self.sharpness_label, self.params['sharpness']),
            (self.gamma_label, self.params['gamma']),
        ]:
            label.config(text=str(int(value)))
        
        if self.cap:
            self.apply_all_params()
            
        self.status_var.set("已重置为默认参数")
        
    def on_closing(self):
        """关闭程序"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
