# USB Camera 数据集采集工具

用于采集图像数据集的USB摄像头拍照软件，支持手动调整曝光、快门、白平衡等参数。

## 功能特点

- **禁用自动曝光/自动白平衡**: 确保拍摄参数一致性
- **手动参数控制**:
  - 曝光值 (Exposure)
  - 增益 (Gain)
  - 亮度 (Brightness)
  - 对比度 (Contrast)
  - 饱和度 (Saturation)
  - 白平衡色温 (White Balance)
  - 锐度 (Sharpness)
  - Gamma
- **实时预览**: 带参数信息叠加显示
- **元数据保存**: 每张图片自动保存拍摄参数的JSON文件
- **参数保存/加载**: 保存常用参数配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python camera_capture.py
```

## 使用说明

1. 选择相机索引 (通常为0)
2. 点击"开启相机"
3. 调整曝光、白平衡等参数
4. 设置保存路径和文件前缀
5. 按 **空格键** 或点击"拍照"按钮拍照

### 快捷键

- `Space`: 拍照
- `Esc`: 退出程序

## 输出文件

- `{prefix}_{timestamp}_{counter}.jpg` - 图像文件
- `{prefix}_{timestamp}_{counter}_meta.json` - 元数据文件 (包含拍摄参数)

## 注意事项

- 不同USB相机支持的参数范围可能不同
- 某些相机可能不支持部分参数调整
- 建议使用DirectShow兼容的USB相机
