# URFP2025-2026 + ELEC4848 Project

开源项目，聚焦电子元件目标检测的完整工程链路：

- 硬件部分：设备结构与控制板相关设计文件
- 数据部分：采集、标注、格式转换与数据集划分
- 模型部分：面向旋转框检测任务的训练配置与实验记录

本仓库适合以下场景：

- 课程项目复现
- 电子元件视觉检测研究
- 旋转目标检测（Oriented Object Detection）实验

## 项目结构

```text
URFP2025-2026AndELEC4848Proj/
├─ hardware/
│  ├─ corexyframe-v2.easm
│  ├─ controlBoard/
│  └─ testboard/
├─ poster80200/
├─ software/
│  ├─ elec-dataset/
│  │  └─ output/
│  │     ├─ train/
│  │     ├─ val/
│  │     └─ test/
│  ├─ prepare-dataset/
│  │  ├─ step1capture-dataset/
│  │  ├─ step2annotationtool/
│  │  └─ step3annotationtransfer/
│  └─ modeltraining/
└─ README.md
```

## 一图看懂工作流

1. 采集图像
2. 交互式标注
3. 标注格式转换与数据划分
4. 训练旋转框检测模型
5. 导出结果与评估指标

## 软件流水线说明

### Step 1: 数据采集

目录：`software/prepare-dataset/step1capture-dataset`

主要脚本：

- `camera_capture.py`：USB 相机采集工具（支持曝光、白平衡等参数控制）
- `copy_zero_images.py`：数据整理辅助脚本

安装与运行（示例）：

```bash
pip install -r software/prepare-dataset/step1capture-dataset/requirements.txt
python software/prepare-dataset/step1capture-dataset/camera_capture.py
```

### Step 2: 标注工具

目录：`software/prepare-dataset/step2annotationtool`

主要脚本：

- `segment_anything_annotator.py`：SAM 标注入口
- `helpers/generate_onnx.py`：SAM 权重转 ONNX（依赖上游 SAM 仓库）
- `statistics.sh`：统计相关脚本

说明：

- 该步骤依赖 Segment Anything 相关环境与模型权重
- 标注结果支持导出 YOLO 与 COCO 相关格式

### Step 3: 标注转换与划分

目录：`software/prepare-dataset/step3annotationtransfer`

主要脚本：

- `convert_yolo_obb_to_dota.py`：YOLO-OBB 转 DOTA，并按比例划分数据集
- `duplicate_label_variants.sh`：批量复制标签变体（如 `-0.txt -> -1.txt, -2.txt`）

运行（示例）：

```bash
python software/prepare-dataset/step3annotationtransfer/convert_yolo_obb_to_dota.py
# 可选：自定义划分比例
python software/prepare-dataset/step3annotationtransfer/convert_yolo_obb_to_dota.py --split-ratios 0.8,0.1,0.1
```

默认划分比例：`train/val/test = 0.7/0.15/0.15`

### 模型训练

目录：`software/modeltraining`

当前包含多套旋转检测配置与实验输出：

- `oriented_rcnn_dotav3/oriented_rcnn_r50_fpn_1x_dota_custom_optv3.py`
- `oriented_reppoints_dotav2/oriented_reppoints_r50_fpn_40e_dota_ms_le135_custom.py`
- `roi_trans_dotav2/roi_trans_r50_fpn_fp16_1x_dota_le90_custom.py`

建议基于 MMRotate 环境运行训练与评估。`software/modeltraining/README.md` 中提供了现有模型权重与环境记录。

## 硬件部分说明

目录：`hardware`

- `corexyframe-v2.easm`：结构设计文件
- `controlBoard/`：控制板设计相关文件
- `testboard/`：测试板工程及封装库

提示：该目录包含 EDA 工具相关工程文件（如 KiCad 与其他格式），请使用对应软件打开。

## 快速开始（建议顺序）

1. 准备 Python 环境（推荐单独虚拟环境）
2. 运行 Step 1 完成图像采集
3. 配置 SAM 并运行 Step 2 完成标注
4. 运行 Step 3 生成 DOTA 格式与 train/val/test
5. 进入 `software/modeltraining` 选择一个配置开始训练

## 已有数据与结果

仓库中已包含：

- `software/elec-dataset/output` 下的 train/val/test 样例数据
- `software/modeltraining` 下的训练日志、基准测试输出与指标汇总

## 依赖与环境

- Python 版本建议与 MMRotate 兼容
- 训练环境建议按 MMRotate 官方文档安装
- SAM 标注需要额外下载权重并准备 ONNX 模型

## 致谢

本项目使用并参考了以下开源生态：

- OpenMMLab / MMRotate
- Segment Anything (Meta)
- 相关数据标注工具与社区实现

## License

当前仓库根目录未提供统一 License 文件。

如果你希望对外发布并明确复用规则，建议补充一个 License（如 MIT、Apache-2.0 或 GPL-3.0）。
