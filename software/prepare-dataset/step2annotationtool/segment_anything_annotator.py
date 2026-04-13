import os
import argparse
import sys
import shutil
import subprocess
from salt.editor import Editor
from salt.interface import ApplicationInterface
from segment_anything import sam_model_registry, SamPredictor
import cv2
from tqdm import tqdm
import numpy as np


def main(checkpoint_path, model_type, device, images_folder, embeddings_folder):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 获取图像文件夹中所有文件，并过滤出常见图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_names = [
        name for name in os.listdir(images_folder)
        if name.lower().endswith(image_extensions)
    ]

    # 遍历图像文件，仅处理不存在embedding的文件
    for image_name in tqdm(image_names, desc="生成图像嵌入"):
        # 构建输入图像路径和输出embedding路径
        image_path = os.path.join(images_folder, image_name)
        base_name = os.path.splitext(image_name)[0]
        embedding_path = os.path.join(embeddings_folder, f"{base_name}.npy")

        # 检查embedding文件是否已存在，存在则跳过
        if os.path.exists(embedding_path):
            continue

        # 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:  # 处理图像读取失败的情况
            print(f"警告：无法读取图像 {image_path}，已跳过")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成并保存嵌入
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        np.save(embedding_path, image_embedding)    


def cleanup_dataset_outputs(dataset_folder):
    embeddings_folder = os.path.join(dataset_folder, "embeddings")
    coco_json_path = os.path.join(dataset_folder, "annotations.json")
    if os.path.exists(embeddings_folder):
        shutil.rmtree(embeddings_folder)
        print(f"已删除旧目录: {embeddings_folder}")
    if os.path.exists(coco_json_path):
        os.remove(coco_json_path)
        print(f"已删除旧文件: {coco_json_path}")


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def is_wsl():
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            version = f.read().lower()
        return "microsoft" in version or "wsl" in version
    except OSError:
        return False


def auto_configure_qt_platform(requested_platform):
    if requested_platform != "auto":
        os.environ["QT_QPA_PLATFORM"] = requested_platform
        return requested_platform

    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    display = os.environ.get("DISPLAY")

    # WSLg 常见场景：优先 Wayland（无需 XLaunch）
    if wayland_display:
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return "wayland"

    # 已有 DISPLAY 的场景：按 X11 走
    if display:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        return "xcb"

    # WSL + XServer 场景：自动补 DISPLAY
    if is_wsl():
        try:
            nameserver = subprocess.check_output(
                "grep -m1 nameserver /etc/resolv.conf | awk '{print $2}'",
                shell=True,
                text=True
            ).strip()
            if nameserver:
                os.environ["DISPLAY"] = f"{nameserver}:0"
                os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
                os.environ["QT_QPA_PLATFORM"] = "xcb"
                return "xcb"
        except Exception:
            pass

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model-path", type=str, default="./sam_onnx.onnx")
    parser.add_argument("--dataset-path", type=str, default="datasets")
    parser.add_argument("--checkpoint-path", type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reset-dataset", action="store_true", help="启动前删除 datasets 下旧的 embeddings 和 annotations.json")
    parser.add_argument("--embed-only", action="store_true", help="只生成 embeddings，不启动 Qt 标注界面")
    parser.add_argument(
        "--qt-platform",
        type=str,
        default="auto",
        choices=["auto", "xcb", "wayland", "offscreen"],
        help="Qt 平台插件选择，默认自动判断（优先 wayland，其次 xcb）"
    )
    

    args = parser.parse_args()
    if args.reset_dataset:
        cleanup_dataset_outputs(args.dataset_path)

    # npy 文件生成启动部分
    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_folder = args.dataset_path
    images_folder = os.path.join(dataset_folder, "images")
    embeddings_folder = os.path.join(dataset_folder, "embeddings")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
    print("快捷键：")
    print("esc：退出")
    print("A：上一张")
    print("D：下一张")
    print("F：打框")
    print("ctrl+Z：撤回上一个框")
    print("ctrl+S：保存内容")

    if args.embed_only:
        print("仅生成 embeddings 模式已完成，未启动标注界面。")
        sys.exit(0)

    chosen_qt = auto_configure_qt_platform(args.qt_platform)
    if chosen_qt:
        print(f"Qt 平台: {chosen_qt}")
    if os.environ.get("DISPLAY"):
        print(f"DISPLAY={os.environ['DISPLAY']}")
    if os.environ.get("WAYLAND_DISPLAY"):
        print(f"WAYLAND_DISPLAY={os.environ['WAYLAND_DISPLAY']}")

    if not has_display():
        print("未检测到图形显示环境（DISPLAY/WAYLAND_DISPLAY）。")
        print("请优先使用 Windows 11 的 WSLg（无需 XLaunch），或使用 --embed-only 仅生成 embeddings。")
        print("如必须用 XLaunch，可指定 --qt-platform xcb。")
        sys.exit(1)

    #标注启动部分
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont

    onnx_model_path = args.onnx_model_path
    dataset_path = args.dataset_path

    coco_json_path = os.path.join(dataset_path,"annotations.json")
    categories=""

    editor = Editor(
        onnx_model_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )
    
    app = QApplication(sys.argv)
    
    # 设置中文字体
    font = QFont("WenQuanYi Micro Hei", 10)
    app.setFont(font)
    
    window = ApplicationInterface(app, editor)
    window.show()
    sys.exit(app.exec_())
