import json
import os
from pathlib import Path
from collections import defaultdict


class MakeTxt():
    def __init__(self):
        self.json_path=""

    def setpath(self,path):
        self.json_path=path
        #print(self.json_path)

    def process_bbox_annotations(self):
        """
        处理bbox标注，生成labels文件夹及对应的txt文件
        每个txt文件包含归一化的bbox坐标和类别ID（若文件已存在则清空重写）
        """
        # 读取JSON文件
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        #print(1)
        # 创建labels文件夹（如果不存在）
        images_dir = Path('labels')
        images_dir.mkdir(exist_ok=True)
        
        # 创建labels子文件夹（如果不存在）
        labels_dir = images_dir / 'labels'
        labels_dir.mkdir(exist_ok=True)
        
        # 构建图片ID到图片信息的映射
        image_info = {img['id']: img for img in data['images']}
        
        # 按image_id分组标注
        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # 处理每张图片的所有标注
        for img_id, img in image_info.items():
            # 获取当前图片的所有标注
            anns = annotations_by_image.get(img_id, [])
            if not anns:
                continue  # 无标注则跳过
            
            # 获取图片信息
            file_name = img['file_name']
            width = img['width']
            height = img['height']
            
            # 生成txt文件路径
            txt_file_name = Path(file_name).stem + '.txt'
            txt_path = labels_dir / txt_file_name
            
            # 收集当前图片的所有标注行
            lines = []
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, w, h]
                
                # 归一化bbox坐标（中心x、中心y、宽、高）
                x, y, w, h = bbox
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # 用w模式写入（清空原有内容）
            with open(txt_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')

    def process_rotated_annotations(self):
        """
        处理rotated标注，生成datasets/labels文件夹及对应的txt文件
        每个txt文件包含归一化的旋转矩形坐标和类别ID（若文件已存在则清空重写）
        """
        # 读取JSON文件
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # OBB 标签导出到数据集目录下，兼容 YOLO 默认 images/labels 约定
        dataset_root = Path(self.json_path).resolve().parent
        labels_obb_dir = dataset_root / "labels"
        labels_obb_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建图片ID到图片信息的映射
        image_info = {img['id']: img for img in data['images']}
        
        # 按image_id分组标注
        annotations_by_image = defaultdict(list)
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # 处理每张图片的所有标注
        for img_id, img in image_info.items():
            # 获取当前图片的所有标注
            anns = annotations_by_image.get(img_id, [])
            if not anns:
                continue  # 无标注则跳过
            
            # 获取图片信息
            file_name = img['file_name']
            width = img['width']
            height = img['height']
            
            # 生成txt文件路径
            txt_file_name = Path(file_name).stem + '.txt'
            txt_path = labels_obb_dir / txt_file_name
            
            # 收集当前图片的所有标注行
            lines = []
            for ann in anns:
                category_id = ann['category_id']
                rotated = ann['rotated']  # [x1, y1, x2, y2, x3, y3, x4, y4]
                
                # 归一化rotated坐标
                normalized_rotated = []
                for i in range(0, 8, 2):
                    x = rotated[i]
                    y = rotated[i+1]
                    normalized_rotated.append(x / width)
                    normalized_rotated.append(y / height)
                
                lines.append(f"{category_id} " + " ".join([f"{coord:.6f}" for coord in normalized_rotated]))
            
            # 用w模式写入（清空原有内容）
            with open(txt_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')

    def process_yolo_obb_yaml(self):
        """
        生成YOLO OBB训练所需的data.yaml
        - 默认写入 datasets/data_obb.yaml
        - 类别名称来自annotations.json中的categories
        """
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        if not categories:
            raise ValueError("annotations.json 中没有 categories，无法生成 yolo obb yaml")

        # 兼容非连续id：按id排序后输出为id:name映射
        categories = sorted(categories, key=lambda c: c["id"])
        names = {cat["id"]: cat["name"] for cat in categories}

        dataset_root = Path(self.json_path).resolve().parent
        yaml_path = dataset_root / "data_obb.yaml"

        # 使用相对路径，典型值为 datasets
        relative_dataset_root = Path(os.path.relpath(dataset_root, Path.cwd())).as_posix()
        yaml_lines = [
            f"path: {relative_dataset_root}",
            "train: images",
            "val: images",
            f"nc: {len(names)}",
            "names:"
        ]
        for class_id, class_name in names.items():
            escaped_name = str(class_name).replace('"', '\\"')
            yaml_lines.append(f'  {class_id}: "{escaped_name}"')

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

        return str(yaml_path)
