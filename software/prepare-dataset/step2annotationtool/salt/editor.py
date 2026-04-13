import os, copy, math
import numpy as np
import cv2
from salt.onnx_model import OnnxModel
from salt.dataset_explorer import DatasetExplorer, rotated_bounding_box_from_mask
from salt.display_utils import DisplayUtils

class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if len(self.input_point) == 0:
            self.input_point = np.array([input_point])
        else:
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    @staticmethod
    def order_box_points(pts):
        pts = np.array(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def __init__(self, onnx_model_path, dataset_path, categories=None, coco_json_path=None):
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        self.onnx_model_path = onnx_model_path
        self.onnx_helper = OnnxModel(self.onnx_model_path)
        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories = self.dataset_explorer.get_categories()
        
        self.current_category = categories[0] if categories else None  # 新增这行
        #print(self.categories)
        self.image_id = 0
        self.category_id = 0
        self.show_other_anns = True
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.du = DisplayUtils()
        self.current_box = None
        self.edit_ann_index = None
        self.reset()

    def add_click(self, new_pt, new_label):
        self.curr_inputs.add_input_click(new_pt, new_label)
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)
        self.update_current_box_from_mask()
        self.render_current()

    def draw_known_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )

        self.display = self.du.draw_annotations(self.display, self.categories, anns, colors)

    def render_current(self):
        self.display = self.image_bgr.copy()
        if self.show_other_anns:
            self.draw_known_annotations()
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.overlay_mask_on_image(
                self.display, self.curr_inputs.curr_mask
            )
        if len(self.curr_inputs.input_point) > 0:
            self.display = self.du.draw_points(
                self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
            )
        if self.current_box is not None and len(self.current_box) == 4:
            self.display = self.du.draw_edit_box(self.display, self.current_box)

    def update_current_box_from_mask(self):
        if self.curr_inputs.curr_mask is None:
            self.current_box = None
            self.edit_ann_index = None
            return
        box = rotated_bounding_box_from_mask(self.curr_inputs.curr_mask)
        if box is None or len(box) == 0:
            self.current_box = None
            self.edit_ann_index = None
            return
        self.current_box = self.order_box_points(box)
        self.edit_ann_index = None

    def set_current_box(self, points):
        self.current_box = self.order_box_points(points)

    def set_edit_annotation_index(self, ann_index):
        self.edit_ann_index = ann_index

    def commit_current_box(self):
        if self.edit_ann_index is None or self.current_box is None:
            return False
        return self.dataset_explorer.update_annotation_by_index(
            self.image_id, self.edit_ann_index, self.current_box
        )

    def update_box_corner(self, corner_idx, point):
        if self.current_box is None:
            return
        pts = self.order_box_points(self.current_box)
        if pts.shape != (4, 2):
            return
        # axes from current orientation (tl->tr, tl->bl)
        tl, tr, br, bl = pts
        x_axis = tr - tl
        y_axis = bl - tl
        x_len = np.linalg.norm(x_axis)
        y_len = np.linalg.norm(y_axis)
        if x_len < 1e-3 or y_len < 1e-3:
            return
        x_axis = x_axis / x_len
        y_axis = y_axis / y_len

        opp_idx = (corner_idx + 2) % 4
        p_opp = pts[opp_idx]
        p_new = np.array(point, dtype=np.float32)

        v = p_new - p_opp
        proj_x = float(np.dot(v, x_axis))
        proj_y = float(np.dot(v, y_axis))
        if abs(proj_x) < 1e-3 or abs(proj_y) < 1e-3:
            return
        sx = 1.0 if proj_x >= 0 else -1.0
        sy = 1.0 if proj_y >= 0 else -1.0

        half_w = max(2.0, abs(proj_x) / 2.0)
        half_h = max(2.0, abs(proj_y) / 2.0)
        center = (p_new + p_opp) / 2.0

        tl_new = center - x_axis * half_w - y_axis * half_h
        tr_new = center + x_axis * half_w - y_axis * half_h
        br_new = center + x_axis * half_w + y_axis * half_h
        bl_new = center - x_axis * half_w + y_axis * half_h

        self.current_box = np.array([tl_new, tr_new, br_new, bl_new], dtype=np.float32)
        self.commit_current_box()

    def update_box_edge(self, edge_idx, point):
        if self.current_box is None:
            return
        pts = self.order_box_points(self.current_box)
        tl, tr, br, bl = pts
        edges = [(tl, tr), (tr, br), (br, bl), (bl, tl)]  # top, right, bottom, left
        opp_edges = [(br, bl), (bl, tl), (tl, tr), (tr, br)]

        a, b = edges[edge_idx]
        opp_a, opp_b = opp_edges[edge_idx]

        edge_vec = b - a
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-3:
            return
        # normal pointing from opposite edge to this edge
        mid = (a + b) / 2.0
        opp_mid = (opp_a + opp_b) / 2.0
        normal = mid - opp_mid
        n_len = np.linalg.norm(normal)
        if n_len < 1e-3:
            return
        normal = normal / n_len

        p = np.array(point, dtype=np.float32)
        # signed distance from point to current edge line
        dist = np.dot(p - a, normal)

        # move selected edge by dist, opposite edge fixed
        new_a = a + normal * dist
        new_b = b + normal * dist

        if edge_idx == 0:  # top
            tl_new, tr_new, br_new, bl_new = new_a, new_b, br, bl
        elif edge_idx == 1:  # right
            tl_new, tr_new, br_new, bl_new = tl, new_a, new_b, bl
        elif edge_idx == 2:  # bottom
            tl_new, tr_new, br_new, bl_new = tl, tr, new_a, new_b
        else:  # left
            tl_new, tr_new, br_new, bl_new = new_b, tr, br, new_a

        self.current_box = np.array([tl_new, tr_new, br_new, bl_new], dtype=np.float32)
        self.commit_current_box()

    def rotate_current_box(self, angle_deg):
        if self.current_box is None:
            return
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        center = self.current_box.mean(axis=0)
        shifted = self.current_box - center
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        self.current_box = shifted @ rot.T + center
        self.commit_current_box()

    def add_category(self, category_name):
        if category_name not in self.categories:
            self.categories.append(category_name)
            # 更新数据集探索器的类别
            self.dataset_explorer.add_category(category_name)

    def get_current_image_name(self):
        return self.dataset_explorer.get_image_name(self.image_id)

    def reset(self, hard=True):
        self.curr_inputs.reset_inputs()
        self.current_box = None
        self.edit_ann_index = None
        self.display = self.image_bgr.copy()
        if self.show_other_anns:
            self.draw_known_annotations()

    def toggle(self):
        self.show_other_anns = not self.show_other_anns
        self.reset()
    
    def step_up_transparency(self):
        self.du.increase_transparency()
        self.reset()

    def step_down_transparency(self):
        self.du.decrease_transparency()
        self.reset()

    def save_ann(self):
        self.dataset_explorer.add_annotation(
            self.image_id,
            self.category_id,
            self.curr_inputs.curr_mask,
            rotated_box=self.current_box,
        )

    def delet_ann(self):
        self.dataset_explorer.delet_annotation(self.image_id)

    def delete_annotation_by_index(self, ann_index):
        removed = self.dataset_explorer.delete_annotation_by_index(self.image_id, ann_index)
        if removed:
            self.reset()
        return removed

    def save(self):
        self.dataset_explorer.save_annotation()

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id += 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1
    
    def get_categories(self):
        return self.categories

    def get_categorie(self):
        return self.dataset_explorer.coco_json["categories"]

    def select_category(self, idx):
        self.category_id = idx

    def remove_category(self, category_name):
        # 从数据集管理器中删除类别
        self.dataset_explorer.remove_category(category_name)
        # 如果当前选中的类别被删除，自动切换到第一个类别
        if self.current_category == category_name and self.get_categories():
            self.select_category(self.get_categories()[0])
    def load_image_by_id(self, image_id):
        """根据image_id加载对应图片"""
        if image_id < 0 or image_id >= len(self.dataset_explorer.coco_json["images"]):
            return
        self.image_id = image_id
        # 加载图片（复用原有加载逻辑，假设已有获取图片路径的方法）
        image_path = self.dataset_explorer.get_image_path_by_id(image_id)  # 需确保DatasetExplorer有此方法
        if image_path:
            self.display = cv2.imread(image_path)
            self.mask = np.zeros_like(self.display)
            # 重新绘制已有的标注
            anns, colors = self.dataset_explorer.get_annotations( self.image_id, return_colors=True)
            self.du.draw_annotations(self.display, self.categories, anns, colors)  # 假设已有绘制标注的方法

    # 添加获取当前类别列表的方法
    def get_categories(self):
        return self.dataset_explorer.categories
