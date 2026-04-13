
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene,QLineEdit,QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,QColor
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QPushButton, QRadioButton, QHBoxLayout,QGroupBox,QScrollArea,QInputDialog,QFrame
from salt.json2txt import MakeTxt
class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None
        self.edit_mode = False
        self.dragging = False
        self.dragging_corner = None
        self.dragging_edge = None
        self.drag_radius = 10
        self.edge_radius = 8

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # Zoom at mouse position without requiring click
        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        event.accept()
    
    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:

        if self.image_item is None:
            return
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()

        if self.edit_mode and self.editor.current_box is not None and event.button() == Qt.LeftButton:
            pts = self.editor.current_box
            if pts is not None and len(pts) == 4:
                # Edge drag: find nearest edge
                edge_idx = self._hit_test_edge(x, y, pts)
                if edge_idx is not None:
                    self.dragging = True
                    self.dragging_edge = edge_idx
                    return
                return

        if event.button() == Qt.LeftButton:
            label = 1
        elif event.button() == Qt.RightButton:
            label = 0
        else:
            return
        self.editor.add_click([int(x), int(y)], label)
        self.imshow(self.editor.display)


    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.dragging or self.image_item is None:
            return
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()
        h, w = self.editor.image_bgr.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if self.dragging_edge is not None:
            self.editor.update_box_edge(self.dragging_edge, [x, y])
            self.editor.render_current()
            self.imshow(self.editor.display)


    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.dragging:
            self.dragging = False
            self.dragging_corner = None
            self.dragging_edge = None

    def _hit_test_edge(self, x, y, pts):
        import math
        p = (x, y)
        pts = self.editor.order_box_points(pts)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        best = None
        best_d = None
        for idx, (i, j) in enumerate(edges):
            a = pts[i]
            b = pts[j]
            d = self._point_to_segment_distance(p, a, b)
            if best_d is None or d < best_d:
                best_d = d
                best = idx
        if best_d is not None and best_d <= self.edge_radius:
            return best
        return None

    def _point_to_segment_distance(self, p, a, b):
        import math
        px, py = p
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by)
        t = c1 / c2
        projx = ax + t * vx
        projy = ay + t * vy
        return math.hypot(px - projx, py - projy)


class ApplicationInterface(QWidget):
    def __init__(self, app, editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()

        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.labels = [] 
        self.selected_label_idx = -1  # 当前选中的标签索引
        self.selected_ann_idx = -1  # 当前选中的对象索引

        self.main_window = QHBoxLayout()
        
        self.graphics_view = CustomGraphicsView(self.editor)
        self.main_window.addWidget(self.graphics_view)
        self.layout.addLayout(self.main_window)

        # 右侧面板 - 标签和标注信息
        self.setLayout(self.layout)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_panel.setFixedWidth(200)
        
        # 标签管理区域
        self.label_group1 = QGroupBox("如果继续标注先导入标签")
        
        #self.label_group2 = QGroupBox("先导入现有标签")
        self.label_layout = QVBoxLayout(self.label_group1)
        #self.label_layout = QVBoxLayout(self.label_group2)
        # 添加标签按钮
        self.import_label_btn = QPushButton("+ 导入标签")
        self.import_label_btn.clicked.connect(self.import_label)
        self.label_layout.addWidget(self.import_label_btn)

        self.add_label_btn = QPushButton("+ 添加标签")
        self.add_label_btn.clicked.connect(self.add_label)
        self.label_layout.addWidget(self.add_label_btn)

        # 标签列表显示区域
        self.label_scroll = QScrollArea()
        self.label_scroll.setWidgetResizable(True)
        self.label_container = QWidget()
        self.label_container_layout = QVBoxLayout(self.label_container)
        self.label_container_layout.setAlignment(Qt.AlignTop)
        self.label_scroll.setWidget(self.label_container)
        self.label_layout.addWidget(self.label_scroll)
        
        self.right_layout.addWidget(self.label_group1)

        #self.right_layout.addWidget(self.label_group2)
        self.right_layout.addSpacing(10)
        self.main_window.addWidget(self.right_panel)

        # 底部对象列表
        self.bottom_bar = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_bar)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.addWidget(QLabel("当前图片对象"))
        self.ann_scroll = QScrollArea()
        self.ann_scroll.setWidgetResizable(True)
        self.ann_container = QWidget()
        self.ann_container_layout = QHBoxLayout(self.ann_container)
        self.ann_container_layout.setAlignment(Qt.AlignLeft)
        self.ann_scroll.setWidget(self.ann_container)
        self.bottom_layout.addWidget(self.ann_scroll)
        self.layout.addWidget(self.bottom_bar)


        self.graphics_view.imshow(self.editor.display)
        self.update_image_name()
        self.update_annotation_list()
        self.json2txt=MakeTxt()


    def reset(self):
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)
        self.update_annotation_list()

    def add(self):
        if self.selected_label_idx == -1:
            QMessageBox.information(self, "提示", "请先选择一个标签")
            return
        self.editor.save_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)
        self.update_annotation_list()

    def delet(self):
        self.editor.delet_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)
        self.update_annotation_list()
        
    def next_image(self):
        self.editor.next_image()
        self.graphics_view.imshow(self.editor.display)
        self.editor.save()

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.imshow(self.editor.display)    

    def toggle(self):
        self.editor.toggle()
        self.graphics_view.imshow(self.editor.display)    

    def genyolotxt(self):
        self.json2txt.setpath(self.editor.coco_json_path)
        self.json2txt.process_bbox_annotations()
        #self.graphics_view.imshow(self.editor.display)

    def genyoloObbtxt(self):
        self.json2txt.setpath(self.editor.coco_json_path)
        self.json2txt.process_rotated_annotations()

    def genyoloObbyaml(self):
        self.json2txt.setpath(self.editor.coco_json_path)
        try:
            yaml_path = self.json2txt.process_yolo_obb_yaml()
            QMessageBox.information(self, "提示", f"YOLO OBB YAML 已生成:\n{yaml_path}")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"生成 YOLO OBB YAML 失败:\n{e}")
    
    def save_all(self):
        self.editor.save()

    def toggle_edit_mode(self):
        self.graphics_view.edit_mode = not self.graphics_view.edit_mode
        if hasattr(self, "edit_box_btn") and self.edit_box_btn is not None:
            state = "开" if self.graphics_view.edit_mode else "关"
            self.edit_box_btn.setText(f"编辑框: {state}")

    def toggle_label_text(self):
        self.editor.du.show_label = not self.editor.du.show_label
        if hasattr(self, "label_text_btn") and self.label_text_btn is not None:
            state = "开" if self.editor.du.show_label else "关"
            self.label_text_btn.setText(f"显示标签: {state}")
        self.editor.render_current()
        self.graphics_view.imshow(self.editor.display)

    def get_top_bar(self):
        top_bar = QWidget()
        main_layout = QVBoxLayout(top_bar)
        
        # 图片名称显示
        self.image_name_label = QLabel("图片名称: ")
        main_layout.addWidget(self.image_name_label)
        # 快捷键说明
        self.shortcut_label = QLabel("快捷键：Esc 退出 | A 上一张 | D 下一张 | F 添加对象 | R 重置 | Ctrl+Z 撤销对象 | Ctrl+S 保存 | W/E 旋转(微调)")
        main_layout.addWidget(self.shortcut_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        
        buttons = [
            ("添加对象 (F)", lambda: self.add()),
            ("撤销对象 (Ctrl+Z)", lambda: self.delet()),
            ("重置 (R)", lambda: self.reset()),
            ("前一张 (A)", lambda: self.prev_image()),
            ("下一张 (D)", lambda: self.next_image()),
            ("显示已标注信息", lambda: self.toggle()),
            ("编辑框: 关", lambda: self.toggle_edit_mode()),
            ("显示标签: 关", lambda: self.toggle_label_text()),
            ("生成yolotxt", lambda: self.genyolotxt()),
            ("生成yoloObb", lambda: self.genyoloObbtxt()),
            ("生成yoloObbYaml", lambda: self.genyoloObbyaml()),
            ("保存 (Ctrl+S)", lambda: self.save_all())       
        ]
        self.edit_box_btn = None
        self.label_text_btn = None
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)
            if button.startswith("编辑框"):
                self.edit_box_btn = bt
            if button.startswith("显示标签"):
                self.label_text_btn = bt

        return top_bar

    # 添加更新图片名称的方法
    def update_image_name(self):
        image_name = self.editor.get_current_image_name()
        self.image_name_label.setText(f"图片名称: {image_name}")

    def update_annotation_list(self):
        if not hasattr(self, "ann_container_layout"):
            return
        while self.ann_container_layout.count():
            item = self.ann_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        anns = self.editor.dataset_explorer.get_annotations(self.editor.image_id, return_colors=False)
        categories = self.editor.get_categories()
        for i, ann in enumerate(anns):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet(f"background-color: {'#dddddd' if i == self.selected_ann_idx else '#ffffff'};")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(4, 4, 4, 4)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setStyleSheet("background-color: #ff4444; color: white; border: none;")
            del_btn.setVisible(i == self.selected_ann_idx)
            del_btn.clicked.connect(lambda checked, idx=i: self.delete_annotation(idx))
            layout.addWidget(del_btn, alignment=Qt.AlignLeft)

            cat_id = ann.get("category_id", -1)
            cat_name = categories[cat_id] if 0 <= cat_id < len(categories) else "unknown"
            text_label = QLabel(f"框 {i}: {cat_name}")
            text_label.setCursor(Qt.PointingHandCursor)
            text_label.mousePressEvent = lambda e, idx=i: self.select_annotation(idx)
            layout.addWidget(text_label)

            self.ann_container_layout.addWidget(frame)

    def select_annotation(self, idx):
        # 点击已选对象则取消选择
        if self.selected_ann_idx == idx:
            self.selected_ann_idx = -1
            self.editor.set_edit_annotation_index(None)
            self.graphics_view.edit_mode = False
            if hasattr(self, "edit_box_btn") and self.edit_box_btn is not None:
                self.edit_box_btn.setText("编辑框: 关")
            self.editor.current_box = None
            self.editor.render_current()
            self.graphics_view.imshow(self.editor.display)
            self.update_annotation_list()
            return

        self.selected_ann_idx = idx
        anns = self.editor.dataset_explorer.get_annotations(self.editor.image_id, return_colors=False)
        if 0 <= idx < len(anns):
            ann = anns[idx]
            if "rotated" in ann and len(ann["rotated"]) == 8:
                pts = [
                    ann["rotated"][0:2],
                    ann["rotated"][2:4],
                    ann["rotated"][4:6],
                    ann["rotated"][6:8],
                ]
                self.editor.set_current_box(pts)
                self.editor.set_edit_annotation_index(idx)
                self.graphics_view.edit_mode = True
                if hasattr(self, "edit_box_btn") and self.edit_box_btn is not None:
                    self.edit_box_btn.setText("编辑框: 开")
                self.editor.render_current()
                self.graphics_view.imshow(self.editor.display)
        self.update_annotation_list()

    def delete_annotation(self, idx):
        removed = self.editor.delete_annotation_by_index(idx)
        if removed:
            self.selected_ann_idx = -1
            self.editor.set_edit_annotation_index(None)
            self.graphics_view.imshow(self.editor.display)
            self.update_annotation_list()

    def add_label(self):
        """添加新标签"""
        # 获取标签名称
        label_name, ok = QInputDialog.getText(self, "添加标签", "请输入标签名称:")
        if not ok or not label_name.strip():
            return
        
        label_name = label_name.strip()
        
        # 检查标签名称是否已存在
        for label in self.labels:
            if label["name"] == label_name:
                QMessageBox.warning(self, "警告", "该标签名称已存在!")
                return
        
        # 生成不重复的随机颜色
        color = self.get_random_color()
        
        # 添加新标签
        new_label = {
            "name": label_name,
            "color": color,
            "id": len(self.labels)
        }
        self.labels.append(new_label)

        # 更新标签列表显示
        self.update_label_list()
    def get_random_color(self):
        """生成随机RGB颜色，格式为#RRGGBB"""
        while True:
            color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # 确保颜色不是白色且与背景有足够对比度
            if color != QColor(255, 255, 255):
                # 检查颜色是否已存在
                color_str = color.name()
                if not any(label["color"] == color_str for label in self.labels):
                    return color_str

    def import_label(self):
        """导入标注数据（json文件）"""
        # 选择文件夹
        
        for i ,cat in enumerate(self.editor.get_categorie()):

            label_name=cat["name"]
            # 检查标签名称是否已存在
            t=0
            for label in self.labels:
                #print(label["name"])
                #print(label_name)
                if label["name"] == label_name:
                    t+=1
                    #QMessageBox.warning(self, "警告", "该标签名称已存在!")
                    #return
                    #print(1)
            if t==0:
                color = self.get_random_color()
                new_label = {
                "name": label_name,
                "color": color,
                "id": len(self.labels)}
                self.labels.append(new_label)
                self.update_label_list()
            
        print(self.labels)


    # 修改 next_image 和 prev_image 方法，添加更新图片名称的调用
    def next_image(self):
        self.editor.next_image()
        self.graphics_view.imshow(self.editor.display)
        self.editor.save()
        self.update_image_name()  # 添加这行
        self.update_annotation_list()

    def prev_image(self):
        self.editor.prev_image()
        self.graphics_view.imshow(self.editor.display)    
        self.update_image_name()  # 添加这行
        self.update_annotation_list()


    def add_new_category(self):
        idx=self.selected_label_idx
        #print(self.labels[idx]["name"])
        category_name = self.labels[idx]["name"]
        if category_name and category_name not in self.editor.get_categories():
            # 添加到编辑器
            self.editor.add_category(category_name)
            # 更新UI

    def update_label_list(self):
        """更新标签列表显示"""
        # 清空现有标签
        while self.label_container_layout.count():
            item = self.label_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 重新添加所有标签
        categories = self.editor.get_categories()
        for i, label in enumerate(self.labels):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet(f"background-color: {'#dddddd' if i == self.selected_label_idx else '#ffffff'};")
            
            layout = QHBoxLayout(frame)
            
            # 标签颜色块
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet(f"background-color: {label['color']};")
            layout.addWidget(color_label)
            
            # 标签名称和ID
            text_label = QLabel(f"{label['id']}: {label['name']}")
            text_label.setStyleSheet("padding: 2px;")
            text_label.setCursor(Qt.PointingHandCursor)

            text_label.mousePressEvent = lambda e, idx=i: self.select_label(idx)
            layout.addWidget(text_label, 1)
            
            # 删除按钮
            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setStyleSheet("background-color: #ff4444; color: white; border: none;")
            del_btn.clicked.connect(lambda checked, idx=i: self.remove_category(idx))
            layout.addWidget(del_btn)
            self.label_container_layout.addWidget(frame)
    def select_label(self, idx):
        # 选择标签
        self.selected_label_idx = idx
        if self.selected_label_idx == -1:
            QMessageBox.information(self, "提示", "请先选择一个标签")
            return
        # 清除对象选择，回到SAM打框模式
        self.selected_ann_idx = -1
        self.editor.set_edit_annotation_index(None)
        self.graphics_view.edit_mode = False
        if hasattr(self, "edit_box_btn") and self.edit_box_btn is not None:
            self.edit_box_btn.setText("编辑框: 关")
        self.editor.current_box = None
        self.editor.render_current()

        self.add_new_category()
        self.editor.select_category(idx)
        self.update_label_list()
        self.update_annotation_list()

    def remove_category(self, idx):
        # 询问用户确认
        reply = QMessageBox.question(
            self, 
            '确认删除', 
            f'确定要删除类别 "{idx}" 吗？\n该类别下的所有标注也将被删除。',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if idx == self.selected_label_idx:
                self.selected_label_idx = -1
            

            # 调用数据集管理器删除类别
            self.editor.dataset_explorer.remove_category(self.labels[idx]["name"])
            # 从列表中删除标签
            del self.labels[idx]
            for i in range(idx, len(self.labels)):
                self.labels[i]["id"] = i
            # 刷新标签栏显示（关键修复：删除后强制更新UI）
            self.update_label_list()
            self.graphics_view.imshow(self.editor.display)




    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_F:
            self.add()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.delet()
        if event.key() == Qt.Key_W:
            self.editor.rotate_current_box(-1)
            self.editor.render_current()
            self.graphics_view.imshow(self.editor.display)
        if event.key() == Qt.Key_E:
            self.editor.rotate_current_box(1)
            self.editor.render_current()
            self.graphics_view.imshow(self.editor.display)
