# 简单的 Qt 曲线和缩略图；只显示数据，不发运动或访问设备。
import math

import cv2 as cv
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QListView, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


def roi_from_drag(start, end, view_size, image_size, display_size=None):
    """扣除等比显示的留边，返回原始图像的整数 ROI。"""
    width, height = image_size
    if min(width, height, *view_size) <= 0:
        raise ValueError("还没有可框选的相机画面")
    scale = min(view_size[0] / width, view_size[1] / height)
    shown_width, shown_height = display_size or (width * scale, height * scale)
    left, top = (view_size[0] - shown_width) / 2, (view_size[1] - shown_height) / 2
    points = [((p[0] - left) * width / shown_width, (p[1] - top) * height / shown_height) for p in (start, end)]
    if any(not 0 <= x <= width or not 0 <= y <= height for x, y in points):
        raise ValueError("请在图像内框选，黑边不属于原始图像")
    x, y = math.floor(min(p[0] for p in points)), math.floor(min(p[1] for p in points))
    right = min(width, math.ceil(max(p[0] for p in points)))
    bottom = min(height, math.ceil(max(p[1] for p in points)))
    if right - x < 8 or bottom - y < 8:
        raise ValueError("ROI 至少为 8×8 原始像素")
    return [x, y, right - x, bottom - y]


class FocusCurve(QWidget):
    point_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.records = []
        self.selected_id = None
        self.setMinimumHeight(90)
        self.setToolTip("点击曲线点只查看采集记录，不启动运动；横轴为 Z 指令脉冲")

    def point_positions(self):
        if not self.records:
            return []
        area = QRectF(65, 14, max(1, self.width() - 82), max(1, self.height() - 43))
        xs = [item["target_steps"] for item in self.records]
        ys = [item["focus_score"] for item in self.records]
        low_x, high_x = min(xs), max(xs)
        low_y, high_y = min(ys), max(ys)
        pad = max((high_y - low_y) * 0.1, max(abs(high_y), 1) * 0.01)
        low_y, high_y = low_y - pad, high_y + pad
        points = [QPointF(area.left() + (x - low_x) / max(1, high_x - low_x) * area.width(),
                         area.bottom() - (y - low_y) / (high_y - low_y) * area.height()) for x, y in zip(xs, ys)]
        return points

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fafbfc"))
        area = QRectF(65, 14, max(1, self.width() - 82), max(1, self.height() - 43))
        painter.setPen(QPen(QColor("#8a939e"), 1))
        painter.drawRect(area)
        painter.drawText(QRectF(area.left() + 80, area.bottom() + 4, max(1, area.width() - 160), 20), Qt.AlignCenter, "Z 指令位置 / 脉冲")
        if not self.records:
            painter.drawText(area, Qt.AlignCenter, "Tenengrad · 等待已保存采样点")
            return
        points = self.point_positions()
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.setPen(QPen(QColor("#236ca4"), 2))
        painter.drawPath(path)
        for record, point in zip(self.records, points):
            selected = record["sample_id"] == self.selected_id
            painter.setBrush(QColor("#d26920" if selected else "#236ca4"))
            painter.drawEllipse(point, 4 if selected else 2.5, 4 if selected else 2.5)
        painter.setPen(QColor("#46515f"))
        xs = [r["target_steps"] for r in self.records]
        ys = [r["focus_score"] for r in self.records]
        pad = max((max(ys) - min(ys)) * 0.1, max(abs(max(ys)), 1) * 0.01)
        painter.drawText(QRectF(2, area.top(), 59, 20), Qt.AlignRight, f"{max(ys) + pad:.4g}")
        painter.drawText(QRectF(2, area.bottom() - 20, 59, 20), Qt.AlignRight, f"{min(ys) - pad:.4g}")
        painter.drawText(QRectF(area.left(), area.bottom() + 4, 80, 18), Qt.AlignLeft, str(min(xs)))
        painter.drawText(QRectF(area.right() - 80, area.bottom() + 4, 80, 18), Qt.AlignRight, str(max(xs)))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.records:
            points = self.point_positions()
            index = min(range(len(points)), key=lambda i: (points[i].x() - event.x()) ** 2 + (points[i].y() - event.y()) ** 2)
            if (points[index].x() - event.x()) ** 2 + (points[index].y() - event.y()) ** 2 <= 16 ** 2:
                self.point_selected.emit(self.records[index]["sample_id"])
        super().mousePressEvent(event)


class ScanPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.run_id = None
        self.records, self.items_by_id = {}, {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.title = QLabel("Z 扫描 · Tenengrad")
        self.title.setTextFormat(Qt.PlainText)
        self.curve = FocusCurve()
        self.thumbnails = QListWidget()
        self.thumbnails.setViewMode(QListView.IconMode)
        self.thumbnails.setFlow(QListView.LeftToRight)
        self.thumbnails.setWrapping(False)
        self.thumbnails.setIconSize(QSize(100, 65))
        self.thumbnails.setFixedHeight(110)
        self.thumbnails.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.description = QLabel("未选择采样点")
        self.description.setTextFormat(Qt.PlainText)
        self.description.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.curve, stretch=1)
        layout.addWidget(self.thumbnails)
        layout.addWidget(self.description)
        self.curve.point_selected.connect(self.select_point)
        self.thumbnails.currentItemChanged.connect(self.on_item)

    def begin_run(self, payload):
        self.run_id = payload["run"]["run_id"]
        self.records.clear()
        self.items_by_id.clear()
        self.thumbnails.clear()
        self.curve.records = []
        self.curve.selected_id = None
        self.curve.update()
        self.title.setText(f"Tenengrad · {self.run_id} · 已保存 0 / {payload['run']['target_count']}")
        self.title.setToolTip(payload["run_dir"])
        self.target_count = payload["run"]["target_count"]
        self.description.setText("未选择采样点")

    def add_sample(self, payload):
        if payload["run_id"] != self.run_id:
            return
        record = payload["record"]
        sample_id = record["sample_id"]
        if sample_id in self.records:
            return
        self.records[sample_id] = record
        self.curve.records.append(record)
        rgb = cv.cvtColor(payload["thumbnail"], cv.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        item = QListWidgetItem(QIcon(QPixmap.fromImage(image)), f"{sample_id + 1} · Z {record['target_steps']:+d}")
        item.setData(Qt.UserRole, (self.run_id, sample_id))
        item.setToolTip(f"Tenengrad {record['focus_score']:.4f}\n采集时原图：" + payload["image_path"])
        self.thumbnails.addItem(item)
        self.items_by_id[sample_id] = item
        self.title.setText(f"Tenengrad · {self.run_id} · 已保存 {len(self.records)} / {self.target_count}")
        self.curve.update()

    def select_point(self, sample_id):
        if sample_id in self.items_by_id:
            self.thumbnails.setCurrentItem(self.items_by_id[sample_id])

    def on_item(self, item, previous):
        if item is None:
            return
        run_id, sample_id = item.data(Qt.UserRole)
        if run_id != self.run_id:
            return
        record = self.records[sample_id]
        self.curve.selected_id = sample_id
        self.curve.update()
        self.description.setText(f"采集记录 {sample_id + 1}：Z {record['target_steps']:+d} 脉冲；"
            f"Tenengrad {record['focus_score']:.4f}；质量未评价")
