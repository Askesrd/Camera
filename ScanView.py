# 曲线、图集和后台文件任务；不发运动或访问设备。
import copy
import json
import math
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2 as cv
from PyQt5.QtCore import QBuffer, QIODevice, QPointF, QRect, QRectF, QSignalBlocker, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QListView,
    QListWidget, QListWidgetItem, QMenu, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QToolButton, QVBoxLayout, QWidget)

from ScanStorage import export_run, image_path, read_history, read_image


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


def run_group(run):
    identity = [run.get(key) for key in ("connection_id", "z_reference_id", "view_context_id")]
    scale = [run.get(key) for key in ("pulses_per_rev", "lead_mm", "gear_ratio", "full_steps_per_rev")]
    if (any(not isinstance(v, str) or not v for v in identity)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in scale)):
        return "unknown:" + run["run_id"]
    # QComboBox 用简单字符串保存分组，避免混合类型元组在 Qt 中比较失败。
    return json.dumps(identity + [float(v) for v in scale], ensure_ascii=False)


class ScanFilesThread(QThread):
    """一个文件线程；预览/缩略图请求各保留最新一份，结果也只留一份。"""
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.jobs, self.replies = OrderedDict(), {}
        self.current = None
        self.progress = ""

    def submit(self, kind, payload):
        with self.condition:
            if self.stopping.is_set() or (kind in ("history", "export", "curve")
                                         and (kind == self.current or kind in self.jobs or kind in self.replies)):
                return False
            self.jobs[kind] = dict(payload) if kind == "curve" else copy.deepcopy(payload)
            self.condition.notify()
            return True

    def busy(self, kind):
        with self.condition:
            return self.current == kind or kind in self.jobs or (kind in ("history", "export", "curve") and kind in self.replies)

    def take_results(self):
        with self.condition:
            replies, self.replies = self.replies, {}
            progress, self.progress = self.progress, ""
            return replies, progress

    def set_progress(self, message):
        with self.condition:
            self.progress = message

    def stop(self):
        self.stopping.set()
        with self.condition:
            self.jobs.clear()
            self.condition.notify()

    def run(self):
        while not self.stopping.is_set():
            with self.condition:
                if not self.jobs:
                    self.condition.wait(0.1)
                    continue
                kind, payload = self.jobs.popitem(last=False)
                self.current = kind
            result = dict(payload)
            try:
                if kind == "history":
                    result.update(read_history(payload["directory"], self.stopping.is_set, self.set_progress))
                elif kind == "export":
                    result.update(export_run(payload["run_dir"], payload["run"], payload["records"],
                        payload["directory"], payload.get("interval"), self.stopping.is_set, self.set_progress))
                elif kind == "image":
                    frame = read_image(image_path(payload["run_dir"], payload["record"]["image_path"]),
                                       payload["run"].get("width"), payload["run"].get("height"))
                    height, width = frame.shape[:2]
                    scale = min(1200 / width, 900 / height, 1)
                    result.update(frame=cv.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale)))),
                                  image_size=(width, height))
                    del frame
                elif kind == "thumbnails":
                    thumbnails = []
                    for record in payload["records"]:
                        with self.condition:
                            replaced = "thumbnails" in self.jobs
                        if self.stopping.is_set() or replaced:
                            break
                        try:
                            original = image_path(payload["run_dir"], record["image_path"])
                            if not original.is_file():
                                raise FileNotFoundError("原图缺失")
                            thumb = image_path(payload["run_dir"], f"thumbs/s{record['sample_id']:06d}.jpg")
                            try:
                                frame = read_image(thumb)
                            except (OSError, ValueError, cv.error):
                                frame = read_image(original, payload["run"].get("width"), payload["run"].get("height"))
                            scale = min(160 / frame.shape[1], 120 / frame.shape[0], 1)
                            small = cv.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))))
                            thumbnails.append({"sample_id": record["sample_id"], "frame": small})
                            del frame
                        except (OSError, ValueError, cv.error) as error:
                            thumbnails.append({"sample_id": record["sample_id"], "error": str(error)})
                    result["thumbnails"] = thumbnails
                elif kind == "curve":
                    buffer = QBuffer()
                    buffer.open(QIODevice.WriteOnly)
                    if not payload["image"].save(buffer, "PNG"):
                        raise OSError("曲线图编码失败")
                    with Path(payload["path"]).open("xb") as stream:
                        stream.write(bytes(buffer.data()))
                    result.pop("image", None)
                else:
                    raise ValueError("未知文件任务")
            except Exception as error:
                result.pop("image", None)
                result["error"] = str(error)
            with self.condition:
                self.replies[kind] = result
                self.current = None


class FocusCurve(QWidget):
    point_selected = pyqtSignal(int)
    activated = pyqtSignal(str)
    interval_changed = pyqtSignal(object)
    view_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.records = []
        self.run_id = ""
        self.caption = "Z 扫描 · Tenengrad"
        self.active = True
        self.selected_id = self.interval = self.x_range = self.drag = None
        self.mode = "inspect"
        self.setFixedHeight(174)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setToolTip("单击激活曲线，再点击真实点查看图像；滚轮缩放，中键拖动；点击不发运动")

    def plot_area(self):
        return QRectF(65, 59, max(1, self.width() - 82), max(1, self.height() - 85))

    def limits(self):
        xs = [item["target_steps"] for item in self.records] or [0, 1]
        ys = [item["focus_score"] for item in self.records] or [0, 1]
        low_x, high_x = self.x_range or (min(xs), max(xs))
        if high_x <= low_x:
            low_x, high_x = low_x - 0.5, low_x + 0.5
        pad = max((max(ys) - min(ys)) * 0.1, max(abs(max(ys)), 1) * 0.01)
        return low_x, high_x, min(ys) - pad, max(ys) + pad

    def x_at(self, pixel):
        low, high, _, _ = self.limits()
        area = self.plot_area()
        return low + min(1, max(0, (pixel - area.left()) / area.width())) * (high - low)

    def point_positions(self):
        area = self.plot_area()
        low_x, high_x, low_y, high_y = self.limits()
        return [QPointF(area.left() + (r["target_steps"] - low_x) / (high_x - low_x) * area.width(),
                        area.bottom() - (r["focus_score"] - low_y) / (high_y - low_y) * area.height()) for r in self.records]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f5faff" if self.active else "#f5f5f5"))
        painter.setPen(QPen(QColor("#236ca4" if self.active else "#c4cbd2"), 2 if self.active else 1))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
        painter.setPen(QColor("#236ca4" if self.active else "#65717e"))
        painter.drawText(QRectF(7, 3, max(1, self.width() - 14), 53), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, self.caption)
        area = self.plot_area()
        low_x, high_x, low_y, high_y = self.limits()
        painter.setPen(QPen(QColor("#8a939e"), 1))
        painter.drawRect(area)
        painter.save()
        painter.setClipRect(area)
        if self.active and self.interval is not None:
            left, right = sorted(self.interval)
            left = area.left() + (left - low_x) / (high_x - low_x) * area.width()
            right = area.left() + (right - low_x) / (high_x - low_x) * area.width()
            painter.fillRect(QRectF(left, area.top(), right - left, area.height()), QColor(226, 151, 52, 55))
            painter.setPen(QPen(QColor("#bd7623"), 2))
            for x in (left, right):
                painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
        if self.records:
            points = self.point_positions()
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            color = QColor("#236ca4" if self.active else "#929da8")
            painter.setPen(QPen(color, 2))
            painter.drawPath(path)
            for record, point in zip(self.records, points):
                selected = record["sample_id"] == self.selected_id and self.active
                bad = record.get("image_status", "ok") != "ok"
                painter.setBrush(QColor("#d26920") if selected or bad else color)
                painter.drawEllipse(point, 4 if selected else 2.5, 4 if selected else 2.5)
        else:
            painter.drawText(area, Qt.AlignCenter, "等待已保存采样点")
        painter.restore()
        painter.setPen(QColor("#46515f"))
        painter.drawText(QRectF(1, area.top(), 60, 20), Qt.AlignRight, f"{high_y:.4g}")
        painter.drawText(QRectF(1, area.bottom() - 20, 60, 20), Qt.AlignRight, f"{low_y:.4g}")
        painter.drawText(QRectF(area.left(), area.bottom() + 3, 65, 20), Qt.AlignLeft, f"{low_x:.6g}")
        painter.drawText(QRectF(area.right() - 65, area.bottom() + 3, 65, 20), Qt.AlignRight, f"{high_x:.6g}")
        painter.drawText(QRectF(area.left() + 65, area.bottom() + 3, max(1, area.width() - 130), 20), Qt.AlignCenter, "Z 指令 / 脉冲")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.active:
            self.activated.emit(self.run_id)
            return  # 第一次只激活，不能顺便选点或选区。
        if not self.active or not self.plot_area().contains(event.pos()):
            return
        if event.button() == Qt.MiddleButton:
            self.drag = ("pan", event.x(), self.limits()[:2])
        elif event.button() == Qt.LeftButton and self.mode == "range":
            x = self.x_at(event.x())
            anchor = x
            if self.interval is not None:
                low, high = sorted(self.interval)
                tolerance = (self.limits()[1] - self.limits()[0]) * 10 / self.plot_area().width()
                if abs(x - low) <= tolerance:
                    anchor = high
                elif abs(x - high) <= tolerance:
                    anchor = low
            self.drag = ("range", anchor)
            self.interval_changed.emit((anchor, x))
        elif event.button() == Qt.LeftButton and self.records:
            points = self.point_positions()
            index = min(range(len(points)), key=lambda i: (points[i].x() - event.x()) ** 2 + (points[i].y() - event.y()) ** 2)
            if (points[index].x() - event.x()) ** 2 + (points[index].y() - event.y()) ** 2 <= 10 ** 2:
                self.point_selected.emit(self.records[index]["sample_id"])

    def mouseMoveEvent(self, event):
        if not self.drag or not self.active:
            return
        if self.drag[0] == "range":
            self.interval_changed.emit((self.drag[1], self.x_at(event.x())))
        else:
            low, high = self.drag[2]
            delta = (event.x() - self.drag[1]) * (high - low) / self.plot_area().width()
            self.view_changed.emit((low - delta, high - delta))

    def mouseReleaseEvent(self, event):
        if self.drag:
            self.mouseMoveEvent(event)
        self.drag = None

    def wheelEvent(self, event):
        if self.active and self.plot_area().contains(event.pos()):
            low, high, _, _ = self.limits()
            center = self.x_at(event.x())
            factor = 0.8 if event.angleDelta().y() > 0 else 1.25
            if 1 <= (high - low) * factor <= 1e10:
                self.view_changed.emit((center + (low - center) * factor, center + (high - center) * factor))
                event.accept()


class ScanPanel(QWidget):
    selection_changed = pyqtSignal()
    PAGE_SIZE = 48
    THUMB_CACHE_SIZE = 128

    def __init__(self):
        super().__init__()
        self.runs, self.items_by_id = {}, {}
        self.active_run_id = self.run_id = self.selected_sample_id = self.selected_interval = None
        self.records = {}
        self.run_order, self.x_ranges = [], {}
        self.thumbnail_cache = OrderedDict()
        self.page = self.gallery_generation = self.image_generation = 0
        self.history_frame = None
        self.closing = False
        self.file_thread = ScanFilesThread()
        self.file_timer = QTimer(self)
        self.file_timer.setInterval(40)
        self.file_timer.timeout.connect(self.poll_files)
        self.file_timer.start()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        toolbar = QHBoxLayout()
        self.btn_history = QPushButton("加载历史…")
        self.btn_history.clicked.connect(lambda: self.choose_history())
        self.group_box = QComboBox()
        self.group_box.setMinimumWidth(80)
        self.group_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.group_box.currentIndexChanged.connect(self.change_group)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["查看", "选区"])
        self.mode_box.currentIndexChanged.connect(self.change_mode)
        self.btn_actions = QToolButton()
        self.btn_actions.setText("图集 / 导出")
        self.btn_actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_actions)
        self.export_action = menu.addAction("导出当前图集与数据…", lambda: self.choose_export())
        self.curve_action = menu.addAction("保存本组曲线图…", self.save_curves)
        menu.addSeparator()
        menu.addAction("清除区间（整条曲线）", lambda: self.set_interval(None))
        menu.addAction("显示完整横轴", self.reset_view)
        self.btn_actions.setMenu(menu)
        self.btn_expand = QToolButton()
        self.btn_expand.setText("展开")
        self.btn_expand.setCheckable(True)
        self.btn_expand.setToolTip("收起实时预览，扩大扫描记录区；再次点击还原")
        self.btn_expand.toggled.connect(lambda checked: self.btn_expand.setText("还原" if checked else "展开"))
        for widget in (self.btn_history, self.group_box, self.mode_box, self.btn_expand, self.btn_actions):
            toolbar.addWidget(widget, 1 if widget is self.group_box else 0)
        layout.addLayout(toolbar)
        self.title = QLabel("Z 扫描记录 · 尚未加载")
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.title.setTextFormat(Qt.PlainText)
        layout.addWidget(self.title)
        self.curves_host = QWidget()
        self.curves_layout = QVBoxLayout(self.curves_host)
        self.curves_layout.setContentsMargins(2, 2, 2, 2)
        self.curves_layout.setSpacing(3)
        self.curve = FocusCurve()
        self.curves_layout.addWidget(self.curve)
        self.curves_layout.addStretch()
        self.curves_scroll = QScrollArea()
        self.curves_scroll.setWidgetResizable(True)
        self.curves_scroll.setMinimumWidth(260)
        self.curves_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.curves_scroll.setWidget(self.curves_host)
        history = QWidget()
        history_layout = QVBoxLayout(history)
        history_layout.setContentsMargins(2, 0, 0, 0)
        self.history_caption = QLabel("历史图像（静态）")
        self.history_caption.setTextFormat(Qt.PlainText)
        self.history_caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.history_image = QLabel("点击曲线点或缩略图查看")
        self.history_image.setAlignment(Qt.AlignCenter)
        self.history_image.setStyleSheet("background: #15191e; color: #d8dee7;")
        self.history_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.history_image.setMinimumSize(80, 50)
        history_layout.addWidget(self.history_caption)
        history_layout.addWidget(self.history_image, stretch=1)
        self.detail_splitter = QSplitter(Qt.Horizontal)
        self.detail_splitter.addWidget(self.curves_scroll)
        self.detail_splitter.addWidget(history)
        history.setMinimumWidth(180)
        self.detail_splitter.setStretchFactor(0, 3)
        self.detail_splitter.setStretchFactor(1, 2)
        self.detail_splitter.setSizes([540, 360])
        self.detail_splitter.splitterMoved.connect(lambda *args: self.show_history_image())
        layout.addWidget(self.detail_splitter, stretch=1)
        self.thumbnails = QListWidget()
        self.thumbnails.setViewMode(QListView.IconMode)
        self.thumbnails.setFlow(QListView.LeftToRight)
        self.thumbnails.setWrapping(False)
        self.thumbnails.setIconSize(QSize(80, 48))
        self.thumbnails.setFixedHeight(83)
        self.thumbnails.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gallery = QHBoxLayout()
        self.btn_previous, self.btn_next = QPushButton("‹"), QPushButton("›")
        for button in (self.btn_previous, self.btn_next):
            button.setFixedWidth(24)
        self.btn_previous.setToolTip("上一页缩略图")
        self.btn_next.setToolTip("下一页缩略图")
        self.btn_previous.clicked.connect(lambda: self.refresh_thumbnails(self.page - 1))
        self.btn_next.clicked.connect(lambda: self.refresh_thumbnails(self.page + 1))
        gallery.addWidget(self.btn_previous)
        gallery.addWidget(self.thumbnails, stretch=1)
        gallery.addWidget(self.btn_next)
        layout.addLayout(gallery)
        self.description = QLabel("未选择采样点")
        self.description.setTextFormat(Qt.PlainText)
        self.description.setWordWrap(True)
        self.description.setFixedHeight(self.description.fontMetrics().lineSpacing() * 2 + 2)
        self.description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.file_status = QLabel("")
        self.file_status.setTextFormat(Qt.PlainText)
        self.file_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addWidget(self.description)
        actions = QHBoxLayout()
        self.btn_revisit = QPushButton("返回此位置")
        self.btn_fine_scan = QPushButton("选区新建扫描")
        for button in (self.btn_revisit, self.btn_fine_scan):
            button.setEnabled(False)
            button.setFixedWidth(115)
            actions.addWidget(button)
        self.action_hint = QLabel("请选择采样点或区间")
        self.action_hint.setTextFormat(Qt.PlainText)
        self.action_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        actions.addWidget(self.action_hint, stretch=1)
        layout.addLayout(actions)
        layout.addWidget(self.file_status)
        self.thumbnails.currentItemChanged.connect(self.on_item)
        self.update_selection()

    def queue_file(self, kind, payload):
        if self.closing or not self.file_thread.submit(kind, payload):
            return False
        if not self.file_thread.isRunning():
            self.file_thread.start()
        return True

    def begin_run(self, payload, activate=True, live=True):
        run = copy.deepcopy(payload["run"])
        run_id = run["run_id"]
        if self.closing or run_id in self.runs:
            return False
        records = {r["sample_id"]: copy.deepcopy(r) for r in payload.get("records", [])}
        curve = FocusCurve()
        curve.run_id, curve.records = run_id, list(records.values())
        curve.active = False
        curve.activated.connect(self.activate_run)
        curve.point_selected.connect(lambda sample_id, rid=run_id: self.select_point(sample_id) if rid == self.active_run_id else None)
        curve.interval_changed.connect(lambda interval, rid=run_id: self.set_interval(interval) if rid == self.active_run_id else None)
        curve.view_changed.connect(lambda limits, rid=run_id: self.set_view(limits) if rid == self.active_run_id else None)
        try:
            created = datetime.fromisoformat(run.get("created_at", "")).timestamp()
        except (ValueError, TypeError):
            created = datetime.now().timestamp() if live else 0
        self.runs[run_id] = {"run": run, "run_dir": payload["run_dir"], "records": records,
                             "curve": curve, "group": run_group(run), "created": created,
                             "issues": payload.get("issues", []), "live": live}
        self.run_order.append(run_id)
        self.run_order.sort(key=lambda rid: self.runs[rid]["created"], reverse=True)
        self.curves_layout.insertWidget(self.run_order.index(run_id), curve)
        if len(self.runs) == 1:
            placeholder = self.curve
            self.curves_layout.removeWidget(placeholder)
            placeholder.deleteLater()
        self.update_caption(run_id)
        self.rebuild_groups()
        if activate or self.active_run_id is None:
            self.activate_run(run_id)
        else:
            self.show_group()
        return True

    def update_caption(self, run_id):
        entry = self.runs[run_id]
        run = entry["run"]
        state = {"running": "采集中", "completed": "完成", "cancelled": "已取消",
                 "failed": "失败", "interrupted": "中断记录", "preparing": "准备中"}.get(run.get("status"), "记录")
        if "export_source_status" in run:
            state = "导出副本" if run.get("status") == "completed" else "未完成导出"
        exposure = run.get("exposure_readback_us")
        exposure = f"{exposure / 1000:g} ms" if type(exposure) in (int, float) else "曝光未记录"
        roi = run.get("roi")
        roi = "全图" if roi is None else ("ROI " + ",".join(str(v) for v in roi) if isinstance(roi, list) and len(roi) == 4 else "ROI 记录异常")
        name = str(run.get("sample_name") or "")
        name = name[:14] + "…" if len(name) > 14 else name
        short_id = run_id if len(run_id) <= 18 else run_id[:7] + "…" + run_id[-4:]
        count = run.get("export_selected_count", run.get("target_count", len(entry["records"])))
        entry["curve"].caption = (f"{short_id} {name} · {state} · {len(entry['records'])}/{count} 点\n"
            f"{run.get('focus_algorithm', 'Tenengrad')} · {exposure} · 增益 {run.get('gain_readback_percent', '—')}% · "
            f"间隔 {run.get('interval_steps', '—')} 脉冲 · {roi}")
        entry["curve"].setToolTip(entry["curve"].caption + f"\n{run_id}\n{entry['run_dir']}\n"
            + "\n".join(entry["issues"]) + "\n滚轮缩放，中键拖动；单击不发运动")
        entry["curve"].update()

    def rebuild_groups(self):
        current = self.runs[self.active_run_id]["group"] if self.active_run_id in self.runs else None
        groups = list(dict.fromkeys(self.runs[rid]["group"] for rid in self.run_order))
        with QSignalBlocker(self.group_box):
            self.group_box.clear()
            for group in groups:
                count = sum(entry["group"] == group for entry in self.runs.values())
                run = next(entry["run"] for entry in self.runs.values() if entry["group"] == group)
                name = (f"参考 {run['z_reference_id'][:8]} · 视野 {run['view_context_id'][:6]}"
                        if not group.startswith("unknown:") else f"参考未完整记录 · {run['run_id']}")
                self.group_box.addItem(f"{name} · {count} 次", group)
            if current in groups:
                self.group_box.setCurrentIndex(groups.index(current))

    def change_group(self, index):
        group = self.group_box.itemData(index)
        for run_id in self.run_order:
            if self.runs[run_id]["group"] == group:
                if run_id != self.active_run_id:
                    self.activate_run(run_id)
                break

    def show_group(self):
        group = self.group_box.currentData()
        for rid in self.run_order:
            curve = self.runs[rid]["curve"]
            curve.setVisible(self.runs[rid]["group"] == group)
        self.set_view(self.x_ranges.get(group), automatic=True)

    def activate_run(self, run_id):
        if run_id not in self.runs or self.closing:
            return
        self.active_run_id = self.run_id = run_id
        self.selected_sample_id = self.selected_interval = None
        self.image_generation += 1
        self.history_frame = None
        self.history_caption.setText("历史图像（静态）")
        self.history_image.clear()
        self.history_image.setText("点击曲线点或缩略图查看")
        self.description.setText("未选择采样点")
        entry = self.runs[run_id]
        self.records, self.curve = entry["records"], entry["curve"]
        self.target_count = entry["run"].get("target_count", len(self.records))
        for rid, other in self.runs.items():
            other["curve"].active = rid == run_id
            other["curve"].selected_id = other["curve"].interval = other["curve"].drag = None
            other["curve"].mode = "range" if self.mode_box.currentIndex() else "inspect"
            other["curve"].update()
        with QSignalBlocker(self.group_box):
            self.group_box.setCurrentIndex(self.group_box.findData(entry["group"]))
        self.show_group()
        self.refresh_thumbnails(0)
        self.curves_scroll.ensureWidgetVisible(self.curve)
        self.update_selection()

    def set_view(self, limits, automatic=False):
        group = self.group_box.currentData()
        entries = [self.runs[rid] for rid in self.run_order if self.runs[rid]["group"] == group]
        if not entries:
            return
        if limits is None:
            xs = [r["target_steps"] for entry in entries for r in entry["records"].values()]
            xs.extend(value for entry in entries for value in (entry["run"].get("a_steps"), entry["run"].get("b_steps")) if type(value) is int)
            low, high = (min(xs), max(xs)) if xs else (0, 1)
            pad = max((high - low) * 0.02, 0.5)
            limits = (low - pad, high + pad)
        if not automatic:
            self.x_ranges[group] = tuple(limits)
        for entry in entries:
            entry["curve"].x_range = tuple(limits)
            entry["curve"].update()

    def reset_view(self):
        self.x_ranges.pop(self.group_box.currentData(), None)
        self.set_view(None, automatic=True)

    def change_mode(self, index):
        for entry in self.runs.values():
            entry["curve"].mode = "range" if index else "inspect"
            entry["curve"].drag = None

    def set_interval(self, interval):
        if self.active_run_id is None:
            return
        self.selected_interval = tuple(sorted(interval)) if interval is not None else None
        self.curve.interval = self.selected_interval
        self.curve.update()
        self.update_selection()

    def selected_records(self):
        records = list(self.records.values())
        if self.selected_interval is not None:
            low, high = self.selected_interval
            records = [r for r in records if low <= r["target_steps"] <= high]
        return records

    def update_selection(self):
        records = self.selected_records()
        entry = self.runs.get(self.active_run_id)
        self.export_action.setEnabled(bool(records) and not self.file_thread.busy("export") and not self.closing)
        self.curve_action.setEnabled(bool(entry) and not self.file_thread.busy("curve") and not self.closing)
        if entry:
            bad = sum(r.get("image_status", "ok") != "ok" for r in records)
            scope = "整条" if self.selected_interval is None else f"区间 {self.selected_interval[0]:g}…{self.selected_interval[1]:g} 脉冲"
            self.title.setText(f"当前 {self.active_run_id} · {scope} · {len(records)} 点 · 异常图 {bad}")
            self.title.setToolTip(self.title.text() + "\n" + entry["run_dir"] + "\n" + "\n".join(entry["issues"]))
        self.selection_changed.emit()

    def cache_thumbnail(self, run_id, sample_id, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        key = (run_id, sample_id)
        self.thumbnail_cache[key] = QIcon(QPixmap.fromImage(image))
        self.thumbnail_cache.move_to_end(key)
        while len(self.thumbnail_cache) > self.THUMB_CACHE_SIZE:
            self.thumbnail_cache.popitem(last=False)

    def refresh_thumbnails(self, page=None):
        self.gallery_generation += 1
        self.page = min(max(0, self.page if page is None else page), max(0, (len(self.records) - 1) // self.PAGE_SIZE))
        records = list(self.records.values())[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]
        self.items_by_id.clear()
        missing = []
        with QSignalBlocker(self.thumbnails):
            self.thumbnails.clear()
            for record in records:
                sample_id = record["sample_id"]
                key = (self.active_run_id, sample_id)
                bad = record.get("image_status", "ok") != "ok"
                item = QListWidgetItem(self.thumbnail_cache.get(key, QIcon()),
                    f"{'! ' if bad else ''}{sample_id + 1} · Z {record['target_steps']:+d}")
                item.setData(Qt.UserRole, key)
                item.setToolTip(f"分数 {record['focus_score']:.4f}\n{record.get('image_error', '')}\n{record.get('image_path', '')}")
                self.thumbnails.addItem(item)
                self.items_by_id[sample_id] = item
                if sample_id == self.selected_sample_id:
                    self.thumbnails.setCurrentItem(item)
                if key not in self.thumbnail_cache and not bad and record.get("image_path"):
                    missing.append(record)
        self.btn_previous.setEnabled(self.page > 0)
        self.btn_next.setEnabled((self.page + 1) * self.PAGE_SIZE < len(self.records))
        if missing and self.active_run_id in self.runs:
            entry = self.runs[self.active_run_id]
            self.queue_file("thumbnails", {"run_id": self.active_run_id, "generation": self.gallery_generation,
                "run": entry["run"], "run_dir": entry["run_dir"], "records": missing})

    def add_sample(self, payload):
        run_id, record = payload["run_id"], payload["record"]
        if self.closing or run_id not in self.runs:
            return
        entry = self.runs[run_id]
        sample_id = record["sample_id"]
        if sample_id in entry["records"]:
            return
        entry["records"][sample_id] = copy.deepcopy(record)
        entry["curve"].records.append(entry["records"][sample_id])
        self.cache_thumbnail(run_id, sample_id, payload["thumbnail"])
        self.update_caption(run_id)
        self.show_group()
        if run_id == self.active_run_id:
            self.refresh_thumbnails()
            self.update_selection()

    def finish_run(self, result):
        entry = self.runs.get(result["run_id"])
        if entry:
            entry["run"].update(status=result["status"], committed_count=result["committed_count"], error=result.get("error"))
            entry["live"] = False
            self.update_caption(result["run_id"])

    def select_point(self, sample_id):
        if sample_id not in self.records:
            return
        page = list(self.records).index(sample_id) // self.PAGE_SIZE
        if page != self.page:
            self.refresh_thumbnails(page)
        item = self.items_by_id.get(sample_id)
        if item is not None:
            self.thumbnails.setCurrentItem(item)
            self.thumbnails.scrollToItem(item)

    def on_item(self, item, previous):
        if item is None:
            return
        run_id, sample_id = item.data(Qt.UserRole)
        if run_id != self.active_run_id or sample_id not in self.records:
            return
        record = self.records[sample_id]
        self.selected_sample_id = self.curve.selected_id = sample_id
        self.curve.update()
        run = self.runs[run_id]["run"]
        quality = {"not_assessed": "质量未评价", "unstable": "评分波动"}.get(record.get("quality_status", "not_assessed"), "质量状态未识别")
        candidates = record.get("candidates") or []
        index = record.get("representative_index")
        representative = candidates[index] if type(index) is int and 0 <= index < len(candidates) else {}
        exposure = representative.get("exposure_us", run.get("exposure_readback_us"))
        exposure = f"{exposure / 1000:g} ms" if type(exposure) in (int, float) else "未记录"
        gain = representative.get("gain_percent", run.get("gain_readback_percent", "—"))
        captured = representative.get("host_received_utc")
        try:
            captured_time = datetime.fromisoformat(captured)
            captured = (captured_time.astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                        if captured_time.tzinfo is not None else "时区未记录")
        except (ValueError, TypeError):
            captured = "时间未记录"
        estimate = record.get("estimated_z_um")
        estimate = f"；理论 {estimate:g} μm" if type(estimate) in (int, float) else ""
        self.description.setText(f"采集记录 {sample_id + 1}：Z {record['target_steps']:+d} 脉冲{estimate}；"
            f"{run.get('focus_algorithm', 'Tenengrad')} {record['focus_score']:.4f}\n"
            f"曝光 {exposure} · 增益 {gain}% · 采集 {captured} · {quality}")
        self.description.setToolTip(self.description.text() + f"\nRun：{run_id}\nROI：{run.get('roi')}\n"
            f"曝光：{run.get('exposure_readback_us')} μs；增益：{run.get('gain_readback_percent')}%\n"
            f"Run 开始：{run.get('started_at', run.get('created_at', '未记录'))}\n"
            f"帧接收 UTC：{representative.get('host_received_utc', '未记录')}\n"
            f"采集方式：{run.get('freshness_method', '未记录')}；时序验证：{run.get('freshness_verified', '未记录')}\n"
            f"停稳单调时刻：{record.get('settled_monotonic_ns', '未记录')} ns\n"
            f"图像状态：{record.get('image_status', 'ok')} {record.get('image_error', '')}")
        self.image_generation += 1
        self.history_frame = None
        self.history_image.clear()
        self.history_caption.setText(f"历史图像（静态）· 记录 {sample_id + 1}")
        if record.get("image_status", "ok") != "ok":
            self.history_image.setText(record.get("image_error") or "原图缺失 / 损坏")
        elif record.get("image_path"):
            self.history_image.setText("正在加载历史原图…")
            entry = self.runs[run_id]
            self.queue_file("image", {"run_id": run_id, "generation": self.image_generation,
                "run": run, "run_dir": entry["run_dir"], "record": record})
        else:
            self.history_image.setText("未记录原图路径")
        self.selection_changed.emit()

    def choose_history(self, initial=""):
        directory = QFileDialog.getExistingDirectory(self, "选择 Session、Run 或扫描数据目录", initial)
        return self.load_history(directory) if directory else False

    def load_history(self, directory):
        accepted = self.queue_file("history", {"directory": str(directory)})
        self.file_status.setText("正在读取历史记录并检查原图…" if accepted else "上一次历史加载尚未完成")
        return accepted

    def choose_export(self):
        if not self.selected_records():
            return
        directory = QFileDialog.getExistingDirectory(self, "选择导出位置（自动新建目录）")
        if directory:
            self.start_export(directory)

    def start_export(self, directory):
        if self.active_run_id is None or not self.selected_records():
            return False
        entry = self.runs[self.active_run_id]
        accepted = self.queue_file("export", {"run": entry["run"], "run_dir": entry["run_dir"],
            "records": self.selected_records(), "directory": str(directory), "interval": self.selected_interval})
        if accepted:
            self.file_status.setText("正在导出已固定的图集与数据…")
        self.update_selection()
        return accepted

    def save_curves(self):
        if self.active_run_id is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存本组曲线图", "Z曲线.png", "PNG 图像 (*.png)")
        if not path:
            return
        if self.curves_host.width() * self.curves_host.height() > 16000000:
            self.file_status.setText("本组曲线图过大，请改为导出数值数据")
            return
        entries = [self.runs[rid] for rid in self.run_order if self.runs[rid]["group"] == self.group_box.currentData()]
        notes = "Z 指令位置 / 脉冲；纵轴为各次扫描的原始清晰度分数\n" + "\n".join(
            entry["run"]["run_id"] + "\n" + entry["curve"].caption.split("\n", 1)[-1] for entry in entries)
        chart = self.curves_host.grab().toImage()
        text_height = self.fontMetrics().boundingRect(QRect(0, 0, max(1, chart.width() - 16), 100000), Qt.TextWordWrap, notes).height()
        if chart.width() * (chart.height() + text_height + 16) > 16000000:
            self.file_status.setText("本组曲线图过大，请改为导出数值数据")
            return
        image = QImage(chart.width(), chart.height() + text_height + 16, QImage.Format_RGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.drawImage(0, 0, chart)
        painter.setFont(self.font())
        painter.setPen(QColor("#293849"))
        painter.drawText(QRect(8, chart.height() + 8, max(1, chart.width() - 16), text_height), Qt.TextWordWrap, notes)
        painter.end()
        image.setText("Run IDs", "\n".join(entry["run"]["run_id"] for entry in entries))
        image.setText("X unit", "Z command position (pulses)")
        self.queue_file("curve", {"path": path, "image": image})
        self.update_selection()

    def poll_files(self):
        if self.closing:
            return
        replies, progress = self.file_thread.take_results()
        if progress:
            self.file_status.setText(progress)
        for kind, result in replies.items():
            if kind == "history" and not result.get("error"):
                loaded = 0
                previous_active = self.active_run_id
                issues = list(result["issues"])
                for payload in sorted(result["runs"], key=lambda p: str(p["run"].get("created_at", ""))):
                    run_id = payload["run"]["run_id"]
                    if run_id in self.runs and Path(self.runs[run_id]["run_dir"]).resolve() != Path(payload["run_dir"]).resolve():
                        issues.append(f"{run_id} 身份重复但目录不同，保留已加载的数据")
                        continue
                    loaded += self.begin_run(payload, activate=False, live=False)
                    issues.extend(f"{run_id}：{issue}" for issue in payload["issues"])
                if previous_active is None and loaded:
                    self.activate_run(self.run_order[0])
                self.file_status.setText(f"加载 {loaded} 次扫描 · {len(issues)} 条异常" + ("（详见提示）" if issues else ""))
                self.file_status.setToolTip("\n".join(issues) or result["directory"])
            elif kind == "thumbnails":
                if result.get("run_id") != self.active_run_id or result.get("generation") != self.gallery_generation:
                    continue
                for small in result.get("thumbnails", []):
                    sample_id = small["sample_id"]
                    item = self.items_by_id.get(sample_id)
                    if item is None:
                        continue
                    if small.get("error"):
                        self.records[sample_id].update(image_status="unavailable", image_error=small["error"])
                        item.setText("! " + item.text())
                        item.setToolTip(small["error"])
                        self.curve.update()
                    else:
                        self.cache_thumbnail(self.active_run_id, sample_id, small["frame"])
                        item.setIcon(self.thumbnail_cache[(self.active_run_id, sample_id)])
            elif kind == "image":
                if result.get("run_id") != self.active_run_id or result.get("generation") != self.image_generation:
                    continue
                if result.get("error"):
                    self.history_image.setText("无法读取原图：" + result["error"])
                    record = self.records.get(self.selected_sample_id)
                    if record is not None:
                        record.update(image_status="unavailable", image_error=result["error"])
                        self.curve.update()
                else:
                    frame = cv.cvtColor(result["frame"], cv.COLOR_BGR2RGB)
                    self.history_frame = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888).copy()
                    self.history_caption.setText(f"历史原图（静态）· {result['image_size'][0]}×{result['image_size'][1]}")
                    self.show_history_image()
            elif kind == "export" and result.get("export_dir"):
                state = {"completed": "完成", "failed": "有失败项", "cancelled": "已取消"}.get(result["status"], "未完成")
                self.file_status.setText(f"导出{state} · 已复制 {len(result['copied_sample_ids'])} / {len(result['selected_sample_ids'])} 张")
                self.file_status.setToolTip(result["export_dir"] + "\n" + result.get("error", "")
                    + "\n" + "\n".join(str(f) for f in result["failures"]))
            elif kind == "curve" and not result.get("error"):
                self.file_status.setText("曲线图已保存")
                self.file_status.setToolTip(result["path"])
            else:
                self.file_status.setText("文件操作未完成：" + result.get("error", "未知错误"))
                self.file_status.setToolTip(result.get("error", ""))
        self.btn_history.setEnabled(not self.file_thread.busy("history"))
        if replies:
            self.update_selection()

    def show_history_image(self):
        if self.history_frame is not None:
            self.history_image.setPixmap(QPixmap.fromImage(self.history_frame).scaled(self.history_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_history_image()

    def shutdown(self):
        self.closing = True
        self.file_timer.stop()
        self.file_thread.stop()

    def closeEvent(self, event):
        self.shutdown()
        if self.file_thread.isRunning():
            event.ignore()
            QTimer.singleShot(40, self.close)
        else:
            event.accept()
