# main.py
# GUI：左侧操作，右上状态，右下图像。阻塞的硬件操作仍放在工作线程里。

import json
import sys
import time
import copy
from pathlib import Path

import cv2 as cv
from PyQt5.QtCore import QEvent, QRect, QSignalBlocker, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
    QRubberBand, QSplitter, QWidget,
)
from serial.tools import list_ports

from Camera import CameraThread, SAVE_DIR
from Laplacian import LaplacianThread
from Control import (
    ControlThread, MAX_XY_MOVE_MM, MAX_Z_MOVE_STEPS,
)
from Scan import ScanThread, make_scan_plan
from ScanView import ScanPanel, roi_from_drag


SETTINGS_PATH = Path(__file__).with_name("settings.json")


class MainWindow(QMainWindow):
    def __init__(self, mode, xy_port="", z_port=""):
        super().__init__()
        self.camera_thread = CameraThread(mode)
        self.laplacian_thread = LaplacianThread()
        self.control_thread = ControlThread(xy_port, z_port)
        self.scan_thread = ScanThread(self.control_thread, self.camera_thread)
        self.scan_active = False
        self.workers_stopping = False
        self.camera_diagnostic_data = {}
        self.current_frame = None
        self.roi = None
        self.roi_start = None
        self.last_marks = None
        self.xy_ready = False
        self.z_ready = False
        self.control_busy = True
        self.closing = False
        self.jog_button = None
        self.jog_needs_press = False
        self.motion_buttons = []
        self.axis_inputs = {}
        self.speed_inputs = {}
        self.exposure_connected = False
        self.exposure_request_id = 0
        self.gain_connected = False
        self.gain_request_id = 0
        self.camera_locked = False
        self.z_snapshot = {}
        self.jog_timer = QTimer(self)
        self.jog_timer.setInterval(80)
        self.jog_timer.timeout.connect(self.keep_jog)

        self.setWindowTitle("显微成像与 XYZ 控制")
        self.resize(1240, 860)
        self.setMinimumSize(960, 640)
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton { min-height: 22px; }
            QDoubleSpinBox, QSpinBox { min-height: 24px; }
        """)

        self.video_label = QLabel("等待相机图像")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #15191e; color: #d8dee7;")
        self.video_label.setMinimumSize(400, 140)
        # 忽略上一张图片的尺寸建议，缩放窗口时图像区域才不会越撑越大。
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMouseTracking(True)
        self.roi_band = QRubberBand(QRubberBand.Rectangle, self.video_label)

        self.status_label = QLabel("控制：正在初始化")
        self.camera_status_label = QLabel("相机：启动中")
        self.info_label = QLabel("相机信息：等待数据")
        self.focus_label = QLabel("等待数据")
        self.focus_label.setAlignment(Qt.AlignCenter)
        self.focus_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #205782;")
        self.control_label = QLabel("XY：等待连接\nZ：等待连接")
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(4)
        for label in (self.status_label, self.camera_status_label, self.info_label,
                      self.control_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            status_layout.addWidget(label)

        # 左侧上下箭头、中间可逐位编辑的毫秒数、右侧自动曝光勾选框。
        self.exposure_box = QDoubleSpinBox()
        self.exposure_box.setObjectName("exposureBox")
        self.exposure_box.setDecimals(3)  # 1 微秒 = 0.001 毫秒
        self.exposure_box.setRange(0, 0)
        self.exposure_box.setSpecialValueText("等待相机")
        self.exposure_box.setSuffix(" ms")
        self.exposure_box.setSingleStep(1.0)
        self.exposure_box.setAccelerated(True)
        self.exposure_box.setKeyboardTracking(False)  # 输入途中不提交，也不改变光标位置
        self.exposure_box.setAlignment(Qt.AlignCenter)
        self.exposure_box.setFixedWidth(230)
        self.exposure_box.setMinimumHeight(38)
        self.exposure_box.setEnabled(False)
        self.exposure_box.setToolTip("单击调整 1 ms，长按加速；直接编辑后按回车或移开焦点应用")
        self.exposure_box.setStyleSheet("""
            QDoubleSpinBox#exposureBox { padding-left: 32px; padding-right: 5px; font-size: 16px; }
            QDoubleSpinBox#exposureBox:disabled { color: #586270; background: #f2f4f7; }
            QDoubleSpinBox#exposureBox::up-button {
                subcontrol-origin: border; subcontrol-position: top left; width: 28px;
            }
            QDoubleSpinBox#exposureBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom left; width: 28px;
            }
        """)
        self.auto_exposure = QCheckBox("自动曝光")
        self.auto_exposure.setChecked(True)
        self.auto_exposure.setEnabled(False)
        self.auto_exposure.toggled.connect(self.change_exposure)
        self.exposure_box.valueChanged.connect(self.change_exposure)
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("曝光时间"))
        exposure_layout.addWidget(self.exposure_box)
        exposure_layout.addWidget(self.auto_exposure)
        exposure_layout.addStretch()
        status_layout.addLayout(exposure_layout)

        self.gain_box = QSpinBox()
        self.gain_box.setSuffix(" %")
        self.gain_box.setRange(0, 0)
        self.gain_box.setSpecialValueText("等待相机")
        self.gain_box.setKeyboardTracking(False)
        self.gain_box.setEnabled(False)
        self.gain_box.setToolTip("相机增益百分数，以设备范围和回读为准；关闭自动曝光后可设置")
        self.gain_box.valueChanged.connect(self.change_gain)
        self.camera_diagnostic = QLabel("相机诊断：等待设备；新帧时序未验证")
        self.camera_diagnostic.setTextFormat(Qt.PlainText)
        self.camera_diagnostic.setWordWrap(True)
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("增益"))
        gain_layout.addWidget(self.gain_box)
        gain_layout.addWidget(self.camera_diagnostic, stretch=1)
        status_layout.addLayout(gain_layout)

        control_panel = QWidget()
        control_panel.setFixedWidth(280)
        controls = QVBoxLayout(control_panel)
        controls.setContentsMargins(8, 4, 8, 4)
        units_label = QLabel("XYZ 控制\nXY：距离（mm） · Z：步数")
        units_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        controls.addWidget(units_label)

        # 参照 ASI Console：八方向长按区，下面分别设置 X/Y 速度。
        xy_group = QGroupBox("XY  /  ASI")
        xy_layout = QVBoxLayout(xy_group)
        xy_layout.setSpacing(4)
        arrows = QGridLayout()
        for row, column, text, dx, dy in (
            (0, 0, "↖", -1, 1), (0, 1, "↑", 0, 1), (0, 2, "↗", 1, 1),
            (1, 0, "←", -1, 0),                         (1, 2, "→", 1, 0),
            (2, 0, "↙", -1, -1), (2, 1, "↓", 0, -1), (2, 2, "↘", 1, -1),
        ):
            button = QPushButton(text)
            button.setFixedHeight(40)
            button.setStyleSheet("font-size: 28px; font-weight: bold; padding: 0;")
            button.setAutoRepeat(False)
            direction = " / ".join(f"{a}{'+' if d > 0 else '−'}" for a, d in (("X", dx), ("Y", dy)) if d)
            button.setToolTip(f"长按 {direction}，松开停止；左 X−，上 Y+")
            button.setAccessibleName(f"长按 {direction}")
            button.pressed.connect(lambda d=(dx, dy), b=button: self.start_jog("XY", d, b))
            button.released.connect(self.end_jog)
            arrows.addWidget(button, row, column)
            self.motion_buttons.append((button, "XY"))
        xy_center = QLabel("XY")
        xy_center.setAlignment(Qt.AlignCenter)
        arrows.addWidget(xy_center, 1, 1)
        xy_layout.addLayout(arrows)
        speed_title = QLabel("Speed  /  mm/s")
        speed_title.setAlignment(Qt.AlignCenter)
        xy_layout.addWidget(speed_title)
        speed_grid = QGridLayout()
        for row, axis in enumerate(("X", "Y")):
            speed = QDoubleSpinBox()
            speed.setDecimals(5)
            speed.setRange(0.00001, 100.0)
            speed.setSingleStep(0.1)
            speed.setToolTip("连接后读取当前速度；点击 Set 才应用，以控制器回读值为准")
            self.speed_inputs[axis] = speed
            set_button = QPushButton("Set")
            set_button.setAccessibleName(f"设置 {axis} 速度")
            set_button.clicked.connect(
                lambda checked=False, a=axis:
                self.control_thread.send_command("speed", a, self.speed_inputs[a].value())
            )
            speed_grid.addWidget(QLabel(axis), row, 0)
            speed_grid.addWidget(speed, row, 1)
            speed_grid.addWidget(set_button, row, 2)
            self.motion_buttons.append((set_button, axis))
        xy_layout.addLayout(speed_grid)
        xy_layout.addWidget(QLabel("单次距离"))
        distance_grid = QGridLayout()
        for row, axis in enumerate(("X", "Y")):
            amount = QDoubleSpinBox()
            amount.setDecimals(4)
            amount.setRange(0.0001, MAX_XY_MOVE_MM)
            amount.setSingleStep(0.1)
            amount.setValue(0.1)
            amount.setPrefix(axis + "  ")
            amount.setSuffix(" mm")
            self.axis_inputs[axis] = amount
            distance_grid.addWidget(amount, row, 0)
            for column, direction in enumerate((-1, 1), start=1):
                sign = "−" if direction < 0 else "+"
                button = QPushButton(f"{axis}{sign} 单次")
                button.clicked.connect(
                    lambda checked=False, a=axis, d=direction:
                    self.control_thread.send_command("move", a, d * self.axis_inputs[a].value())
                )
                distance_grid.addWidget(button, row, column)
                self.motion_buttons.append((button, axis))
        xy_layout.addLayout(distance_grid)
        controls.addWidget(xy_group)

        # Z 保留原来的步数和短段长按，不换算成毫米。
        z_group = QGroupBox("Z 轴  /  Arduino")
        z_grid = QGridLayout(z_group)
        amount = QSpinBox()
        amount.setRange(1, MAX_Z_MOVE_STEPS)
        amount.setValue(200)
        amount.setSuffix(" 步")
        self.axis_inputs["Z"] = amount
        z_grid.addWidget(amount, 0, 0, 1, 2)
        for column, direction in enumerate((-1, 1)):
            sign = "−" if direction < 0 else "+"
            step_button = QPushButton(f"Z{sign} 单次")
            step_button.clicked.connect(
                lambda checked=False, d=direction:
                self.control_thread.send_command("move", "Z", d * self.axis_inputs["Z"].value())
            )
            hold_button = QPushButton(f"长按 Z{sign}")
            hold_button.setAutoRepeat(False)
            hold_button.setToolTip("松开或移出按钮停止；Z 需走完当前最多 50 步")
            hold_button.pressed.connect(lambda d=direction, b=hold_button: self.start_jog("Z", d, b))
            hold_button.released.connect(self.end_jog)
            z_grid.addWidget(step_button, 1, column)
            z_grid.addWidget(hold_button, 2, column)
            self.motion_buttons.extend(((step_button, "Z"), (hold_button, "Z")))

        self.z_reference_buttons = []
        for column, (text, reset) in enumerate((("确认软件参考", False), ("当前位置设零", True))):
            button = QPushButton(text)
            button.setToolTip("仅建立新软件参考并清除 A/B，不发送运动，也不是机械回零")
            button.clicked.connect(lambda checked=False, zero=reset:
                                   self.control_thread.send_command("reference", "Z", zero))
            z_grid.addWidget(button, 3, column)
            self.z_reference_buttons.append(button)
        self.z_reference_label = QLabel("指令累计：等待连接\n软件参考：未确认\nA：未记录\nB：未记录")
        self.z_reference_label.setWordWrap(False)
        self.z_reference_label.setFixedHeight(self.z_reference_label.fontMetrics().lineSpacing() * 4 + 6)
        self.z_reference_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        z_grid.addWidget(self.z_reference_label, 4, 0, 1, 2)
        self.z_mark_buttons, self.z_return_buttons = {}, {}
        for column, mark in enumerate(("A", "B")):
            button = QPushButton("记录 " + mark)
            button.clicked.connect(lambda checked=False, m=mark:
                                   self.control_thread.send_command("mark", "Z", m))
            z_grid.addWidget(button, 5, column)
            self.z_mark_buttons[mark] = button
            button = QPushButton("返回 " + mark)
            button.setToolTip("明确点击后，移动到当前软件参考的指令位置；不补偿反向间隙")
            button.clicked.connect(lambda checked=False, m=mark: self.return_z_mark(m))
            z_grid.addWidget(button, 6, column)
            self.z_return_buttons[mark] = button
        self.btn_z_invalidate = QPushButton("手动移轴 / 失步后使参考失效")
        self.btn_z_invalidate.clicked.connect(lambda:
            self.control_thread.send_command("invalidate", "Z", "操作者报告位置或脉冲配置改变"))
        z_grid.addWidget(self.btn_z_invalidate, 7, 0, 1, 2)
        controls.addWidget(z_group)

        scan_group = QGroupBox("Z 定点扫描")
        scan_grid = QGridLayout(scan_group)
        self.scan_a, self.scan_b = QSpinBox(), QSpinBox()
        for box, prefix in ((self.scan_a, "A  "), (self.scan_b, "B  ")):
            box.setRange(-2147483647, 2147483647)
            box.setPrefix(prefix)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self.update_scan_plan)
        scan_grid.addWidget(self.scan_a, 0, 0)
        scan_grid.addWidget(self.scan_b, 0, 1)
        self.scan_interval = QSpinBox()
        self.scan_interval.setRange(1, MAX_Z_MOVE_STEPS)
        self.scan_interval.setValue(8)
        self.scan_interval.setSuffix(" 脉冲")
        self.scan_interval.valueChanged.connect(self.update_scan_plan)
        self.scan_settle = QSpinBox()
        self.scan_settle.setRange(0, 60000)
        self.scan_settle.setValue(200)
        self.scan_settle.setSuffix(" ms")
        self.scan_settle.setToolTip("初始试验等待值，需要实机比较；不代表已测得停稳")
        scan_grid.addWidget(QLabel("采样间隔"), 1, 0)
        scan_grid.addWidget(self.scan_interval, 1, 1)
        scan_grid.addWidget(QLabel("停稳等待"), 2, 0)
        scan_grid.addWidget(self.scan_settle, 2, 1)
        self.sample_name = QLineEdit()
        self.sample_name.setPlaceholderText("样品 / 视野名称")
        scan_grid.addWidget(self.sample_name, 3, 0, 1, 2)
        self.scan_output = QLineEdit(str(SAVE_DIR))
        self.scan_output.setToolTip("扫描数据目录；每条曲线使用唯一 Run，不覆盖旧数据")
        self.btn_scan_output = QPushButton("目录…")
        self.btn_scan_output.clicked.connect(self.choose_scan_output)
        scan_grid.addWidget(self.scan_output, 4, 0)
        scan_grid.addWidget(self.btn_scan_output, 4, 1)
        self.btn_roi = QPushButton("框选 ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self.on_roi_mode)
        self.btn_full_roi = QPushButton("使用全图")
        self.btn_full_roi.clicked.connect(self.use_full_roi)
        scan_grid.addWidget(self.btn_roi, 5, 0)
        scan_grid.addWidget(self.btn_full_roi, 5, 1)
        self.roi_label = QLabel("ROI：全图")
        self.roi_label.setWordWrap(True)
        scan_grid.addWidget(self.roi_label, 6, 0, 1, 2)
        self.scan_plan_label = QLabel("等待相机和 A/B")
        self.scan_plan_label.setWordWrap(True)
        self.scan_plan_label.setTextFormat(Qt.PlainText)
        self.scan_plan_label.setFixedHeight(self.scan_plan_label.fontMetrics().lineSpacing() * 4 + 6)
        scan_grid.addWidget(self.scan_plan_label, 7, 0, 1, 2)
        self.btn_scan = QPushButton("开始 Z 扫描")
        self.btn_scan.clicked.connect(self.start_scan)
        scan_grid.addWidget(self.btn_scan, 8, 0, 1, 2)
        self.scan_status = QLabel("扫描：空闲")
        self.scan_status.setWordWrap(True)
        self.scan_status.setTextFormat(Qt.PlainText)
        scan_grid.addWidget(self.scan_status, 9, 0, 1, 2)
        self.scan_inputs = [self.scan_a, self.scan_b, self.scan_interval, self.scan_settle,
            self.sample_name, self.scan_output, self.btn_scan_output, self.btn_roi,
            self.btn_full_roi]
        controls.addWidget(scan_group)

        self.btn_position = QPushButton("刷新 XY 位置")
        self.btn_position.clicked.connect(lambda: self.control_thread.send_command("position"))
        self.motion_buttons.append((self.btn_position, "X"))
        controls.addWidget(self.btn_position)

        focus_group = QGroupBox("实时清晰度 · Laplacian")
        focus_layout = QVBoxLayout(focus_group)
        focus_layout.setContentsMargins(10, 14, 10, 8)
        focus_layout.addWidget(self.focus_label)
        controls.addWidget(focus_group)
        self.btn_stop = QPushButton("停止当前运动  /  Esc")
        self.btn_stop.setStyleSheet("background: #fbe5e5; color: #9d2525; font-weight: bold;")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.clicked.connect(self.stop_movement)


        capture_group = QGroupBox("图像采集")
        capture_grid = QGridLayout(capture_group)
        self.btn_save = QPushButton("保存一张")
        self.btn_record_start = QPushButton("开始录像")
        self.btn_record_stop = QPushButton("停止录像")
        self.btn_auto = QPushButton("自动截图")
        for index, (button, command) in enumerate((
            (self.btn_save, "save"), (self.btn_record_start, "record_start"),
            (self.btn_auto, "auto"), (self.btn_record_stop, "record_stop"),
        )):
            button.clicked.connect(lambda checked=False, c=command: self.send_text(c))
            capture_grid.addWidget(button, index // 2, index % 2)
        controls.addWidget(capture_group)
        self.btn_quit = QPushButton("退出")
        self.btn_quit.clicked.connect(self.close)
        controls.addWidget(self.btn_quit)
        controls.addStretch()

        # 左侧允许滚动，较小屏幕仍能访问采集和退出按钮。
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setFixedWidth(304)
        control_scroll.setWidget(control_panel)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        left_layout.addWidget(control_scroll, stretch=1)
        left_layout.addWidget(self.btn_stop)  # 固定在滚动区外，任何位置都能停止

        right_layout = QVBoxLayout()
        right_layout.addWidget(status_group)
        self.scan_panel = ScanPanel()
        self.scan_panel.hide()
        self.image_splitter = QSplitter(Qt.Vertical)
        self.image_splitter.addWidget(self.video_label)
        self.image_splitter.addWidget(self.scan_panel)
        self.image_splitter.setStretchFactor(0, 2)
        self.image_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(self.image_splitter, stretch=1)
        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout, stretch=1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # GUI 定时取最新图，不让全分辨率图像在 Qt 事件队列中排成长队。
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_timer.start()
        self.camera_thread.text_ready.connect(self.camera_status_label.setText)
        self.camera_thread.info_ready.connect(self.on_info)
        self.camera_thread.exposure_ready.connect(self.on_exposure)
        self.camera_thread.gain_ready.connect(self.on_gain)
        self.camera_thread.diagnostic_ready.connect(self.on_diagnostic)
        self.camera_thread.capture_lock_changed.connect(self.on_capture_lock)
        self.camera_thread.finished.connect(self.on_camera_stopped)
        self.laplacian_thread.focus_score_ready.connect(self.on_focus_score)
        self.laplacian_thread.text_ready.connect(self.camera_status_label.setText)
        self.control_thread.text_ready.connect(self.on_text)
        self.control_thread.info_ready.connect(self.control_label.setText)
        self.control_thread.busy_changed.connect(self.update_controls)
        self.control_thread.connection_ready.connect(self.on_connections)
        self.control_thread.speed_ready.connect(self.on_speed)
        self.control_thread.position_ready.connect(self.on_z_position)
        self.control_thread.result_ready.connect(self.on_motion_result)
        self.scan_thread.state_changed.connect(self.on_scan_state)
        self.scan_thread.prepared.connect(self.on_scan_prepared)
        self.scan_thread.sample_ready.connect(self.scan_panel.add_sample)
        self.scan_thread.task_result.connect(self.on_scan_result)
        self.scan_thread.finished.connect(self.on_scan_stopped)

        QApplication.instance().installEventFilter(self)
        self.update_controls(True)
        self.camera_thread.start()
        self.laplacian_thread.start()
        self.control_thread.start()

    def send_text(self, text):
        self.camera_thread.send_text(text)

    def on_text(self, text):
        self.status_label.setText("控制：" + text)

    def on_info(self, text):
        self.info_label.setText(text)

    def on_exposure(self, request_id, auto, milliseconds, minimum, maximum):
        # 丢弃较早请求的回读，避免快速切换时旧状态把新选择覆盖。
        if self.closing or request_id < self.exposure_request_id:
            return
        first_read = not self.exposure_connected
        self.exposure_connected = True
        with QSignalBlocker(self.auto_exposure), QSignalBlocker(self.exposure_box):
            self.auto_exposure.setChecked(auto)
            if first_read:
                self.exposure_box.setSpecialValueText("")
                self.exposure_box.setRange(minimum, maximum)
            # 手动编辑期间保留文字、选区和光标；回车/失焦时才提交。
            if auto or first_read or not self.exposure_box.hasFocus():
                self.exposure_box.setValue(milliseconds)
        unlocked = not self.camera_locked and not self.scan_active
        self.auto_exposure.setEnabled(unlocked)
        self.exposure_box.setEnabled(not auto and unlocked)
        self.gain_box.setEnabled(self.gain_connected and not auto and unlocked)

    def change_exposure(self):
        if not self.exposure_connected or self.closing or self.camera_locked or self.scan_active:
            return
        auto = self.auto_exposure.isChecked()
        self.exposure_box.setEnabled(not auto)
        self.gain_box.setEnabled(self.gain_connected and not auto and not self.camera_locked)
        self.exposure_request_id = self.camera_thread.set_exposure(auto, self.exposure_box.value())

    def on_gain(self, request_id, percent, minimum, maximum):
        if self.closing or request_id < self.gain_request_id:
            return
        self.gain_connected = True
        with QSignalBlocker(self.gain_box):
            self.gain_box.setSpecialValueText("")
            self.gain_box.setRange(minimum, maximum)
            if not self.gain_box.hasFocus():
                self.gain_box.setValue(percent)
        self.gain_box.setEnabled(not self.auto_exposure.isChecked() and not self.camera_locked and not self.scan_active)

    def change_gain(self):
        if self.gain_connected and not self.closing and not self.camera_locked and not self.scan_active:
            self.gain_request_id = self.camera_thread.set_gain(self.gain_box.value())

    def on_diagnostic(self, diagnostic):
        self.camera_diagnostic_data = copy.deepcopy(diagnostic)
        trigger = "支持软件触发能力位" if diagnostic.get("software_trigger_supported") else "未报告软件触发能力"
        api = diagnostic.get("frame_info_api") or "等待帧信息"
        freshness = "自动连续采集 · 时序未实测" if diagnostic.get("freshness_method") == "continuous_unverified" else "连续预览 · 时序未验证"
        self.camera_diagnostic.setText(f"{diagnostic.get('camera_model', '未知型号')} · {api} · {freshness}")
        self.camera_diagnostic.setToolTip(
            f"设备 ID：{diagnostic.get('camera_id')}\nSDK：{diagnostic.get('sdk_version')}\n{trigger}\n"
            f"帧序号有效：{diagnostic.get('seq_valid')}；设备时间戳有效：{diagnostic.get('timestamp_valid')}\n"
            f"本次连接环境：{diagnostic.get('camera_environment_id')}\n"
            "扫描自动清缓存并按曝光估算等待；估算等待不是实测的新帧时序保证。"
        )
        self.update_scan_plan()

    def on_capture_lock(self, locked):
        self.camera_locked = locked
        locked = locked or self.scan_active
        self.auto_exposure.setEnabled(self.exposure_connected and not locked and not self.closing)
        self.exposure_box.setEnabled(self.exposure_connected and not self.auto_exposure.isChecked()
                                     and not locked and not self.closing)
        self.gain_box.setEnabled(self.gain_connected and not self.auto_exposure.isChecked()
                                 and not locked and not self.closing)
        for button in (self.btn_save, self.btn_record_start, self.btn_record_stop, self.btn_auto):
            button.setEnabled(not locked and not self.closing)

    def on_camera_stopped(self):
        self.exposure_connected = False
        self.auto_exposure.setEnabled(False)
        self.exposure_box.setEnabled(False)
        self.gain_connected = False
        self.gain_box.setEnabled(False)
        if self.scan_active:
            self.scan_thread.cancel_task()
        self.update_scan_plan()

    def on_z_position(self, position):
        if position.get("state_monotonic_ns", 0) < self.z_snapshot.get("state_monotonic_ns", 0):
            return
        self.z_snapshot = position
        known = position["z_position_valid"]
        reference = "已确认" if position["z_reference_confirmed"] else "未确认"
        steps = f"{position['z_command_steps']:+d} 步" if known else "未知"
        marks = position["marks"]
        if marks != self.last_marks:
            self.last_marks = dict(marks)
            self.scan_a.setValue(marks.get("A", 0))
            self.scan_b.setValue(marks.get("B", 0))
        self.z_reference_label.setText(f"指令累计：{steps}\n软件参考：{reference}\n"
                                      f"A：{marks.get('A', '未记录')}\nB：{marks.get('B', '未记录')}")
        self.z_reference_label.setToolTip(self.z_reference_label.text())
        self.update_controls(position["motion_busy"])

    def on_motion_result(self, result):
        self.on_z_position(result["position"])

    def return_z_mark(self, mark):
        if mark in self.z_snapshot.get("marks", {}):
            self.control_thread.request_z_move(self.z_snapshot["marks"][mark],
                                               request_id=None, owner="manual")

    def on_focus_score(self, score):
        self.focus_label.setText(f"{score:.2f}")

    def on_speed(self, axis, speed):
        self.speed_inputs[axis].setValue(speed)

    def on_connections(self, xy_ready, z_ready):
        self.xy_ready = xy_ready
        self.z_ready = z_ready
        self.update_controls(self.control_busy)

    def update_controls(self, busy):
        self.control_busy = busy
        manual = self.z_snapshot.get("control_owner", "manual") == "manual" and not self.scan_active
        if not busy:
            self.end_jog()
        for button, axis in self.motion_buttons:
            ready = self.z_ready if axis == "Z" else self.xy_ready
            # 正在按住的按钮必须保留，以便接收 released。
            button.setEnabled(ready and manual and (not busy or button is self.jog_button))
        for amount in self.axis_inputs.values():
            amount.setEnabled(not busy and manual)
        for speed in self.speed_inputs.values():
            speed.setEnabled(self.xy_ready and not busy and manual)
        z_idle = self.z_ready and not busy and manual
        valid = self.z_snapshot.get("z_position_valid", False)
        for button in self.z_reference_buttons:
            button.setEnabled(z_idle and valid)
        for button in self.z_mark_buttons.values():
            button.setEnabled(z_idle and valid and self.z_snapshot.get("z_reference_confirmed", False))
        for button in self.z_return_buttons.values():
            button.setEnabled(z_idle and valid and self.z_snapshot.get("allowed_z_range") is not None)
        self.btn_z_invalidate.setEnabled(z_idle and valid)
        self.update_scan_plan()

    def update_scan_plan(self):
        width = self.camera_diagnostic_data.get("width", 0)
        height = self.camera_diagnostic_data.get("height", 0)
        try:
            plan = make_scan_plan(self.scan_a.value(), self.scan_b.value(), self.scan_interval.value(), width, height, self.roi)
            self.scan_plan_label.setText(f"{plan['target_count']} 点 · {'A→B 正向' if plan['scan_direction'] > 0 else 'A→B 反向'}\n"
                f"原图未压缩约 {plan['raw_image_bytes'] / 1048576:.1f} MiB\n"
                f"磁盘保守预算 {plan['disk_estimate_bytes'] / 1048576:.1f} MiB\n"
                f"处理内存预算约 {plan['memory_estimate_bytes'] / 1048576:.1f} MiB")
            plan_ok = True
        except ValueError as error:
            self.scan_plan_label.setText(str(error) if width and height else "等待相机尺寸及有效的 A/B")
            plan_ok = False
        idle = not self.scan_active and not self.closing
        for widget in self.scan_inputs:
            widget.setEnabled(idle)
        self.btn_roi.setEnabled(idle and self.current_frame is not None)
        ready = (self.z_ready and self.exposure_connected and self.gain_connected
                 and self.z_snapshot.get("z_reference_confirmed", False)
                 and self.z_snapshot.get("z_position_valid", False)
                 and self.z_snapshot.get("allowed_z_range") is not None
                 and self.z_snapshot.get("control_owner", "manual") == "manual"
                 and not self.control_busy and not self.camera_locked)
        self.btn_scan.setEnabled(idle and ready and plan_ok)

    def choose_scan_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择扫描输出目录", self.scan_output.text())
        if path:
            self.scan_output.setText(path)

    def on_roi_mode(self, checked):
        if not checked:
            self.roi_start = None
            self.roi_band.hide()

    def use_full_roi(self):
        if self.scan_active:
            return
        self.roi = None
        self.btn_roi.setChecked(False)
        self.roi_label.setText("ROI：全图")
        if self.current_frame is not None:
            self.on_frame(self.current_frame)
        self.update_scan_plan()

    def start_scan(self):
        if self.scan_active or self.closing:
            return
        if not self.scan_output.text().strip():
            self.scan_status.setText("请先选择输出目录")
            return
        exposure_us = round(self.exposure_box.value() * 1000)
        config = {"a_steps": self.scan_a.value(), "b_steps": self.scan_b.value(),
            "interval_steps": self.scan_interval.value(), "settle_ms": self.scan_settle.value(),
            "sample_name": self.sample_name.text().strip(), "output_dir": self.scan_output.text().strip(),
            "position": copy.deepcopy(self.z_snapshot), "roi": copy.deepcopy(self.roi),
            "width": self.camera_diagnostic_data.get("width", 0), "height": self.camera_diagnostic_data.get("height", 0),
            "mode": self.camera_thread.mode, "exposure_us": exposure_us,
            "gain_percent": self.gain_box.value()}
        try:
            make_scan_plan(config["a_steps"], config["b_steps"], config["interval_steps"], config["width"], config["height"], config["roi"])
        except ValueError as error:
            self.scan_status.setText(f"扫描计划无效：{error}")
            return
        if not self.scan_thread.configure(config):
            self.scan_status.setText("上一次扫描尚未收尾")
            return
        self.btn_roi.setChecked(False)
        self.scan_active = True
        self.on_capture_lock(self.camera_locked)
        self.update_controls(self.control_busy)
        self.scan_status.setText("扫描：正在预检，尚未移动")
        self.scan_thread.start()

    def on_scan_state(self, state):
        names = {"PREPARING": "预检", "MOVING": "移动", "SETTLING": "停稳等待",
                 "CAPTURING": "采集三帧", "PROCESSING": "评分选图", "SAVING": "保存本点",
                 "COMPLETED": "完成", "CANCELLED": "已取消", "FAILED": "失败"}
        self.scan_status.setText("扫描：" + names.get(state, state))

    def on_scan_prepared(self, payload):
        self.scan_panel.begin_run(payload)
        self.scan_panel.show()
        self.image_splitter.setSizes([420, 270])

    def on_scan_result(self, result):
        name = {"completed": "完成", "cancelled": "已取消", "failed": "失败"}[result["status"]]
        self.scan_status.setText(f"扫描{name}，已保存 {result['committed_count']} 点\n"
            + (result.get("cleanup_error") or result.get("error") or "结束后停留在终点，未自动回位"))
        self.scan_status.setToolTip(result.get("run_dir") or "未创建 Run")

    def on_scan_stopped(self):
        self.scan_active = False
        self.on_z_position(self.control_thread.position_snapshot())
        self.on_capture_lock(self.camera_locked)

    def start_jog(self, axis, direction, button):
        # 鼠标移出再移入也可能触发 pressed；必须真正重新按下才能再次运动。
        if self.jog_needs_press:
            button.setDown(False)
            return
        self.jog_needs_press = True
        self.jog_button = button
        if self.control_thread.send_command("jog", axis, direction):
            self.jog_timer.start()
        else:
            self.jog_button = None

    def keep_jog(self):
        if self.jog_button is not None and self.jog_button.isDown() and self.isActiveWindow():
            self.control_thread.jog_alive = time.monotonic()
        else:
            self.end_jog()

    def end_jog(self):
        self.jog_timer.stop()
        if self.jog_button is not None:
            button = self.jog_button
            self.jog_button = None
            self.control_thread.stop_motion()
            button.setDown(False)

    def stop_movement(self):
        if self.scan_active:
            self.scan_thread.cancel_task()
            self.scan_status.setText("扫描：正在停止，等待当前动作或写盘收尾")
        self.end_jog()
        self.control_thread.stop_motion()
        self.camera_thread.cancel_capture()
        self.on_text("已请求停止；Z 将走完当前短段后停止")

    def eventFilter(self, watched, event):
        if watched is self.video_label and event.type() == QEvent.Resize and self.current_frame is not None:
            self.on_frame(self.current_frame)
        if (watched is self.video_label and self.btn_roi.isChecked()
                and not self.scan_active and self.current_frame is not None):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.roi_start = event.pos()
                self.roi_band.setGeometry(QRect(self.roi_start, self.roi_start))
                self.roi_band.show()
                return True
            if event.type() == QEvent.MouseMove and self.roi_start is not None:
                self.roi_band.setGeometry(QRect(self.roi_start, event.pos()).normalized())
                return True
            if event.type() == QEvent.MouseButtonRelease and self.roi_start is not None:
                start = self.roi_start
                try:
                    self.roi = roi_from_drag((start.x(), start.y()), (event.x(), event.y()),
                        (self.video_label.width(), self.video_label.height()),
                        (self.current_frame.shape[1], self.current_frame.shape[0]),
                        (self.video_label.pixmap().width(), self.video_label.pixmap().height()))
                    self.roi_label.setText("ROI：" + ", ".join(map(str, self.roi)) + "（原图像素）")
                except ValueError as error:
                    self.roi_label.setText(str(error))
                self.btn_roi.setChecked(False)
                self.on_frame(self.current_frame)
                self.update_scan_plan()
                return True
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.jog_needs_press = False
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.jog_needs_press = False
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.stop_movement()
            return True
        if self.jog_button is not None:
            if (event.type() == QEvent.ApplicationDeactivate
                    or (watched is self and event.type() == QEvent.WindowDeactivate)
                    or (watched is self.jog_button and event.type() == QEvent.Leave)
                    or event.type() == QEvent.MouseButtonRelease):
                self.end_jog()
        return super().eventFilter(watched, event)

    def on_frame(self, frame):
        self.current_frame = frame
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        if self.roi is not None:
            x, y, width, height = self.roi
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#f1c54b"), 2))
            painter.drawRect(round(x * pixmap.width() / w), round(y * pixmap.height() / h),
                             round(width * pixmap.width() / w), round(height * pixmap.height() / h))
            painter.end()
        self.video_label.setPixmap(pixmap)
        self.btn_roi.setEnabled(not self.scan_active and not self.closing)

    def refresh_preview(self):
        frame = self.camera_thread.take_preview_frame()
        if frame is not None and not self.closing:
            self.on_frame(frame)
            self.laplacian_thread.add_frame(frame)

    def closeEvent(self, event):
        if not self.closing:
            self.closing = True
            self.preview_timer.stop()
            self.stop_movement()
            self.setEnabled(False)
        if self.scan_thread.isRunning():
            self.status_label.setText("正在等待扫描和当前写盘收尾…")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if not self.workers_stopping:
            self.workers_stopping = True
            self.send_text("stop")
            self.laplacian_thread.stop_thread()
            self.control_thread.stop_thread()
        threads = (self.camera_thread, self.laplacian_thread, self.control_thread)
        if any(thread.isRunning() for thread in threads):
            # 不销毁仍在运行的 QThread，也不在 GUI 主线程阻塞等待。
            self.status_label.setText("正在停止运动并关闭设备…")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        QApplication.instance().removeEventFilter(self)
        event.accept()


def choose_settings():
    """启动时说明单位，并分别选择相机模式、ASI 和 Arduino 串口。"""
    saved = {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            saved = data
    except (OSError, ValueError):
        pass  # 首次启动没有设置文件，或文件损坏时，仍允许手动选择。
    dialog = QDialog()
    dialog.setWindowTitle("启动设置")
    dialog.setMinimumWidth(530)
    layout = QVBoxLayout(dialog)
    note = QLabel(
        "XY 使用 ASI MS-2000，输入距离（mm）；Z 使用 Arduino，输入步数。\n"
        "两套单位暂不统一，请分别选择串口，不可选同一个。\n"
        "XY 初始沿用 ASI 当前速度，可分别设置 X/Y 速度（mm/s）。\n"
        "XY 八方向长按移动，松开停止；单次限 5 秒或每轴 5 mm。\n"
        "Z 沿用固件约 1000 步/秒，松开后走完当前最多 50 步。\n"
        "请先确认方向和可用行程。\n"
        "串口号会自动记住；可留空仅查看相机。\n"
        "连接前请关闭占用串口的其他程序。"
    )
    note.setWordWrap(True)
    layout.addWidget(note)
    form = QFormLayout()
    mode_box = QComboBox()
    mode_box.addItems(["0 - 5440 x 3648", "1 - 2736 x 1824", "2 - 1824 x 1216"])
    mode_box.setCurrentIndex(2)
    xy_box = QComboBox()
    z_box = QComboBox()
    ports = [port.device for port in list_ports.comports()]
    for box in (xy_box, z_box):
        box.setEditable(True)
        box.addItems([""] + ports)
        box.lineEdit().setPlaceholderText("留空不连接，也可输入 COM 口")
    for box, key in ((xy_box, "xy_port"), (z_box, "z_port")):
        if isinstance(saved.get(key), str):
            box.setCurrentText(saved[key])
    form.addRow("相机模式", mode_box)
    form.addRow("XY · ASI 串口", xy_box)
    form.addRow("Z · Arduino 串口", z_box)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("启动")
    buttons.button(QDialogButtonBox.Cancel).setText("取消")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    while dialog.exec_() == QDialog.Accepted:
        xy_port = xy_box.currentText().strip().upper()
        z_port = z_box.currentText().strip().upper()
        if xy_port and xy_port == z_port:
            QMessageBox.warning(dialog, "串口重复", "XY 和 Z 是不同设备，请选择不同的串口。")
            continue
        try:
            SETTINGS_PATH.write_text(
                json.dumps({"xy_port": xy_port, "z_port": z_port}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            QMessageBox.warning(dialog, "设置未保存", f"本次仍可启动，但无法记住串口号：{e}")
        return mode_box.currentIndex(), xy_port, z_port
    return None


def main():
    app = QApplication(sys.argv)
    settings = choose_settings()
    if settings is None:
        return
    window = MainWindow(*settings)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
