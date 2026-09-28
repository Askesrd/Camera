# main.py
# GUI：左侧操作，右上状态，右下图像。阻塞的硬件操作仍放在工作线程里。

import json
import sys
import time
from pathlib import Path

import cv2 as cv
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
    QWidget,
)
from serial.tools import list_ports

from Camera import CameraThread
from Laplacian import LaplacianThread
from Control import (
    ControlThread, MAX_XY_MOVE_MM, MAX_Z_MOVE_STEPS,
    JOG_MAX_SECONDS, JOG_MAX_XY_MM, JOG_MAX_Z_STEPS,
)


SETTINGS_PATH = Path(__file__).with_name("settings.json")


class MainWindow(QMainWindow):
    def __init__(self, mode, xy_port="", z_port=""):
        super().__init__()
        self.camera_thread = CameraThread(mode)
        self.laplacian_thread = LaplacianThread()
        self.control_thread = ControlThread(xy_port, z_port)
        self.xy_ready = False
        self.z_ready = False
        self.control_busy = True
        self.closing = False
        self.jog_button = None
        self.jog_needs_press = False
        self.motion_buttons = []
        self.axis_inputs = {}
        self.speed_inputs = {}
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
        self.video_label.setMinimumSize(400, 280)
        # 忽略上一张图片的尺寸建议，缩放窗口时图像区域才不会越撑越大。
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

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
        controls.addWidget(z_group)

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
        self.btn_stop.clicked.connect(self.stop_movement)
        controls.addWidget(self.btn_stop)

        hint = QLabel(
            f"长按上限：{JOG_MAX_SECONDS:g} 秒；XY 每轴 {JOG_MAX_XY_MM:g} mm / "
            f"Z {JOG_MAX_Z_STEPS} 步，先到即停。\n"
            "Z 松开后走完当前短段，约 50 ms 加通信延迟；不是硬件急停。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #60666e; font-size: 11px;")
        controls.addWidget(hint)

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

        right_layout = QVBoxLayout()
        right_layout.addWidget(status_group)
        right_layout.addWidget(self.video_label, stretch=1)
        layout = QHBoxLayout()
        layout.addWidget(control_scroll)
        layout.addLayout(right_layout, stretch=1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.frame_ready.connect(self.laplacian_thread.add_frame)
        self.camera_thread.text_ready.connect(self.camera_status_label.setText)
        self.camera_thread.info_ready.connect(self.on_info)
        self.laplacian_thread.focus_score_ready.connect(self.on_focus_score)
        self.laplacian_thread.text_ready.connect(self.camera_status_label.setText)
        self.control_thread.text_ready.connect(self.on_text)
        self.control_thread.info_ready.connect(self.control_label.setText)
        self.control_thread.busy_changed.connect(self.update_controls)
        self.control_thread.connection_ready.connect(self.on_connections)
        self.control_thread.speed_ready.connect(self.on_speed)

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
        if not busy:
            self.end_jog()
        for button, axis in self.motion_buttons:
            ready = self.z_ready if axis == "Z" else self.xy_ready
            # 正在按住的按钮必须保留，以便接收 released。
            button.setEnabled(ready and (not busy or button is self.jog_button))
        for amount in self.axis_inputs.values():
            amount.setEnabled(not busy)
        for speed in self.speed_inputs.values():
            speed.setEnabled(self.xy_ready and not busy)

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
        self.end_jog()
        self.control_thread.stop_motion()
        self.on_text("已请求停止；Z 将走完当前短段后停止")

    def eventFilter(self, watched, event):
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
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        if not self.closing:
            self.closing = True
            self.stop_movement()
            self.send_text("stop")
            self.laplacian_thread.stop_thread()
            self.control_thread.stop_thread()
            self.setEnabled(False)
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
