# main.py
# GUI 文件：显示视频窗口，把按钮变成文本命令发给 Camera.py

import sys

import cv2 as cv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from Camera import CameraThread
from Laplacian import LaplacianThread
from t2 import AutofocusThread


class MainWindow(QMainWindow):
    def __init__(self, mode):
        super().__init__()

        self.camera_thread = CameraThread(mode)
        self.laplacian_thread = LaplacianThread()
        self.focus_thread = AutofocusThread()

        self.setWindowTitle("视频窗口")
        self.resize(1100, 760)

        self.video_label = QLabel("等待图像")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: black; color: white;")
        self.video_label.setMinimumSize(640, 360)

        self.status_label = QLabel("状态：启动中")
        self.info_label = QLabel("相机信息：等待数据")
        self.focus_label = QLabel("清晰度：等待数据")

        self.btn_save = QPushButton("保存一张")
        self.btn_record_start = QPushButton("开始录像")
        self.btn_record_stop = QPushButton("停止录像")
        self.btn_auto = QPushButton("自动截图")
        self.btn_autofocus = QPushButton("自动对焦")
        self.btn_quit = QPushButton("退出")


        # 按钮连接函数
        # 按钮点击后，把文本命令发给 Camera.py

        self.btn_save.clicked.connect(lambda: self.send_text("save"))
        self.btn_record_start.clicked.connect(lambda: self.send_text("record_start"))
        self.btn_record_stop.clicked.connect(lambda: self.send_text("record_stop"))
        self.btn_auto.clicked.connect(lambda: self.send_text("auto"))
        self.btn_autofocus.clicked.connect(self.focus_thread.start_autofocus)
        self.btn_quit.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_record_start)
        button_layout.addWidget(self.btn_record_stop)
        button_layout.addWidget(self.btn_auto)
        button_layout.addWidget(self.btn_autofocus)
        button_layout.addWidget(self.btn_quit)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.focus_label)
        layout.addLayout(button_layout)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # ==============================
        # 多线程通信：
        # Camera.py 发图像 → main.py on_frame
        # Camera.py 发文字 → main.py on_text
        # ==============================
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.frame_ready.connect(self.laplacian_thread.add_frame)
        self.camera_thread.text_ready.connect(self.on_text)
        self.camera_thread.info_ready.connect(self.on_info)
        self.laplacian_thread.focus_score_ready.connect(self.on_focus_score)
        self.laplacian_thread.focus_score_ready.connect(self.focus_thread.add_focus_score)
        self.laplacian_thread.text_ready.connect(self.on_text)
        self.focus_thread.text_ready.connect(self.on_text)

        # 启动后台线程：相机发图，Laplacian 算清晰度，t2 按钮按下后运动电机。
        self.camera_thread.start()
        self.laplacian_thread.start()
        self.focus_thread.start()

    def send_text(self, text):
        """
        main.py 给 Camera.py 发文本命令。
        """
        self.camera_thread.send_text(text)

    def on_text(self, text):
        """
        Camera.py 给 main.py 发状态文字，将文本传递到status_label中
        """
        self.status_label.setText(text)

    def on_info(self, text):
        """
        Camera.py 给 main.py 发相机实时信息。
        """
        self.info_label.setText(text)
        self.setWindowTitle(text)

    def on_focus_score(self, score):
        """
        Laplacian.py 给 main.py 发清晰度分数，同时这份分数也会送到 t2.py。
        """
        self.focus_label.setText(f"清晰度：{score:.2f}")

    def on_frame(self, frame):
        """
        Camera.py 给 main.py 发图像 frame。
        这里负责把 OpenCV 图像显示到 QLabel。
        """
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        image = QImage(
            rgb.data,
            w,
            h,
            w * ch,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """
        关闭窗口时，通知相机线程退出。
        """
        self.send_text("stop")
        self.laplacian_thread.stop_thread()
        self.focus_thread.stop_thread()
        self.camera_thread.wait(3000)
        self.laplacian_thread.wait(3000)
        self.focus_thread.wait(3000)
        event.accept()


def choose_mode():
    items = ["0 - 5440 x 3648", "1 - 2736 x 1824", "2 - 1824 x 1216"]

    item, ok = QInputDialog.getItem(
        None,
        "选择模式",
        "模式：",
        items,
        2,
        False
    )

    if not ok:
        return None

    return int(item.split(" ", 1)[0])


def main():
    app = QApplication(sys.argv)

    mode = choose_mode()
    if mode is None:
        return

    window = MainWindow(mode) #调用MainWindow.__init__(self, mode)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
