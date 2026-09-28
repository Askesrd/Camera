import queue

import cv2 as cv
from PyQt5.QtCore import QThread, pyqtSignal


def calculate_focus_score(frame, roi=None):
    """
    计算一帧图像的清晰度。

    frame 来自 Camera.py 里 buffer_to_frame() 得到的实时相机图像。
    当前 demo 使用 Laplacian 方差：数值越大，通常表示边缘越多、图像越清晰。
    """
    if frame is None:
        return 0.0

    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]

    if frame.size == 0:
        return 0.0

    if len(frame.shape) == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray = frame

    lap = cv.Laplacian(gray, cv.CV_64F)
    return float(lap.var())


# 保留旧函数名，后面如果还有地方调用 laplacian(img)，也不会立刻失效。
def laplacian(img):
    return calculate_focus_score(img)


class LaplacianThread(QThread):
    """
    清晰度计算线程。

    Camera.py 把实时图像 frame 发到 add_frame()。
    本线程计算 Laplacian 清晰度后，通过 focus_score_ready 发给 GUI 显示。
    """

    focus_score_ready = pyqtSignal(float)
    text_ready = pyqtSignal(str)

    def __init__(self, roi=None):
        super().__init__()
        self.roi = roi
        self.running = True
        self.frames = queue.Queue(maxsize=2)

    def add_frame(self, frame):
        """
        接收 Camera.py 发来的图像。

        如果清晰度计算来不及处理所有帧，就丢掉旧帧，只保留较新的图像，
        避免界面显示已经过时的清晰度。
        """
        if not self.running:
            return

        if self.frames.full():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass

        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            pass

    def stop_thread(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue

            score = calculate_focus_score(frame, self.roi)
            self.focus_score_ready.emit(score)

        self.text_ready.emit("清晰度计算线程已退出")
