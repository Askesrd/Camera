# Camera.py
# 相机线程文件：保留 OpenCV 风格，只增加“接收文本命令”和“发送图像帧”两个接口

import ctypes
import queue
import time
from pathlib import Path

import cv2 as cv
import numpy as np

import toupcam
from PyQt5.QtCore import QThread, pyqtSignal


SAVE_DIR = Path.home() / "Desktop" / "111"
WAIT_FRAME_HRESULTS = {0x8001011F, 0x8000000A}
MODE_FPS = {0: 15, 1: 50, 2: 60}


class CameraThread(QThread):
    # ==============================
    # 多线程输出接口：相机线程 → GUI
    # ==============================
    frame_ready = pyqtSignal(object)  # 发送图像 frame
    text_ready = pyqtSignal(str)      # 发送状态文字
    info_ready = pyqtSignal(str)      # 发送帧率、曝光等相机信息

    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        self.fps = MODE_FPS[mode]

        # ==============================
        # 多线程输入接口：GUI → 相机线程
        # GUI 发来的文本命令会放到这个队列
        # ==============================
        self.commands = queue.Queue()

        self.running = True

        # 相机相关
        self.cam = None
        self.buf = None
        self.width = 0
        self.height = 0
        self.row_pitch = 0

        # 录像相关
        self.recording = False
        self.out = None
        self.video_count = 0

        # 自动截图相关
        self.auto_capture = False
        self.auto_start = 0
        self.auto_last = 0
        self.auto_duration = 10
        self.auto_interval = 2
        self.session_count = 0
        self.image_count = 0

        # 截图相关
        self.one_frame_ready = False
        self.one_frame = 0

        # 帧率显示相关
        self.fps_count = 0
        self.fps_last_time = time.perf_counter()
    # ==================================================
    # 多线程输入接口
    # main.py 只需要调用这个函数发送文本命令即可
    # ==================================================
    def send_text(self, text):
        """
        GUI 给相机线程发命令。
        将文本放入self.commands队列中

            send_text("save")
            send_text("record_start")
            send_text("record_stop")
            send_text("auto")
            send_text("stop")
        """
        self.commands.put(text)

    # ==================================================
    # 线程主函数
    # 调用 start() 后，Qt 会自动在后台线程里执行 run()
    # ==================================================
    def run(self):
        try:
            # 1. 打开相机
            self.cam = toupcam.Toupcam.Open(None)
            if self.cam is None:
                self.text_ready.emit("相机打开失败")
                return

            self.cam.put_eSize(self.mode)
            self.width, self.height = self.cam.get_Size()

            self.row_pitch = ((self.width * 24 + 31) // 32 * 4)
            self.buf = ctypes.create_string_buffer(self.row_pitch * self.height)

            self.cam.put_Option(toupcam.TOUPCAM_OPTION_TRIGGER, 0)
            self.cam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 0)

            try:
                self.cam.put_Option(toupcam.TOUPCAM_OPTION_BYTEORDER, 1)
            except toupcam.HRESULTException:
                pass

            self.cam.StartPullModeWithCallback(None, None)

            self.text_ready.emit(f"相机已启动")


            while self.running:
                # 处理 GUI 发来的文本命令
                self.handle_text_commands()

                # 拉取相机图像
                try:
                    self.cam.PullImageV2(self.buf, 24, None)
                except toupcam.HRESULTException as ex:
                    hr = ex.hr & 0xFFFFFFFF

                    if hr in WAIT_FRAME_HRESULTS:
                        time.sleep(0.001)
                        continue

                    self.text_ready.emit(f"拉图失败：hr=0x{hr:08X}")
                    break

                # buffer 转成 OpenCV frame
                frame = self.buffer_to_frame()

                # ==============================
                # 多线程输出接口：发送图像给 GUI
                # ==============================
                self.frame_ready.emit(frame)

                # 把当前帧交给命令处理函数，由它统一判断录像、单张保存、自动截图等任务。
                self.handle_text_commands(frame)

                # 每秒刷新一次 SDK 帧率和曝光时间，避免把状态提示文字刷掉。
                self.show_camera_info()

        except Exception as e:
            self.text_ready.emit(f"相机线程错误：{e}")

        finally:
            self.cleanup()

    def show_camera_info(self):
        """
        显示相机实时信息。

        Python 帧率：本程序成功 PullImageV2 的速度。
        SDK 帧率：相机 SDK 自己统计的真实输出速度。
        曝光时间：当前相机实际曝光时间，单位是微秒。
        """
        self.fps_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_last_time

        if elapsed < 1.0:
            return

        python_fps = self.fps_count / elapsed

        try:
            sdk_frame_count, sdk_time_ms, sdk_total_frames = self.cam.get_FrameRate()
            sdk_fps = sdk_frame_count * 1000.0 / sdk_time_ms if sdk_time_ms else 0.0
        except toupcam.HRESULTException:
            sdk_fps = 0.0
            sdk_total_frames = 0

        try:
            real_exposure_us = self.cam.get_RealExpoTime()
        except toupcam.HRESULTException:
            real_exposure_us = 0

        self.info_ready.emit(
            f"相机信息：分辨率 {self.width} x {self.height} | "
            f"计算帧率 {python_fps:.2f} fps | "
            f"SDK帧率 {sdk_fps:.2f} fps | 曝光 {real_exposure_us} us"
        )

        self.fps_count = 0
        self.fps_last_time = now

    def handle_text_commands(self, frame=None):
        """
        统一处理 GUI 发来的文本命令，以及依赖当前帧的后台任务。

        frame 为 None 时：只处理按钮命令，例如开始录像、停止录像、启动自动截图。
        frame 不为 None 时：同时处理需要当前画面的任务，例如写录像帧、保存单张、自动截图。
        """
        while True:
            try:
                text = self.commands.get_nowait()
            except queue.Empty:
                break

            if text == "stop":
                self.running = False

            elif text == "save":
                self.save_one_frame()

            elif text == "record_start":
                self.handle_record("start")

            elif text == "record_stop":
                self.handle_record("stop")

            elif text == "auto":
                self.handle_auto_capture()

        if frame is None:
            return

        # 下面这些判断原本容易散落在主循环里，现在统一收进 handle_text_commands。
        self.handle_record("write", frame)

        if self.one_frame_ready:
            self.save_one_frame(frame)

        if self.auto_capture:
            self.handle_auto_capture(frame)

    def buffer_to_frame(self):
        """
        转换图像。
        """
        raw = np.frombuffer(
            self.buf,
            dtype=np.uint8,
            count=self.row_pitch * self.height
        )

        if self.row_pitch == self.width * 3:
            return raw.reshape((self.height, self.width, 3)).copy()

        rows = raw.reshape((self.height, self.row_pitch))
        frame = rows[:, :self.width * 3].reshape((self.height, self.width, 3)).copy()

        return frame




    def handle_record(self, action, frame=None):
        """
        统一处理录像功能。

        action == "start"：开始录像。
        action == "stop" ：停止录像。
        action == "write"：如果正在录像，就把当前帧写入视频。
        """
        if action == "start":
            if self.recording:
                return

            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            self.video_count += 1
            path = SAVE_DIR / f"视频-{self.video_count}.mp4"

            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self.out = cv.VideoWriter(
                str(path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not self.out.isOpened():
                self.text_ready.emit("录像启动失败")
                self.out = None
                return

            self.recording = True
            self.text_ready.emit(f"开始录像：{path}")
            return

        if action == "stop":
            if not self.recording:
                return

            self.recording = False
            if self.out is not None:
                self.out.release()
                self.out = None
            self.text_ready.emit("录像已停止")
            return

        if action == "write":
            if self.recording and self.out is not None and frame is not None:
                self.out.write(frame)
            return

        self.text_ready.emit(f"未知录像命令：{action}")

    def save_one_frame(self, frame=None):
        """
        保存单张图片。

        如果收到 save 命令时还没有当前帧，就先设置 self.one_frame_ready。
        下一次 handle_text_commands(frame) 收到有效帧时再真正保存。
        """
        if frame is None:
            self.one_frame_ready = True
            self.text_ready.emit("收到截图命令，等待下一帧")
            return

        self.one_frame_ready = False
        self.one_frame += 1
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = SAVE_DIR / f"single-{self.one_frame}-{timestamp}.jpg"

        if cv.imwrite(str(path), frame):
            self.text_ready.emit(f"单张图片保存：{path}")
        else:
            self.text_ready.emit("单张图片保存失败")

    def handle_auto_capture(self, frame=None):
        """
        统一处理自动截图。

        frame is None：收到 GUI 的 auto 命令，启动一次自动截图会话。
        frame 不为 None：有新图像时，根据时间间隔保存图片，并判断会话是否结束。
        """
        if frame is None:
            if self.auto_capture:
                return

            self.session_count += 1
            self.image_count = 0
            self.auto_capture = True
            self.auto_start = time.time()
            self.auto_last = self.auto_start

            self.text_ready.emit("开始自动截图")
            return

        if not self.auto_capture:
            return

        now = time.time()

        if now - self.auto_last >= self.auto_interval:
            self.image_count += 1
            self.auto_last = now

            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVE_DIR / f"{self.session_count}-{self.image_count}.jpg"

            if cv.imwrite(str(path), frame):
                self.text_ready.emit(f"自动截图保存：{path}")
            else:
                self.text_ready.emit("自动截图失败")

        if now - self.auto_start >= self.auto_duration:
            self.auto_capture = False
            self.text_ready.emit("自动截图结束")

    def cleanup(self):
        """
        退出时释放相机和视频资源
        """
        self.handle_record("stop")

        if self.cam is not None:
            try:
                self.cam.Stop()
            except Exception:
                pass

            try:
                self.cam.Close()
            except Exception:
                pass

            self.cam = None

        self.text_ready.emit("相机线程已退出")
