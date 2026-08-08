import queue
import time

from PyQt5.QtCore import QThread, pyqtSignal


SERIAL_PORT = "COM11"
BAUDRATE = 115200

MOVE_STEPS = 200       # 每次移动步数
BACK_STEPS = 120       # 最后回退步数
MAX_MOVES = 12         # 最多移动次数，防止电机一直运动
SCORE_SAMPLES = 3      # 每次读取几帧清晰度后取平均
SCORE_TIMEOUT = 3.0    # 等待清晰度数据的最长时间
IMPROVE_RATIO = 0.03   # 清晰度提升超过3%才认为真的变清晰


class AutofocusThread(QThread):
    """

    Laplacian.py 负责计算清晰度，并把分数送到 add_focus_score()。
    本线程只负责：接收清晰度、判断移动方向、发送 Arduino 串口命令。
    """

    text_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.running = True
        self.autofocusing = False

        # GUI 发来的命令，例如 "start" / "stop"。
        self.commands = queue.Queue()

        # Laplacian.py 发来的清晰度分数。
        self.scores = queue.Queue(maxsize=30)

        self.ser = None

    def start_autofocus(self):
        """
        GUI 点击“自动对焦”按钮时调用。
        """
        if self.autofocusing:
            self.text_ready.emit("自动对焦已经在运行")
            return

        self.commands.put("start")

    def stop_thread(self):
        """
        窗口关闭时调用。
        """
        self.running = False
        self.commands.put("stop")

    def add_focus_score(self, score):
        """
        接收 Laplacian.py 发来的清晰度分数。
        """
        if not self.running:
            return

        # 队列满时丢掉旧分数，避免自动对焦使用过期画面。
        if self.scores.full():
            try:
                self.scores.get_nowait()
            except queue.Empty:
                pass

        try:
            self.scores.put_nowait(float(score))
        except queue.Full:
            pass

    def run(self):
        """
        线程主循环：平时等待命令；收到 start 后执行一次自动对焦。
        """
        while self.running:
            try:
                command = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue

            if command == "stop":
                break

            if command == "start":
                self.autofocus_once()

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        self.text_ready.emit("自动对焦线程已退出")

    def autofocus_once(self):
        """
        执行一次简单自动对焦：
        向前移动，如果变模糊就换方向；再次变模糊就回退一点并停止。
        """
        self.autofocusing = True
        self.text_ready.emit("自动对焦启动")

        try:
            try:
                import serial

                if self.ser is None or not self.ser.is_open:
                    self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.2)
                    time.sleep(2)  # Arduino 打开串口后通常会重启。
                    self.ser.reset_input_buffer()
                    self.text_ready.emit(f"串口已打开：{SERIAL_PORT}")
            except Exception as e:
                self.text_ready.emit(f"串口打开失败：{e}")
                return

            last_score = self.get_score()
            if last_score is None:
                self.text_ready.emit("自动对焦失败：没有收到清晰度数据")
                return

            direction = 1
            changed_direction = False
            best_score = last_score

            self.text_ready.emit(f"初始清晰度：{last_score:.2f}")

            for move_index in range(1, MAX_MOVES + 1):
                if not self.running:
                    return

                if not self.move_motor(direction, MOVE_STEPS):
                    self.text_ready.emit("自动对焦停止：电机运动失败或 Arduino 无返回")
                    return

                # 等机械稳定后再读图像清晰度。
                time.sleep(0.2)

                score = self.get_score()
                if score is None:
                    self.text_ready.emit("自动对焦停止：运动后没有收到清晰度数据")
                    return

                self.text_ready.emit(
                    f"第 {move_index} 次移动：方向 {direction}，清晰度 {score:.2f}"
                )

                if score > last_score * (1.0 + IMPROVE_RATIO):
                    last_score = score
                    best_score = max(best_score, score)
                    continue

                if not changed_direction:
                    direction = -direction
                    changed_direction = True
                    last_score = score
                    self.text_ready.emit("清晰度下降，换方向继续寻找")
                    continue

                self.move_motor(-direction, BACK_STEPS)
                self.text_ready.emit(
                    f"再次变模糊，回退 {BACK_STEPS} 步后停止；最佳记录：{best_score:.2f}"
                )
                return

            self.text_ready.emit(f"达到最大移动次数，最佳记录：{best_score:.2f}")

        finally:
            self.autofocusing = False

    def get_score(self):
        """
        读取几帧清晰度并取平均。
        """
        while True:
            try:
                self.scores.get_nowait()
            except queue.Empty:
                break

        values = []
        start_time = time.time()

        while self.running and len(values) < SCORE_SAMPLES:
            if time.time() - start_time > SCORE_TIMEOUT:
                break

            try:
                values.append(self.scores.get(timeout=0.2))
            except queue.Empty:
                pass

        if not values:
            return None

        return sum(values) / len(values)

    def move_motor(self, direction, steps):
        """
        发送运动命令，并等待 Arduino 返回一行文字。
        命令格式：'<方向> <步数>\\n'，例如 '1 200\\n'。
        """
        if self.ser is None or not self.ser.is_open:
            return False

        command = f"{direction} {steps}\n"

        try:
            self.ser.write(command.encode())
        except Exception as e:
            self.text_ready.emit(f"串口发送失败：{e}")
            return False

        start_time = time.time()
        while self.running and time.time() - start_time < 8.0:
            try:
                reply = self.ser.readline().decode(errors="ignore").strip()
            except Exception as e:
                self.text_ready.emit(f"串口读取失败：{e}")
                return False

            if reply:
                self.text_ready.emit(f"已发送 {command.strip()}，Arduino 返回：{reply}")
                return True

        return False
