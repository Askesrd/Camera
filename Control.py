# Control.py
# 统一管理 ASI 的 X/Y 和 Arduino 的 Z，负责手动移动与长按移动。
# XY 仍然使用 mm，Z 仍然使用步数；两套串口协议分别处理

import math
import queue
import re
import threading
import time

import serial
from PyQt5.QtCore import QThread, pyqtSignal


BAUDRATE = 115200
XY_UNITS_PER_MM = 10000
Z_CHUNK_STEPS = 50      # 固件不能中途停止，因此每次只发送 50 步
MAX_XY_MOVE_MM = 10.0
MAX_Z_MOVE_STEPS = 10000
XY_MOVE_TIMEOUT = 10.0
JOG_MAX_SECONDS = 5.0   # 一次长按的上限；到达后需松开重新按
JOG_MAX_XY_MM = 5.0     # XY 长按一次只发一条 R，各参与轴最多走 5 mm
JOG_MAX_Z_STEPS = 2000
JOG_HEARTBEAT_TIMEOUT = 0.4  # GUI 卡住时，XY 发 HALT，Z 不再发送下一段

class ControlThread(QThread):
    text_ready = pyqtSignal(str)
    info_ready = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    connection_ready = pyqtSignal(bool, bool)
    speed_ready = pyqtSignal(str, float)

    def __init__(self, xy_port="", z_port=""):
        super().__init__()
        self.xy_port = xy_port.strip().upper()
        self.z_port = z_port.strip().upper()
        self.xy_ser = None
        self.z_ser = None
        self.xy_speed = {}
        self.xy_position = "未读取"
        self.z_steps = 0    # 本次连接后已收到 Finish 的指令累计，不是编码器位置
        self.z_position_known = True

        self.running = True
        self.busy = True    # 串口初始化完成前不接收运动
        self.commands = queue.Queue(maxsize=1)
        self.command_lock = threading.Lock()
        self.cancel_motion = threading.Event()
        self.jogging = False
        self.jog_alive = 0.0
        self.jog_started = 0.0

    def send_command(self, command, axis="", value=0):
        """GUI 只放入一条任务；运动期间不积压点击，也不在这里读写串口。"""
        with self.command_lock:
            if not self.running or self.busy:
                self.text_ready.emit("控制器忙，请等待完成或先停止当前运动")
                return False
            if command == "jog" and axis == "XY":
                if (not isinstance(value, tuple) or len(value) != 2
                        or any(v not in (-1, 0, 1) for v in value) or value == (0, 0)):
                    return False
            elif command == "speed":
                if axis not in ("X", "Y") or not math.isfinite(value) or not 0 < value <= 100:
                    return False
            elif command in ("move", "jog"):
                if axis not in ("X", "Y", "Z") or not math.isfinite(value):
                    return False
                limit = MAX_Z_MOVE_STEPS if axis == "Z" else MAX_XY_MOVE_MM
                if command == "move" and (value == 0 or abs(value) > limit):
                    return False
                if axis == "Z" and value != int(value):
                    return False
                if command == "jog" and value not in (-1, 1):
                    return False
            elif command != "position":
                return False

            port = self.z_ser if axis == "Z" else self.xy_ser
            if command != "position" and (port is None or not port.is_open):
                self.text_ready.emit("对应设备未连接，请在启动时选择正确串口")
                return False
            self.cancel_motion.clear()
            self.jog_alive = time.monotonic()
            self.busy = True
            self.commands.put_nowait((command, axis, value))
        self.busy_changed.emit(True)
        return True

    def stop_motion(self):
        # 不排在普通任务后面，运动等待循环也能立即看到这个标志。
        self.cancel_motion.set()

    def stop_thread(self):
        self.running = False
        self.stop_motion()

    def should_stop(self):
        if not self.running or self.cancel_motion.is_set():
            return True
        if self.jogging:
            now = time.monotonic()
            if now - self.jog_alive > JOG_HEARTBEAT_TIMEOUT:
                self.cancel_motion.set()
                self.text_ready.emit("长按已停止：按钮松开或界面未继续响应")
                return True
            if now - self.jog_started >= JOG_MAX_SECONDS:
                self.cancel_motion.set()
                self.text_ready.emit("达到单次长按时间上限，请松开后重新按")
                return True
        return False

    def read_reply(self, port, timeout=1.0):
        """等待完整的一行；兼容 CR/LF，串口短超时不会把半行当成回复。"""
        data = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            char = port.read(1)
            if char in (b"\r", b"\n"):
                if data:
                    return data.decode("ascii", errors="replace").strip()
            elif char:
                data.extend(char)
                if len(data) > 256:
                    raise RuntimeError("串口回复过长，协议可能不匹配")
        raise TimeoutError("等待设备完整回复超时")

    def send_xy(self, command):
        """只用于 ASI：命令以回车结尾；不要用它给 Arduino 发步数。"""
        self.xy_ser.write((command + "\r").encode("ascii"))
        return self.read_reply(self.xy_ser)

    def open_serials(self):
        if self.xy_port and self.xy_port == self.z_port:
            self.text_ready.emit("XY 和 Z 不能选择同一串口，本次不连接运动设备")
            return
        if self.xy_port and self.running:
            try:
                self.xy_ser = serial.Serial(self.xy_port, BAUDRATE, timeout=0.05, write_timeout=0.5)
                self.cancel_motion.wait(0.3)
                self.xy_ser.reset_input_buffer()
                reply = self.send_xy("N")
                if "MS2000" not in reply.upper().replace("-", ""):
                    raise RuntimeError(f"未识别为 ASI MS-2000：{reply}")
                for axis in ("X", "Y"):
                    # 查询而不修改单位和速度。旧固件 UM 回复可能没有 :A 前缀。
                    reply = self.send_xy(f"UM {axis}?")
                    match = re.search(rf"\b{axis}\s*=\s*([-+\d.eE]+)", reply)
                    if not match or float(match[1]) != XY_UNITS_PER_MM:
                        raise RuntimeError(f"{axis} 单位不是预期的 10000 units/mm：{reply}")
                    reply = self.send_xy(f"S {axis}?")
                    match = re.search(rf"\b{axis}\s*=\s*([-+\d.eE]+)", reply)
                    speed = float(match[1]) if match else 0
                    if not math.isfinite(speed) or speed <= 0:
                        raise RuntimeError(f"无法读取 {axis} 速度：{reply}")
                    self.xy_speed[axis] = speed
                    self.speed_ready.emit(axis, speed)
                self.text_ready.emit(f"ASI XY 已连接：{self.xy_port}，沿用控制器当前速度")
            except Exception as e:
                if self.xy_ser is not None:
                    self.xy_ser.close()
                self.xy_ser = None
                self.text_ready.emit(f"XY 连接失败：{e}")

        if self.z_port and self.running:
            try:
                self.z_ser = serial.Serial(self.z_port, BAUDRATE, timeout=0.05, write_timeout=0.5)
                # test.ino 在 UNO 串口打开复位后输出 Ready；此处不发送运动来试探设备。
                reply = self.read_reply(self.z_ser, timeout=4.0)
                if reply != "Ready":
                    raise RuntimeError(f"未收到 test.ino 的 Ready：{reply}")
                self.text_ready.emit(f"Arduino Z 已连接：{self.z_port}，约 1000 步/秒")
            except Exception as e:
                if self.z_ser is not None:
                    self.z_ser.close()
                self.z_ser = None
                self.text_ready.emit(f"Z 连接失败：{e}")

    def show_control_info(self, read_position=False):
        if self.xy_ser is not None and read_position:
            positions = []
            for axis in ("X", "Y"):
                reply = self.send_xy(f"W {axis}")
                parts = reply.split()
                if len(parts) != 2 or parts[0] != ":A":
                    raise RuntimeError(f"ASI 位置回复异常：{reply}")
                position = float(parts[1]) / XY_UNITS_PER_MM
                if not math.isfinite(position):
                    raise RuntimeError("ASI 位置不是有效数值")
                positions.append(f"{axis} {position:.4f} mm")
            self.xy_position = "，".join(positions)
        xy = "XY：未连接"
        if self.xy_ser is not None:
            xy = (f"XY [{self.xy_port}]：{self.xy_position} | 设定速度 "
                  f"X {self.xy_speed['X']:.3f} / Y {self.xy_speed['Y']:.3f} mm/s")
        z = "Z：未连接"
        if self.z_ser is not None:
            z = f"Z [{self.z_port}]：本次指令累计 {self.z_steps:+d} 步（非实测位置）| 脉冲期间约 1000 步/s"
        elif not self.z_position_known:
            z = "Z：连接已关闭，最后一段是否完成未知，请检查设备"
        self.info_ready.emit(xy + "\n" + z)
        self.connection_ready.emit(self.xy_ser is not None, self.z_ser is not None)

    def halt_xy(self):
        """HALT 的中止通知与确认可能粘在一起，也可能分成两行。"""
        reply = self.send_xy("\\")
        # 本机曾返回 N-21:A；-21 仅在主动 HALT 时是预期回复。
        stop_reply = r"(?:(?::?N-21|:?A)\s*)+"
        if not re.fullmatch(stop_reply, reply):
            raise RuntimeError(f"ASI 停止回复异常：{reply}")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.xy_ser.write(b"/\r")
            # 等这一次 / 的状态；消化残留的 HALT 确认，不额外堆积查询。
            while time.monotonic() < deadline:
                reply = self.read_reply(
                    self.xy_ser,
                    timeout=max(0.01, deadline - time.monotonic())
                )
                reply = re.sub("^" + stop_reply, "", reply)
                if reply:
                    break
            if reply == "N":
                return
            if reply != "B":
                raise RuntimeError(f"ASI 停止状态异常：{reply}")
            time.sleep(0.02)
        raise TimeoutError("无法确认 ASI 已停止，请检查控制器")

    def move_xy(self, axis, distance_mm):
        if self.should_stop():
            return False
        # 斜向长按在同一条 R 中带 X/Y；不轮流移动两轴。
        distances = zip(("X", "Y"), distance_mm) if axis == "XY" else [(axis, distance_mm)]
        parts = [f"{a}={round(d * XY_UNITS_PER_MM)}" for a, d in distances if d != 0]
        if not parts:
            return True
        completed = False
        try:
            reply = self.send_xy("R " + " ".join(parts))
            if reply != ":A":
                raise RuntimeError(f"ASI 未接受移动：{reply}")
            # :A 只说明接收成功，还需要等 / 返回 N。
            deadline = time.monotonic() + XY_MOVE_TIMEOUT
            while time.monotonic() < deadline:
                if self.should_stop():
                    return False
                reply = self.send_xy("/")
                if reply == "N":
                    completed = True
                    return True
                if reply != "B":
                    raise RuntimeError(f"ASI 状态异常：{reply}")
                self.cancel_motion.wait(0.02)
            raise TimeoutError("ASI 运动等待超过 10 秒")
        finally:
            if not completed:
                self.halt_xy()

    def move_motor(self, direction, steps):
        """Arduino Z：分短段发送，严格等 Finish，避免松开后还执行一大串步数。"""
        remaining = steps
        while remaining > 0:
            if self.should_stop():
                return False
            count = min(remaining, Z_CHUNK_STEPS)
            # test.ino: 500 us 高 + 500 us 低，约 1000 脉冲/s。
            # 若为 4 mm 导程、200 整步/圈、1/32 细分，脉冲期间约 0.625 mm/s。
            # 每段约 50 ms；等待回复产生的间隙会让实际平均速度略低。
            self.z_position_known = False
            self.z_ser.write(f"{direction} {count}\n".encode("ascii"))
            reply = self.read_reply(self.z_ser, timeout=1.0)
            if reply != "Finish":
                raise RuntimeError(f"Arduino 未确认运动完成：{reply}")
            self.z_position_known = True
            self.z_steps += direction * count
            remaining -= count
        return not self.should_stop()

    def run(self):
        try:
            self.open_serials()
            self.show_control_info()
            if not self.xy_port and not self.z_port:
                self.text_ready.emit("仅相机预览；运动设备未连接")
            with self.command_lock:
                self.busy = False
            self.busy_changed.emit(False)

            while self.running:
                try:
                    command, axis, value = self.commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if self.should_stop():
                        continue
                    if command == "position":
                        axis = "X"
                        self.show_control_info(read_position=True)
                    elif command == "speed":
                        reply = self.send_xy(f"S {axis}={value:.5f}")
                        if reply != ":A":
                            raise RuntimeError(f"ASI 未接受速度设置：{reply}")
                        # 控制器会按自身分辨率/上限调整速度，以回读值为准。
                        reply = self.send_xy(f"S {axis}?")
                        match = re.search(rf"\b{axis}\s*=\s*([-+\d.eE]+)", reply)
                        speed = float(match[1]) if match else 0
                        if not math.isfinite(speed) or speed <= 0:
                            raise RuntimeError(f"无法确认 {axis} 速度：{reply}")
                        self.xy_speed[axis] = speed
                        self.speed_ready.emit(axis, speed)
                        self.text_ready.emit(f"{axis} 速度已设为 {speed:.5f} mm/s")
                    elif command == "move":
                        self.text_ready.emit(f"{axis} 开始移动：{value:+g} {'步' if axis == 'Z' else 'mm'}")
                        if axis == "Z":
                            completed = self.move_motor(1 if value > 0 else -1, abs(int(value)))
                        else:
                            completed = self.move_xy(axis, value)
                        self.text_ready.emit(f"{axis} 移动完成" if completed else f"{axis} 运动已停止")
                    elif command == "jog":
                        self.jogging = True
                        self.jog_started = time.monotonic()
                        self.text_ready.emit(f"{axis} 长按移动，松开停止")
                        message = f"{axis} 长按运动已停止"
                        if axis != "Z":
                            # ASI Console 同样使用按下 R、松开 HALT。
                            # 保留有限终点，串口断开后也不会留下无限速度运动。
                            distance = (tuple(v * JOG_MAX_XY_MM for v in value) if axis == "XY"
                                        else value * JOG_MAX_XY_MM)
                            if self.move_xy(axis, distance):
                                message = "达到单次长按距离/步数上限，请松开后重新按"
                        else:
                            total = 0
                            while not self.should_stop():
                                if total >= JOG_MAX_Z_STEPS:
                                    message = "达到单次长按距离/步数上限，请松开后重新按"
                                    break
                                if not self.move_motor(value, min(Z_CHUNK_STEPS, JOG_MAX_Z_STEPS - total)):
                                    break
                                total += Z_CHUNK_STEPS
                        self.text_ready.emit(message)
                    if axis in ("X", "Y", "XY") and command in ("move", "jog"):
                        self.show_control_info(read_position=True)
                except Exception as e:
                    # 超时后不重试运动：无法判断前一条是否已经执行，必须重新连接。
                    if axis in ("X", "Y", "XY") and self.xy_ser is not None:
                        self.xy_ser.close()
                        self.xy_ser = None
                    if axis == "Z" and self.z_ser is not None:
                        self.z_ser.close()
                        self.z_ser = None
                    self.text_ready.emit(f"控制失败：{e}；请检查设备，重启应用后重新连接")
                finally:
                    self.jogging = False
                    self.show_control_info()
                    with self.command_lock:
                        self.busy = False
                    self.busy_changed.emit(False)
        finally:
            for port in (self.xy_ser, self.z_ser):
                if port is not None:
                    port.close()
            self.xy_ser = None
            self.z_ser = None
            self.connection_ready.emit(False, False)
            self.text_ready.emit("控制线程已退出")
