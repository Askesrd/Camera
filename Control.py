# Control.py
# 统一管理 ASI 的 X/Y 和 Arduino 的 Z，负责手动移动与长按移动。
# XY 仍然使用 mm，Z 仍然使用步数；两套串口协议分别处理

import math
import queue
import re
import threading
import time
import uuid

import serial
from PyQt5.QtCore import QThread, pyqtSignal


BAUDRATE = 115200
XY_UNITS_PER_MM = 10000
Z_CHUNK_STEPS = 50      # 固件不能中途停止，因此每次只发送 50 步
MAX_XY_MOVE_MM = 10.0
MAX_Z_MOVE_STEPS = 10000
XY_MOVE_TIMEOUT = 10.0
XY_MAX_TIMEOUT = 600.0
Z_PULSES_PER_REV = 3200  # 用户已确认的驱动器档位
Z_LEAD_MM = 4.0         # 1204、1:1 传动下的理论换算，不是位移标定
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
    position_ready = pyqtSignal(object)
    result_ready = pyqtSignal(object)

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
        self.connection_id = uuid.uuid4().hex
        self.z_reference_id = None
        self.z_reference_confirmed = False
        self.z_marks = {}
        self.allowed_z_range = None
        self.control_owner = "manual"
        self.view_context_id = uuid.uuid4().hex
        self.request_context = {}

        self.running = True
        self.busy = True    # 串口初始化完成前不接收运动
        self.commands = queue.Queue(maxsize=1)
        self.command_lock = threading.RLock()
        self.cancel_motion = threading.Event()
        self.jogging = False
        self.jog_alive = 0.0
        self.jog_started = 0.0

    def send_command(self, command, axis="", value=0, owner="manual", request_id=None):
        """GUI 只放入一条任务；运动期间不积压点击，也不在这里读写串口。"""
        with self.command_lock:
            if not self.running or self.busy:
                self.text_ready.emit("控制器忙，请等待完成或先停止当前运动")
                return False
            if command != "snapshot" and owner != self.control_owner:
                self.text_ready.emit("运动控制权由其他任务持有，请先停止该任务")
                return False
            if owner != "manual" and self.cancel_motion.is_set() and command != "snapshot":
                self.text_ready.emit("当前任务已取消，不能继续发送运动")
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
            elif command == "z_to":
                if axis != "Z" or type(value) is not int:
                    return False
            elif command in ("reference", "mark"):
                if (axis != "Z" or (command == "mark" and value not in ("A", "B"))
                        or (command == "reference" and type(value) is not bool)):
                    return False
            elif command not in ("position", "snapshot", "invalidate"):
                return False

            port = self.z_ser if axis == "Z" else self.xy_ser
            if command not in ("position", "snapshot", "invalidate") and (port is None or not port.is_open):
                self.text_ready.emit("对应设备未连接，请在启动时选择正确串口")
                return False
            if self.control_owner == "manual":
                self.cancel_motion.clear()
            self.jog_alive = time.monotonic()
            self.busy = True
            self.request_context = {"request_id": request_id or uuid.uuid4().hex, "owner": owner}
            self.commands.put_nowait((command, axis, value))
        self.busy_changed.emit(True)
        return True

    def acquire_control(self, owner):
        """锁住整个任务；停稳、拍照、写盘期间也不允许手动插入运动。"""
        with self.command_lock:
            if not owner or owner == "manual" or not self.running or self.busy or self.control_owner != "manual":
                return False
            self.control_owner = owner
            self.cancel_motion.clear()
            self.position_ready.emit(self.position_snapshot())
            return True

    def release_control(self, owner):
        with self.command_lock:
            if self.busy or self.control_owner != owner:
                return False
            self.control_owner = "manual"
            self.position_ready.emit(self.position_snapshot())
            return True

    def request_z_move(self, target_steps, request_id, owner):
        return self.send_command("z_to", "Z", target_steps, owner, request_id)

    def request_position(self, request_id):
        return self.send_command("snapshot", request_id=request_id)

    def position_snapshot(self):
        """只描述已确认指令，不把累计步数包装成编码器位置。"""
        with self.command_lock:
            return {
                "connection_id": self.connection_id,
                "z_reference_id": self.z_reference_id,
                "z_command_steps": self.z_steps,
                "z_position_valid": self.z_position_known and self.z_ser is not None,
                "z_reference_confirmed": self.z_reference_confirmed,
                "motion_busy": self.busy, "control_owner": self.control_owner,
                "marks": dict(self.z_marks), "allowed_z_range": self.allowed_z_range,
                "pulses_per_rev": Z_PULSES_PER_REV, "lead_mm": Z_LEAD_MM,
                "full_steps_per_rev": 200, "gear_ratio": 1,
                "scale_calibrated": False,
                "position_source": "acknowledged_command_steps",
                "view_context_id": self.view_context_id,
                "state_monotonic_ns": time.monotonic_ns(),
            }

    def invalidate_z_reference(self, reason):
        self.z_reference_id = None
        self.z_reference_confirmed = False
        self.z_marks.clear()
        self.allowed_z_range = None
        self.text_ready.emit(f"Z 参考已失效：{reason}")

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
                self.connection_id = uuid.uuid4().hex
                self.z_steps = 0
                self.z_position_known = True
                self.invalidate_z_reference("新连接，请确认当前软件参考")
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
        self.position_ready.emit(self.position_snapshot())

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
        distances = list(zip(("X", "Y"), distance_mm)) if axis == "XY" else [(axis, distance_mm)]
        parts = [f"{a}={round(d * XY_UNITS_PER_MM)}" for a, d in distances if d != 0]
        if not parts:
            return True
        # 长按独立受 5 秒约束，不能按其有限远端距离拒绝很低的合法速度。
        timeout = JOG_MAX_SECONDS + 1 if self.jogging else self.xy_timeout(distances)
        self.view_context_id = uuid.uuid4().hex  # 已尝试 XY 移动，错误后也不能沿用旧视野
        completed = False
        try:
            reply = self.send_xy("R " + " ".join(parts))
            if reply != ":A":
                raise RuntimeError(f"ASI 未接受移动：{reply}")
            # :A 只说明接收成功，还需要等 / 返回 N。
            deadline = time.monotonic() + timeout
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
            raise TimeoutError(f"ASI 运动等待超过 {timeout:.1f} 秒")
        finally:
            if not completed:
                self.halt_xy()

    def xy_timeout(self, distances):
        seconds = []
        for axis, distance in distances:
            if not distance:
                continue
            speed = self.xy_speed.get(axis, 0)
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError(f"缺少 {axis} 的有效速度回读，未发送运动")
            seconds.append(abs(distance) / speed)
        timeout = max(XY_MOVE_TIMEOUT, 1.5 * max(seconds, default=0) + 2)
        if timeout > XY_MAX_TIMEOUT:
            raise ValueError("预计 XY 移动超过 600 秒，请减小距离或提高速度；未发送运动")
        return timeout

    def move_z_to(self, target_steps):
        """Python 指令绝对定位；UNO 仍只接收方向和相对脉冲数。"""
        if type(target_steps) is not int:
            raise ValueError("Z 目标必须是整数脉冲")
        if not self.z_position_known or not self.z_reference_confirmed:
            raise ValueError("Z 参考无效或尚未确认，不能自动定位")
        if self.allowed_z_range is None:
            raise ValueError("请先手动记录 A/B，建立允许自动移动的范围")
        low, high = self.allowed_z_range
        if not low <= self.z_steps <= high or not low <= target_steps <= high:
            raise ValueError("当前位置或目标超出已确认的 A/B 范围")
        delta = target_steps - self.z_steps
        if abs(delta) > MAX_Z_MOVE_STEPS:
            raise ValueError("Z 定位跨度超过 10000 脉冲，未发送运动")
        if self.should_stop():
            return False
        if delta == 0:
            return True
        return self.move_motor(1 if delta > 0 else -1, abs(delta))

    def move_motor(self, direction, steps):
        """Arduino Z：分短段发送，严格等 Finish，避免松开后还执行一大串步数。"""
        if direction not in (-1, 1) or type(steps) is not int or not 0 < steps <= MAX_Z_MOVE_STEPS:
            raise ValueError("Z 方向或脉冲数无效")
        remaining = steps
        deadline = time.monotonic() + 2 + steps / 1000 + math.ceil(steps / Z_CHUNK_STEPS)
        while remaining > 0:
            if self.should_stop():
                return False
            count = min(remaining, Z_CHUNK_STEPS)
            # test.ino: 500 us 高 + 500 us 低，约 1000 脉冲/s。
            # 当前 3200 脉冲/转，若 4 mm 导程且 1:1 传动，理论约 1.25 mm/s。
            # 每段约 50 ms；等待回复产生的间隙会让实际平均速度略低。
            self.z_position_known = False
            # 只允许上一回复剩下的换行符；额外 Finish/Ready 不能预确认下一段。
            self.check_z_input()
            if time.monotonic() >= deadline:
                raise TimeoutError("Z 逻辑移动总等待超时")
            self.z_ser.write(f"{direction} {count}\n".encode("ascii"))
            reply = self.read_reply(self.z_ser, timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            if reply != "Finish":
                raise RuntimeError(f"Arduino 未确认运动完成：{reply}")
            self.z_steps += direction * count
            self.check_z_input()
            self.z_position_known = True
            remaining -= count
        return not self.should_stop()

    def check_z_input(self):
        """只消耗已确认回复后的 CR/LF；额外消息意味着请求对应关系失效。"""
        waiting = getattr(self.z_ser, "in_waiting", 0)
        if waiting and self.z_ser.read(waiting).strip(b"\r\n"):
            self.z_position_known = False
            self.invalidate_z_reference("Arduino 出现未对应请求的额外回复")
            raise RuntimeError("Arduino 回复失序，不再发送后续脉冲")

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
                    if self.z_ser is not None:
                        try:
                            self.check_z_input()
                        except Exception as error:
                            self.z_position_known = False
                            self.invalidate_z_reference(str(error))
                            self.z_ser.close()
                            self.z_ser = None
                            self.show_control_info()
                    continue
                context = dict(self.request_context)
                result = {**context, "command": command, "axis": axis,
                          "target_steps": value if command == "z_to" else None,
                          "status": "cancelled", "error": None}
                try:
                    if self.should_stop():
                        continue
                    if command == "position":
                        axis = "X"
                        self.show_control_info(read_position=True)
                    elif command == "snapshot":
                        pass
                    elif command == "invalidate":
                        self.invalidate_z_reference(str(value))
                        self.z_position_known = False
                    elif command == "reference":
                        if not self.z_position_known:
                            raise ValueError("Z 累计已不可信，请检查设备并重新连接")
                        if value:  # 显式“当前位置设为零”，只修改软件参考
                            self.z_steps = 0
                        self.z_reference_id = uuid.uuid4().hex
                        self.z_reference_confirmed = True
                        self.z_marks.clear()
                        self.allowed_z_range = None
                        self.text_ready.emit("已建立新的 Z 软件参考，未发送运动；请重新记录 A/B")
                    elif command == "mark":
                        if not self.z_position_known or not self.z_reference_confirmed:
                            raise ValueError("请先确认当前 Z 软件参考")
                        marks = {**self.z_marks, value: self.z_steps}
                        if len(marks) == 2 and abs(marks["B"] - marks["A"]) > MAX_Z_MOVE_STEPS:
                            raise ValueError("A/B 跨度不能超过 10000 脉冲")
                        self.z_marks = marks
                        if len(marks) == 2:
                            self.allowed_z_range = (min(marks.values()), max(marks.values()))
                        self.text_ready.emit(f"已记录 {value}：{self.z_steps:+d} 脉冲")
                    elif command == "z_to":
                        completed = self.move_z_to(value)
                        result["status"] = "completed" if completed else "cancelled"
                        self.text_ready.emit("Z 目标指令已完成" if completed else "Z 定位已取消")
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
                        result["status"] = "completed" if completed else "cancelled"
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
                        result["status"] = "cancelled" if self.should_stop() else "completed"
                    if command not in ("move", "jog", "z_to"):
                        result["status"] = "completed"
                    if axis in ("X", "Y", "XY") and command in ("move", "jog"):
                        self.show_control_info(read_position=True)
                except Exception as e:
                    # 超时后不重试运动：无法判断前一条是否已经执行，必须重新连接。
                    if not isinstance(e, ValueError) and axis in ("X", "Y", "XY") and self.xy_ser is not None:
                        self.xy_ser.close()
                        self.xy_ser = None
                    if axis == "Z" and self.z_ser is not None and (not isinstance(e, ValueError) or not self.z_position_known):
                        self.z_ser.close()
                        self.z_ser = None
                        self.z_position_known = False
                        self.invalidate_z_reference(str(e))
                    result["status"], result["error"] = "error", str(e)
                    self.text_ready.emit(f"控制请求失败：{e}")
                finally:
                    self.jogging = False
                    with self.command_lock:
                        self.busy = False
                    self.show_control_info()
                    result["position"] = self.position_snapshot()
                    result["finished_monotonic_ns"] = time.monotonic_ns()
                    self.result_ready.emit(result)
                    self.busy_changed.emit(False)
        finally:
            for port in (self.xy_ser, self.z_ser):
                if port is not None:
                    port.close()
            self.xy_ser = None
            self.z_ser = None
            self.z_position_known = False
            self.invalidate_z_reference("控制连接已关闭")
            with self.command_lock:
                self.busy = False
            if not self.commands.empty():
                command, axis, value = self.commands.get_nowait()
                self.result_ready.emit({**self.request_context, "command": command, "axis": axis,
                    "target_steps": value if command == "z_to" else None, "status": "cancelled",
                    "error": None, "position": self.position_snapshot(),
                    "finished_monotonic_ns": time.monotonic_ns()})
            self.position_ready.emit(self.position_snapshot())
            self.busy_changed.emit(False)
            self.connection_ready.emit(False, False)
            self.text_ready.emit("控制线程已退出")
