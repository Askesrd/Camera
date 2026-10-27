# Camera.py
# 相机采集、参数回读和图像保存；SDK 只在工作线程内访问。

import ctypes
import copy
import json
import math
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

import toupcam
from PyQt5.QtCore import QThread, pyqtSignal
from ScanStorage import create_session, write_image


SAVE_DIR = Path.home() / "Desktop" / "111"
WAIT_FRAME_HRESULTS = {0x8001011F, 0x8000000A}
MODE_FPS = {0: 15, 1: 50, 2: 60}


class CameraThread(QThread):
    # ==============================
    # 多线程输出接口：相机线程 → GUI
    # ==============================
    frame_ready = pyqtSignal(object)  # 发送图像 frame
    text_ready = pyqtSignal(str)      # 发送状态文字
    info_ready = pyqtSignal(str)      # 发送分辨率和帧率信息
    exposure_ready = pyqtSignal(int, bool, float, float, float)  # 请求编号、自动、时间/最小/最大（ms）
    gain_ready = pyqtSignal(int, int, int, int)  # 请求编号、当前/最小/最大（百分数）
    diagnostic_ready = pyqtSignal(object)
    camera_result = pyqtSignal(object)
    capture_lock_changed = pyqtSignal(bool)

    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        self.fps = MODE_FPS[mode]

        # ==============================
        # 多线程输入接口：GUI → 相机线程
        # GUI 发来的文本命令会放到这个队列
        # ==============================
        self.commands = queue.Queue()
        # 长按可能连续产生设置，只保存最新的一次，不积压旧曝光值。
        self.exposure_lock = threading.RLock()
        self.exposure_request = None
        self.exposure_request_id = 0
        self.exposure_applied_id = 0
        self.exposure_min = 0
        self.exposure_max = 0
        self.exposure_last_time = 0.0
        self.gain_request = None
        self.gain_request_id = 0
        self.gain_applied_id = 0
        self.gain_min = self.gain_max = None
        self.current_settings = {}
        self.camera_owner = "manual"
        self.camera_busy = False
        self.manual_processing = False
        self.camera_requests = queue.Queue(maxsize=1)
        self.capture_cancel = threading.Event()
        self.freshness_profile = None  # 实机未验证前不宣称正式新帧采集
        self.capture_policy = "validated"
        self.locked_settings = None
        self.diagnostics = {}
        self.camera_stream_id = None
        self.host_frame_id = 0
        self.frame_info_api = None
        self.preview_frame = None
        self.save_session = None

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
        if text == "stop":
            self.running = False
            self.capture_cancel.set()
        with self.exposure_lock:
            if text != "stop" and self.camera_owner != "manual":
                self.text_ready.emit("相机由当前采集任务持有，请先结束该任务")
                return False
            self.commands.put(text)
        return True

    def set_exposure(self, auto, milliseconds):
        """GUI 只提交请求；相机 SDK 的读写仍由采集线程执行。"""
        with self.exposure_lock:
            if self.camera_owner != "manual" or self.camera_busy:
                self.text_ready.emit("曝光由当前采集任务锁定")
                return self.exposure_request_id
            self.exposure_request_id += 1
            self.exposure_request = (self.exposure_request_id, auto, round(milliseconds * 1000))
            return self.exposure_request_id

    def set_gain(self, percent):
        with self.exposure_lock:
            if self.camera_owner != "manual" or self.camera_busy:
                self.text_ready.emit("增益由当前采集任务锁定")
                return self.gain_request_id
            self.gain_request_id += 1
            self.gain_request = (self.gain_request_id, percent)
            return self.gain_request_id

    def take_preview_frame(self):
        """GUI 定时取最新图；不在 Qt 事件队列中积压全分辨率数组。"""
        with self.exposure_lock:
            frame = self.preview_frame
            self.preview_frame = None
            return frame

    def acquire_capture(self, owner):
        with self.exposure_lock:
            if (not owner or owner == "manual" or not self.running or self.cam is None
                    or self.camera_owner != "manual" or self.camera_busy
                    or self.manual_processing
                    or self.recording or self.auto_capture or not self.commands.empty()
                    or self.one_frame_ready
                    or self.exposure_request is not None or self.gain_request is not None):
                return False
            self.camera_owner = owner
            self.capture_cancel.clear()
            self.locked_settings = None
            self.capture_policy = "validated"
            self.capture_lock_changed.emit(True)
            return True

    def request_camera_settings(self, config, request_id, owner):
        return self.queue_camera_request({"kind": "settings", "config": copy.deepcopy(config),
                                         "request_id": request_id, "owner": owner})

    def request_capture(self, request_id, run_id, sample_id, owner, count=3, debug=False):
        return self.queue_camera_request({"kind": "capture", "request_id": request_id,
                    "run_id": run_id, "sample_id": sample_id, "owner": owner,
                    "count": count, "debug": debug})

    def release_capture(self, owner, request_id):
        return self.queue_camera_request({"kind": "release", "request_id": request_id, "owner": owner})

    def queue_camera_request(self, request):
        # 普通方法只入队，SDK 的设置和采图仍然在 run() 所在线程执行。
        with self.exposure_lock:
            if (not self.running or self.camera_busy or self.camera_owner == "manual"
                    or request["owner"] != self.camera_owner):
                return False
            if request["kind"] == "capture" and (
                    request["run_id"] != self.camera_owner
                    or type(request["sample_id"]) is not int or request["sample_id"] < 0
                    or type(request["debug"]) is not bool):
                return False
            self.camera_busy = True
            self.camera_requests.put_nowait(request)
            return True

    def cancel_capture(self, owner=None):
        if owner is None or owner == self.camera_owner:
            self.capture_cancel.set()

    def read_camera_settings(self, processing=False):
        settings = {"ae_enabled": bool(self.cam.get_AutoExpoEnable()),
                    "exposure_us": self.cam.get_ExpoTime(), "gain_percent": None}
        try:
            settings["real_exposure_us"] = self.cam.get_RealExpoTime()
        except (toupcam.HRESULTException, AttributeError):
            settings["real_exposure_us"] = settings["exposure_us"]
        if self.gain_min is not None:
            settings["gain_percent"] = self.cam.get_ExpoAGain()
        if processing:
            for key, method in (("gamma", "get_Gamma"), ("temp_tint", "get_TempTint"),
                                ("white_balance_gain", "get_WhiteBalanceGain")):
                try:
                    settings[key] = getattr(self.cam, method)()
                except (toupcam.HRESULTException, AttributeError):
                    settings[key] = None
        self.current_settings = dict(settings)
        return settings

    def show_exposure(self, force=False):
        """即使尚无新帧，也每 0.2 秒更新自动曝光的当前时间。"""
        now = time.monotonic()
        if not force and now - self.exposure_last_time < 0.2:
            return
        self.exposure_last_time = now
        settings = self.read_camera_settings()
        auto, exposure_us = settings["ae_enabled"], settings["real_exposure_us"]
        self.exposure_ready.emit(self.exposure_applied_id, auto, exposure_us / 1000,
                                 self.exposure_min / 1000, self.exposure_max / 1000)
        if self.gain_min is not None:
            self.gain_ready.emit(self.gain_applied_id, settings["gain_percent"], self.gain_min, self.gain_max)

    # ==================================================
    # 线程主函数
    # 调用 start() 后，Qt 会自动在后台线程里执行 run()
    # ==================================================
    def run(self):
        try:
            # 1. 打开相机
            if not self.running:
                return
            devices = toupcam.Toupcam.EnumV2()
            if not devices:
                self.text_ready.emit("未发现相机")
                return
            device = devices[0]
            self.cam = toupcam.Toupcam.Open(device.id)
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

            self.exposure_min, self.exposure_max, _ = self.cam.get_ExpTimeRange()
            try:
                self.gain_min, self.gain_max, _ = self.cam.get_ExpoAGainRange()
            except (toupcam.HRESULTException, AttributeError):
                self.gain_min = self.gain_max = None
            # 每次启动明确恢复连续自动曝光，不沿用上次手动状态。
            self.cam.put_AutoExpoEnable(1)
            self.cam.StartPullModeWithCallback(None, None)
            self.camera_stream_id = uuid.uuid4().hex
            self.diagnostics = {
                "camera_id": device.id, "camera_model": device.model.name,
                "sdk_version": toupcam.Toupcam.Version(), "capability_flags": int(device.model.flag),
                "software_trigger_supported": bool(device.model.flag & toupcam.TOUPCAM_FLAG_TRIGGER_SOFTWARE),
                "trigger_timing_verified": False, "freshness_method": "unverified",
                "mode": self.mode, "width": self.width, "height": self.height,
                "pixel_format": "BGR24", "camera_stream_id": self.camera_stream_id,
                "camera_environment_id": uuid.uuid4().hex,
                "frame_info_api": None, "seq_valid": False, "timestamp_valid": False,
                "flush_supported": None,
                "buffer_config": {},
            }
            for name, option in (("ddr_depth", toupcam.TOUPCAM_OPTION_DDR_DEPTH),
                                 ("frontend", toupcam.TOUPCAM_OPTION_FRONTEND_DEQUE_LENGTH),
                                 ("backend", toupcam.TOUPCAM_OPTION_BACKEND_DEQUE_LENGTH)):
                try:
                    self.diagnostics["buffer_config"][name] = self.cam.get_Option(option)
                except (toupcam.HRESULTException, AttributeError):
                    self.diagnostics["buffer_config"][name] = None
            self.diagnostic_ready.emit(dict(self.diagnostics))
            self.show_exposure(force=True)

            self.text_ready.emit(f"相机已启动")


            while self.running:
                # 处理 GUI 发来的文本命令
                with self.exposure_lock:
                    self.manual_processing = True
                try:
                    self.handle_text_commands()
                finally:
                    with self.exposure_lock:
                        self.manual_processing = False
                self.handle_camera_request()
                if not self.running:
                    break
                self.show_exposure()

                # 拉取相机图像
                packet = self.read_frame()
                if packet is None:
                    time.sleep(0.001)
                    continue
                frame = packet["frame"]
                with self.exposure_lock:
                    self.preview_frame = frame

                # ==============================
                # 多线程输出接口：发送图像给 GUI
                # ==============================
                self.frame_ready.emit(frame)

                # 把当前帧交给命令处理函数，由它统一判断录像、单张保存、自动截图等任务。
                with self.exposure_lock:
                    self.manual_processing = True
                try:
                    self.handle_text_commands(frame)
                finally:
                    with self.exposure_lock:
                        self.manual_processing = False

                # 每秒刷新一次帧率，曝光时间通过独立信号更新。
                self.show_camera_info()

        except Exception as e:
            self.text_ready.emit(f"相机线程错误：{e}")

        finally:
            self.cleanup()

    def read_frame(self):
        """复制像素和有效的 SDK 帧信息；缺失字段保留 None，不伪造 0。"""
        use_v3 = self.frame_info_api != "V2" and hasattr(self.cam, "PullImageV3")
        info = toupcam.ToupcamFrameInfoV3() if use_v3 else toupcam.ToupcamFrameInfoV2()
        try:
            if use_v3:
                self.cam.PullImageV3(self.buf, 0, 24, self.row_pitch, info)
            else:
                self.cam.PullImageV2(self.buf, 24, info)
        except (toupcam.HRESULTException, AttributeError) as error:
            code = getattr(error, "hr", 0) & 0xFFFFFFFF
            if code in WAIT_FRAME_HRESULTS:
                return None
            if use_v3 and (isinstance(error, AttributeError) or code == toupcam.E_NOTIMPL):
                self.frame_info_api = "V2"
                return self.read_frame()
            raise
        if info.width != self.width or info.height != self.height:
            raise RuntimeError("相机帧尺寸意外变化，本次采集无效")
        self.host_frame_id += 1
        flags = info.flag
        metadata = {
            "host_frame_id": self.host_frame_id, "camera_stream_id": self.camera_stream_id,
            "camera_seq": info.seq if flags & toupcam.TOUPCAM_FRAMEINFO_FLAG_SEQ else None,
            "camera_timestamp_us": info.timestamp if flags & toupcam.TOUPCAM_FRAMEINFO_FLAG_TIMESTAMP else None,
            "host_received_monotonic_ns": time.monotonic_ns(),
            "host_received_utc": datetime.now(timezone.utc).isoformat(),
            "frame_flags": flags, "frame_info_api": "V3" if use_v3 else "V2",
            "exposure_us": self.current_settings.get("real_exposure_us"),
            "gain_percent": self.current_settings.get("gain_percent"),
            "exposure_source": "settings_readback", "gain_source": "settings_readback",
        }
        if use_v3 and flags & toupcam.TOUPCAM_FRAMEINFO_FLAG_EXPOTIME:
            metadata.update(exposure_us=info.expotime, exposure_source="frame_metadata")
        if use_v3 and flags & toupcam.TOUPCAM_FRAMEINFO_FLAG_EXPOGAIN:
            metadata.update(gain_percent=info.expogain, gain_source="frame_metadata")
        old = (self.diagnostics.get("frame_info_api"), self.diagnostics.get("seq_valid"), self.diagnostics.get("timestamp_valid"))
        self.frame_info_api = metadata["frame_info_api"]
        self.diagnostics.update(frame_info_api=self.frame_info_api,
                seq_valid=metadata["camera_seq"] is not None,
                timestamp_valid=metadata["camera_timestamp_us"] is not None)
        new = (self.frame_info_api, self.diagnostics["seq_valid"], self.diagnostics["timestamp_valid"])
        if old != new:
            self.diagnostic_ready.emit(dict(self.diagnostics))
        return {"frame": self.buffer_to_frame(), "metadata": metadata}

    def handle_camera_request(self):
        """一次处理一项任务；所有结果带原请求身份，忙状态不代表成功。"""
        try:
            request = self.camera_requests.get_nowait()
        except queue.Empty:
            return
        result = {key: value for key, value in request.items() if key != "config"}
        result.update(status="cancelled", error=None)
        try:
            if request["kind"] == "release":
                with self.exposure_lock:
                    if self.capture_cancel.is_set():
                        self.diagnostics["freshness_method"] = "unverified"
                        self.diagnostic_ready.emit(copy.deepcopy(self.diagnostics))
                    self.camera_owner = "manual"
                    self.locked_settings = None
                    self.freshness_profile = None
                    self.capture_policy = "validated"
                    self.capture_cancel.clear()
                self.capture_lock_changed.emit(False)
                result["status"] = "completed"
            elif not self.running or self.capture_cancel.is_set():
                pass
            elif request["kind"] == "settings":
                if self.locked_settings is not None:
                    raise ValueError("本任务参数已锁定；修改曝光请结束当前任务并新建 Run")
                config = request["config"]
                policy = config.get("capture_policy", "validated")
                if policy not in ("validated", "continuous_auto"):
                    raise ValueError("不支持的采集策略")
                exposure, gain = config.get("exposure_us"), config.get("gain_percent")
                if (type(exposure) is not int or not self.exposure_min <= exposure <= self.exposure_max
                        or self.gain_min is None or type(gain) is not int
                        or not self.gain_min <= gain <= self.gain_max):
                    raise ValueError("曝光或增益超出设备范围，未应用采集参数")
                if policy == "continuous_auto":
                    current = self.read_camera_settings()
                    if not current["ae_enabled"] and exposure == current["real_exposure_us"]:
                        exposure = current["exposure_us"]  # 重用原命令，避免回读时间反复量化。
                self.cam.put_AutoExpoEnable(0)
                if not self.running or self.capture_cancel.is_set():
                    return
                if self.cam.get_AutoExpoEnable():
                    raise RuntimeError("无法关闭自动曝光，不能进行固定参数采集")
                self.cam.put_ExpoTime(exposure)
                if not self.running or self.capture_cancel.is_set():
                    return
                self.cam.put_ExpoAGain(gain)
                settings = self.read_camera_settings(processing=True)
                if settings["ae_enabled"] or settings["gain_percent"] is None:
                    raise RuntimeError("无法确认固定曝光/增益状态")
                self.locked_settings = copy.deepcopy(settings)
                self.capture_policy = policy
                self.freshness_profile = copy.deepcopy(config.get("freshness_profile")) if policy == "validated" else None
                capture_info = None
                if config.get("formal", False):
                    method, guard_ms, timeout = self.check_freshness_profile()
                    capture_info = {"policy": policy, "freshness_method": method,
                        "freshness_verified": method == "continuous_validated",
                        "fresh_guard_ms": guard_ms, "timeout_seconds": timeout}
                result.update(status="completed", requested=copy.deepcopy(config), readback=settings,
                              diagnostic=copy.deepcopy(self.diagnostics), capture_info=capture_info)
                self.show_exposure(force=True)
            elif request["kind"] == "capture":
                packets, method = self.capture_frames(request["count"], request["debug"])
                if self.running and not self.capture_cancel.is_set():
                    for packet in packets:
                        packet["metadata"].update(request_id=request["request_id"],
                            run_id=request["run_id"], sample_id=request["sample_id"])
                    result.update(status="debug_only" if method == "unverified" else "completed",
                                  frames=packets, freshness_method=method,
                                  freshness_verified=method == "continuous_validated",
                                  freshness_profile=copy.deepcopy(self.freshness_profile),
                                  settings=copy.deepcopy(self.locked_settings))
        except Exception as error:
            if request["kind"] == "settings":
                self.locked_settings = None
            if isinstance(error, InterruptedError) and self.capture_cancel.is_set():
                result.update(status="cancelled", error=None)
            else:
                result.update(status="error", error=str(error))
                self.text_ready.emit(f"相机采集请求失败：{error}")
        finally:
            if request["kind"] != "release" and (not self.running or self.capture_cancel.is_set()):
                result["status"] = "cancelled"
                result.pop("frames", None)
            with self.exposure_lock:
                self.camera_busy = False
            result["finished_monotonic_ns"] = time.monotonic_ns()
            self.camera_result.emit(result)

    def check_freshness_profile(self, debug=False):
        """准备等待和截止期限；自动策略是试扫估算，不生成验证结论。"""
        if not self.locked_settings or self.locked_settings["ae_enabled"]:
            raise ValueError("请先应用并锁定手动曝光和增益")
        automatic = self.capture_policy == "continuous_auto"
        profile = {} if automatic else (self.freshness_profile or {})
        guard_ms = math.ceil(max(200, 2 * self.locked_settings["real_exposure_us"] / 1000)) if automatic else profile.get("fresh_guard_ms", 0)
        if (type(guard_ms) not in (int, float) or not math.isfinite(guard_ms)
                or not 0 <= guard_ms <= 60000):
            raise ValueError("新帧等待窗口必须在 0–60000 ms 内")
        verified = profile.get("verified") is True and profile.get("method") == "continuous_validated"
        if verified:
            # 验证记录必须对应当前设备、分辨率及缓冲配置；缺项也不能默认匹配。
            for key in ("camera_id", "sdk_version", "camera_environment_id", "mode", "width",
                        "height", "pixel_format", "buffer_config"):
                if key not in profile or profile[key] != self.diagnostics.get(key):
                    raise ValueError(f"新帧验证条件已改变或缺失：{key}")
            maximum = profile.get("max_exposure_us")
            if (type(maximum) not in (int, float) or not math.isfinite(maximum)
                    or maximum < self.locked_settings["real_exposure_us"]
                    or not profile.get("conditions_note") or not profile.get("verification_note")
                    or guard_ms <= 0):
                raise ValueError("曝光超出验证范围，或新帧验证记录不完整")
            if "check_settings" in profile and json.dumps(profile["check_settings"], sort_keys=True) != json.dumps(self.locked_settings, sort_keys=True):
                raise ValueError("历史验证记录的固定参数与当前曝光、增益或图像处理设置不一致")
        elif not debug and not automatic:
            raise ValueError("当前相机的新帧时序未经过实机验证，不能作为正式采集成功")
        method = "continuous_unverified" if automatic else ("continuous_validated" if verified else "unverified")
        exposure_seconds = self.locked_settings["real_exposure_us"] / 1000000
        timeout = guard_ms / 1000 + 3 * (1.02 * exposure_seconds + 4)
        if timeout > 600:
            raise ValueError("三帧采集预计等待超过 600 秒，未开始采集")
        # FLUSH 清理缓存，但不保证中断正在曝光的帧；仍须等待已验证窗口。
        try:
            self.cam.put_Option(toupcam.TOUPCAM_OPTION_FLUSH, 3)
            self.diagnostics["flush_supported"] = True
        except (toupcam.HRESULTException, AttributeError):
            self.diagnostics["flush_supported"] = False
        self.diagnostic_ready.emit(dict(self.diagnostics))
        if verified and profile.get("flush_supported") != self.diagnostics["flush_supported"]:
            raise ValueError("缓存清理能力与新帧验证记录不符")
        self.diagnostics["freshness_method"] = method
        self.diagnostic_ready.emit(dict(self.diagnostics))
        return method, guard_ms, timeout

    def capture_frames(self, count=3, debug=False):
        """预览始终更新；等待内丢弃的图只显示，不进入三帧采样集合。"""
        if type(count) is not int or count != 3:
            raise ValueError("每个采样请求固定收集三帧")
        if self.read_camera_settings(processing=True) != self.locked_settings:
            raise RuntimeError("曝光、增益或图像处理设置已改变")
        method, guard_ms, timeout = self.check_freshness_profile(debug)
        verified = method == "continuous_validated"
        self.text_ready.emit(f"采图：等待新帧 {guard_ms:.0f} ms，然后收集三帧")
        stream_id = self.camera_stream_id
        barrier = time.monotonic()
        fresh_after = barrier + guard_ms / 1000
        deadline = barrier + timeout
        packets, seen_seq, seen_timestamps = [], set(), set()
        while time.monotonic() < deadline:
            if not self.running or self.capture_cancel.is_set():
                return [], method
            if stream_id != self.camera_stream_id:
                raise RuntimeError("相机流意外重启，本点采集无效")
            packet = self.read_frame()
            if packet is None:
                self.capture_cancel.wait(0.002)
                continue
            metadata = packet["metadata"]
            with self.exposure_lock:
                self.preview_frame = packet["frame"]
            if metadata["camera_stream_id"] != stream_id:
                raise RuntimeError("采集帧来自不同的相机流")
            seq, timestamp = metadata["camera_seq"], metadata["camera_timestamp_us"]
            duplicate = (seq is not None and seq in seen_seq) or (timestamp is not None and timestamp in seen_timestamps)
            if seq is not None:
                seen_seq.add(seq)
            if timestamp is not None:
                seen_timestamps.add(timestamp)
            # 等待窗口内的帧也登记身份，避免之后的重复缓冲被接收。
            if time.monotonic() < fresh_after or duplicate:
                continue
            if method != "unverified" and seq is None and timestamp is None:
                raise RuntimeError("没有有效设备帧身份，无法确认三张不同的新帧")
            if self.read_camera_settings(processing=True) != self.locked_settings:
                raise RuntimeError("采集期间曝光、增益或图像处理参数失锁")
            if (metadata["exposure_us"] != self.locked_settings["real_exposure_us"]
                    or metadata["gain_percent"] != self.locked_settings["gain_percent"]):
                raise RuntimeError("帧的曝光或增益与锁定参数不一致")
            metadata.update(freshness_method=method, freshness_verified=verified,
                            fresh_guard_ms=guard_ms, barrier_monotonic_ns=int(barrier * 1000000000))
            packets.append(packet)
            self.text_ready.emit(f"采图：已收到 {len(packets)} / {count} 帧")
            if len(packets) == count:
                return packets, method
        raise TimeoutError("未在曝光相关期限内收到三张有效新帧")

    def show_camera_info(self):
        """
        显示相机实时信息。

        Python 帧率：本程序成功拉取预览帧的速度。
        SDK 帧率：相机 SDK 自己统计的真实输出速度。
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

        self.info_ready.emit(
            f"相机信息：分辨率 {self.width} x {self.height} | "
            f"计算帧率 {python_fps:.2f} fps | "
            f"SDK帧率 {sdk_fps:.2f} fps"
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

            elif not self.running:
                continue

            elif self.camera_owner != "manual":
                self.text_ready.emit("当前采集任务持有相机，已忽略手动采集命令")

            elif text == "save":
                self.save_one_frame()

            elif text == "record_start":
                self.handle_record("start")

            elif text == "record_stop":
                self.handle_record("stop")

            elif text == "auto":
                self.handle_auto_capture()

        with self.exposure_lock:
            request = self.exposure_request
            self.exposure_request = None
            gain_request = self.gain_request
            self.gain_request = None
        if request is not None and self.running:
            self.diagnostics["freshness_method"] = "unverified"
            self.diagnostic_ready.emit(copy.deepcopy(self.diagnostics))
            request_id, auto, exposure_us = request
            self.exposure_applied_id = request_id
            try:
                # 必须先关闭自动曝光，手动时间才不会马上被自动算法改写。
                self.cam.put_AutoExpoEnable(1 if auto else 0)
                if not auto:
                    exposure_us = max(self.exposure_min, min(exposure_us, self.exposure_max))
                    self.cam.put_ExpoTime(exposure_us)
            except toupcam.HRESULTException as e:
                self.text_ready.emit(f"曝光设置失败：{e}")
            # 成功或失败都读回设备状态，避免界面显示未生效的设置。
            self.show_exposure(force=True)

        if gain_request is not None and self.running:
            self.diagnostics["freshness_method"] = "unverified"
            self.diagnostic_ready.emit(copy.deepcopy(self.diagnostics))
            request_id, gain = gain_request
            self.gain_applied_id = request_id
            try:
                if self.cam.get_AutoExpoEnable():
                    raise ValueError("请先关闭自动曝光，再设置固定增益")
                if self.gain_min is None or type(gain) is not int or not self.gain_min <= gain <= self.gain_max:
                    raise ValueError("增益超出设备范围或不支持增益设置")
                self.cam.put_ExpoAGain(gain)
            except (toupcam.HRESULTException, ValueError) as error:
                self.text_ready.emit(f"增益设置失败：{error}")
            self.show_exposure(force=True)

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

            self.video_count += 1
            try:
                path = self.get_output_dir() / f"video-{self.video_count}-{uuid.uuid4().hex[:12]}.mp4"
            except OSError as error:
                self.text_ready.emit(f"录像启动失败：{error}")
                return

            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self.out = cv.VideoWriter(
                str(path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

            if not self.out.isOpened():
                self.text_ready.emit("录像启动失败")
                self.out.release()
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
        try:
            path = self.get_output_dir() / f"single-{self.one_frame}-{uuid.uuid4().hex[:12]}.jpg"
            write_image(path, frame)
            self.text_ready.emit(f"单张图片保存：{path}")
        except (OSError, ValueError, cv.error) as error:
            self.text_ready.emit(f"单张图片保存失败：{error}")

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
            self.auto_start = time.monotonic()
            self.auto_last = self.auto_start

            self.text_ready.emit("开始自动截图")
            return

        if not self.auto_capture:
            return

        now = time.monotonic()

        if now - self.auto_last >= self.auto_interval:
            self.image_count += 1
            self.auto_last = now

            try:
                path = self.get_output_dir() / f"auto-{self.session_count}-{self.image_count}-{uuid.uuid4().hex[:12]}.jpg"
                write_image(path, frame)
                self.text_ready.emit(f"自动截图保存：{path}")
            except (OSError, ValueError, cv.error) as error:
                self.auto_capture = False
                self.text_ready.emit(f"自动截图失败：{error}")
                return

        if now - self.auto_start >= self.auto_duration:
            self.auto_capture = False
            self.text_ready.emit("自动截图结束")

    def get_output_dir(self):
        # 每次启动首次保存时新建 Session，不复用上次的截图/录像编号。
        if self.save_session is None:
            self.save_session = create_session(SAVE_DIR, "手动采集")
        return self.save_session

    def cleanup(self):
        """
        退出时释放相机和视频资源
        """
        self.running = False
        self.capture_cancel.set()
        self.handle_camera_request()  # 排队但尚未开始的请求也有明确的取消结果
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
        with self.exposure_lock:
            self.preview_frame = None
            self.locked_settings = None
            self.freshness_profile = None
            self.capture_policy = "validated"
            self.camera_owner = "manual"
        self.text_ready.emit("相机线程已退出")
        self.capture_lock_changed.emit(False)
