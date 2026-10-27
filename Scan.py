# 一次只处理一个点。此线程调度现有 worker，不接触串口或相机 SDK 句柄。
import copy
import json
import math
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal

from Control import MAX_Z_MOVE_STEPS, Z_CHUNK_STEPS
from Laplacian import tenengrad
from ScanStorage import create_session, create_run, write_json, write_image, commit_sample, read_run


MAX_SCAN_POINTS = 2000


def make_scan_plan(a, b, interval, width, height, roi=None):
    if any(type(v) is not int for v in (a, b, interval, width, height)):
        raise ValueError("端点、间隔和图像尺寸必须是整数")
    span = abs(b - a)
    if not 0 < span <= MAX_Z_MOVE_STEPS or not 0 < interval <= MAX_Z_MOVE_STEPS:
        raise ValueError("A/B 不能相同，跨度和间隔须在 1–10000 脉冲内")
    count = span // interval + 1 + (span % interval != 0)
    if count > MAX_SCAN_POINTS or width <= 0 or height <= 0:
        raise ValueError("计划超过 2000 点，或相机尺寸无效")
    if roi is not None:
        if (len(roi) != 4 or any(type(v) is not int for v in roi)
                or roi[0] < 0 or roi[1] < 0 or min(roi[2:]) < 8
                or roi[0] + roi[2] > width or roi[1] + roi[3] > height):
            raise ValueError("ROI 越界或小于 8×8 原始像素")
    direction = 1 if b > a else -1
    targets = list(range(a, b, direction * interval)) + [b]
    raw_bytes = count * width * height * 3
    return {"a_steps": a, "b_steps": b, "interval_steps": interval, "targets": targets,
            "target_count": count, "scan_direction": direction,
            "last_interval_steps": abs(targets[-1] - targets[-2]),
            "raw_image_bytes": raw_bytes, "memory_estimate_bytes": width * height * 3 * 16,
            "disk_estimate_bytes": math.ceil(raw_bytes * 1.1) + count * 16384 + 1048576}


class ScanThread(QThread):
    state_changed = pyqtSignal(str)
    prepared = pyqtSignal(object)
    sample_ready = pyqtSignal(object)
    task_result = pyqtSignal(object)

    def __init__(self, control, camera):
        super().__init__()
        self.control, self.camera = control, camera
        self.config = None
        self.owner = None
        self.cancelled = threading.Event()
        self.result_lock = threading.RLock()
        self.expected = None
        self.results = queue.Queue(maxsize=1)
        self.control_owned = self.camera_owned = False
        self.pending_timeout = 0
        self.session_dir = None
        self.session_context = None
        self.run_dir = None
        self.emitted_ids = set()
        # DirectConnection 只做有界入队，不把三张原图排进 GUI 的事件队列。
        control.result_ready.connect(self.on_control_result, Qt.DirectConnection)
        camera.camera_result.connect(self.on_camera_result, Qt.DirectConnection)

    def configure(self, config):
        if self.isRunning() or self.control_owned or self.camera_owned:
            return False
        self.config = copy.deepcopy(config)
        self.owner = None
        self.run_dir = None
        self.emitted_ids.clear()
        self.cancelled.clear()
        return True

    def cancel_task(self):
        self.cancelled.set()
        if self.control_owned:
            self.control.stop_motion()
        if self.camera_owned:
            self.camera.cancel_capture(self.owner)

    def on_control_result(self, result):
        self.accept_result("control", result)

    def on_camera_result(self, result):
        self.accept_result("camera", result)

    def accept_result(self, source, result):
        with self.result_lock:
            if self.expected != (source, result.get("request_id")):
                return
            if result.get("owner") != self.owner and not (source == "control" and result.get("command") == "snapshot"):
                return
            if result.get("kind") == "capture" and result.get("run_id") != self.owner:
                return
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass  # 重复回调不能增加在途结果数量

    def request(self, source, submit, timeout, cleanup=False):
        if self.cancelled.is_set() and not cleanup:
            raise InterruptedError("扫描已取消")
        request_id = uuid.uuid4().hex
        with self.result_lock:
            self.expected = (source, request_id)
            while not self.results.empty():
                self.results.get_nowait()
        self.pending_timeout = timeout
        try:
            if not submit(request_id):
                raise RuntimeError("设备拒绝请求，可能仍在忙或已被其他任务持有")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    result = self.results.get(timeout=0.03)
                except queue.Empty:
                    device = self.control if source == "control" else self.camera
                    if not device.running:
                        raise RuntimeError("控制器或相机线程已退出")
                    continue
                if not cleanup and (self.cancelled.is_set() or result["status"] == "cancelled"):
                    raise InterruptedError("扫描已取消")
                if result["status"] != "completed":
                    raise RuntimeError(result.get("error") or "设备没有返回有效完成结果")
                return result
            raise TimeoutError("等待设备的结构化结果超时，未发送下一步")
        finally:
            with self.result_lock:
                self.expected = None
                while not self.results.empty():
                    self.results.get_nowait()

    def change_state(self, state):
        self.state_changed.emit(state)
        if self.cancelled.is_set():
            raise InterruptedError("扫描已取消")

    def publish_sample(self, record, frame):
        """只发布已提交点；缩略图缓存失败不改变原图和索引的事实。"""
        scale = min(160 / frame.shape[1], 120 / frame.shape[0], 1)
        thumbnail = cv.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))))
        try:
            write_image(self.run_dir / "thumbs" / f"s{record['sample_id']:06d}.jpg", thumbnail)
        except (OSError, ValueError, cv.error):
            pass
        self.sample_ready.emit({"run_id": self.owner, "record": record, "thumbnail": thumbnail,
                                "image_path": str(self.run_dir / record["image_path"])})
        self.emitted_ids.add(record["sample_id"])

    def run(self):
        status, error, cleanup_error, committed = "failed", None, None, 0
        run = None
        try:
            self.change_state("PREPARING")
            config = copy.deepcopy(self.config)
            position = config["position"]
            plan = make_scan_plan(config["a_steps"], config["b_steps"], config["interval_steps"],
                                  config["width"], config["height"], config.get("roi"))
            settle_ms = config.get("settle_ms", 200)
            if type(settle_ms) is not int or not 0 <= settle_ms <= 60000:
                raise ValueError("停稳等待须为 0–60000 ms")
            output = Path(config["output_dir"]).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output).free < plan["disk_estimate_bytes"]:
                raise OSError("磁盘剩余空间小于保守数据预算，未开始扫描")
            context = (str(output), config.get("sample_name", ""), position["connection_id"],
                       position["z_reference_id"], position["view_context_id"])
            if self.session_context != context or self.session_dir is None:
                self.session_dir = create_session(output, config.get("sample_name", ""))
                self.session_context = context
            metadata = {key: value for key, value in config.items() if key not in ("position", "output_dir", "freshness_profile")}
            metadata.update({key: value for key, value in position.items()
                             if key not in ("marks", "motion_busy", "state_monotonic_ns", "control_owner")})
            metadata.update(plan, focus_algorithm="Tenengrad", algorithm_version="sobel3_mean_v1",
                focus_parameters={"ksize": 3, "scale": 1, "border": "BORDER_REFLECT_101", "threshold": None},
                candidate_count=3, representative_rule="median_score_then_capture_order",
                quality_policy="not_assessed", approach_steps=0, error=None,
                exposure_requested_us=config["exposure_us"], exposure_readback_us=None,
                gain_readback_percent=None, ae_enabled=None, xy_position_mm=position.get("xy_position_mm"),
                xy_read_time=position.get("xy_read_time"), xy_source=position.get("xy_source"))
            self.run_dir = create_run(self.session_dir, metadata)
            run = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
            self.owner = run["run_id"]
            if not self.control.acquire_control(self.owner):
                raise RuntimeError("运动设备忙，无法取得扫描控制权")
            self.control_owned = True
            snapshot = self.request("control", self.control.request_position, 2)["position"]
            if (not snapshot["z_position_valid"] or not snapshot["z_reference_confirmed"]
                    or any(snapshot[key] != position[key] for key in ("connection_id", "z_reference_id", "view_context_id",
                        "pulses_per_rev", "lead_mm", "gear_ratio", "full_steps_per_rev"))):
                raise ValueError("Z 参考或视野已改变，请重新确认扫描范围")
            allowed = snapshot["allowed_z_range"]
            if allowed is None or any(not allowed[0] <= value <= allowed[1]
                                      for value in (snapshot["z_command_steps"], plan["a_steps"], plan["b_steps"])):
                raise ValueError("当前位置、A 或 B 超出已手动确认的允许范围")
            if abs(snapshot["z_command_steps"] - plan["a_steps"]) > MAX_Z_MOVE_STEPS:
                raise ValueError("返回起点超过单次 10000 脉冲上限")
            if self.camera.recording or self.camera.auto_capture:
                raise ValueError("请先结束录像或自动截图，再开始 Z 扫描")
            # 只重试取得空闲控制权，不重发设置或运动。
            acquire_deadline = time.monotonic() + 0.7
            while not self.camera.acquire_capture(self.owner):
                if self.cancelled.wait(0.02):
                    raise InterruptedError("扫描已取消")
                if time.monotonic() >= acquire_deadline:
                    raise RuntimeError("相机仍有未完成的手动任务，请稍后再试")
            self.camera_owned = True
            camera_config = {"exposure_us": config["exposure_us"], "gain_percent": config["gain_percent"],
                             "capture_policy": "continuous_auto", "formal": True}
            settings = self.request("camera", lambda req: self.camera.request_camera_settings(camera_config, req, self.owner), 5)
            diagnostic = settings["diagnostic"]
            if (diagnostic["width"], diagnostic["height"]) != (config["width"], config["height"]):
                raise ValueError("相机尺寸已改变，请重新生成扫描计划")
            readback = settings["readback"]
            capture_info = settings.get("capture_info") or {}
            if (capture_info.get("policy") != "continuous_auto"
                    or capture_info.get("freshness_method") != "continuous_unverified"
                    or capture_info.get("freshness_verified") is not False
                    or type(capture_info.get("fresh_guard_ms")) is not int
                    or not 200 <= capture_info["fresh_guard_ms"] <= 60000
                    or type(capture_info.get("timeout_seconds")) not in (int, float)
                    or not math.isfinite(capture_info["timeout_seconds"])
                    or not 0 < capture_info["timeout_seconds"] <= 600):
                raise ValueError("相机未确认自动取帧策略及等待期限，未发送运动")
            run.update(camera_id=diagnostic["camera_id"], camera_model=diagnostic.get("camera_model"),
                position_source="acknowledged_command_steps",
                sdk_version=diagnostic["sdk_version"], camera_stream_id=diagnostic["camera_stream_id"],
                pixel_format=diagnostic["pixel_format"], camera_diagnostic=diagnostic,
                exposure_readback_us=readback["real_exposure_us"], gain_readback_percent=readback["gain_percent"],
                ae_enabled=readback["ae_enabled"], camera_settings=readback,
                capture_policy="continuous_auto", freshness_method="continuous_unverified", freshness_verified=False,
                fresh_guard_ms=capture_info["fresh_guard_ms"], fresh_guard_source="exposure_estimate",
                fresh_guard_rule="max(200 ms, 2 * actual exposure ms)",
                allowed_min_steps=allowed[0], allowed_max_steps=allowed[1],
                started_at=datetime.now(timezone.utc).isoformat(), status="running")
            write_json(self.run_dir / "run.json", run)
            self.prepared.emit({"run": copy.deepcopy(run), "run_dir": str(self.run_dir)})
            current = snapshot["z_command_steps"]
            capture_timeout = capture_info["timeout_seconds"] + 2
            last_host_frame_id = -1
            for sample_id, target in enumerate(plan["targets"]):
                self.change_state("MOVING")
                distance = abs(target - current)
                motion_timeout = 4 + distance / 1000 + math.ceil(distance / Z_CHUNK_STEPS)
                moved = self.request("control", lambda req: self.control.request_z_move(target, req, self.owner), motion_timeout)
                point_position = moved["position"]
                if (not point_position["z_position_valid"] or point_position["z_command_steps"] != target
                        or point_position["z_reference_id"] != position["z_reference_id"]):
                    raise RuntimeError("运动完成结果的位置或参考不匹配")
                current = target
                self.change_state("SETTLING")
                if self.cancelled.wait(settle_ms / 1000):
                    raise InterruptedError("停稳等待期间取消")
                settled = time.monotonic_ns()
                self.change_state("CAPTURING")
                captured = self.request("camera", lambda req: self.camera.request_capture(req, self.owner, sample_id, self.owner), capture_timeout)
                packets = captured.get("frames", [])
                if (captured.get("sample_id") != sample_id or len(packets) != 3
                        or captured.get("freshness_method") != run["freshness_method"]
                        or captured.get("freshness_verified") is not False
                        or captured.get("settings") != readback):
                    raise RuntimeError("采集结果的点身份或三帧状态不匹配")
                self.change_state("PROCESSING")
                scores, candidates, identities = [], [], set()
                sequences, timestamps = set(), set()
                for packet in packets:
                    info, frame = packet["metadata"], packet["frame"]
                    identity = (info["camera_stream_id"], info["host_frame_id"])
                    seq, timestamp = info.get("camera_seq"), info.get("camera_timestamp_us")
                    if (identity in identities or info["camera_stream_id"] != run["camera_stream_id"]
                            or info.get("run_id") != self.owner or info.get("sample_id") != sample_id
                            or info.get("request_id") != captured["request_id"]
                            or type(info["host_frame_id"]) is not int or info["host_frame_id"] <= last_host_frame_id
                            or info.get("host_received_monotonic_ns", 0) < settled
                            or info.get("freshness_method") != run["freshness_method"]
                            or info.get("freshness_verified") is not False
                            or info.get("fresh_guard_ms") != run["fresh_guard_ms"]
                            or (seq is None and timestamp is None)
                            or (seq is not None and seq in sequences) or (timestamp is not None and timestamp in timestamps)
                            or frame.shape != (config["height"], config["width"], 3)):
                        raise RuntimeError("候选帧的身份、流或尺寸不匹配")
                    identities.add(identity)
                    last_host_frame_id = info["host_frame_id"]
                    if seq is not None:
                        sequences.add(seq)
                    if timestamp is not None:
                        timestamps.add(timestamp)
                    score = tenengrad(frame, config.get("roi"))
                    scores.append(score)
                    candidates.append({**info, "score": score})
                index = sorted(range(3), key=lambda i: (scores[i], i))[1]
                record = {"run_id": self.owner, "sample_id": sample_id, "request_id": captured["request_id"],
                    "z_reference_id": position["z_reference_id"], "target_steps": target,
                    "acknowledged_steps": current, "position_source": "acknowledged_command_steps",
                    "estimated_z_um": target * position["lead_mm"] * 1000 / position["pulses_per_rev"] / position["gear_ratio"],
                    "motion_finished_monotonic_ns": moved["finished_monotonic_ns"],
                    "settled_monotonic_ns": settled, "roi": config.get("roi"),
                    "representative_index": index, "focus_score": scores[index], "candidates": candidates,
                    "candidate_images_saved": False, "score_min": min(scores), "score_max": max(scores),
                    "score_median": scores[index], "score_span": max(scores) - min(scores),
                    "quality_status": "not_assessed", "freshness_method": captured["freshness_method"],
                    "freshness_verified": False, "fresh_guard_ms": run["fresh_guard_ms"],
                    "fresh_guard_source": "exposure_estimate"}
                self.change_state("SAVING")
                record = commit_sample(self.run_dir, packets[index]["frame"], record)
                committed += 1
                run["committed_count"] = committed
                self.publish_sample(record, packets[index]["frame"])
                del packets, captured, packet, frame
                if self.cancelled.is_set():
                    raise InterruptedError("本点已保存，后续扫描已取消")
            status = "completed"
        except InterruptedError as exception:
            status, error = "cancelled", str(exception)
        except Exception as exception:
            status, error = "failed", str(exception)
        finally:
            self.cancel_task()
            # 未决请求收尾前不能释放控制权，也不能让窗口销毁仍在运行的设备线程。
            deadline = time.monotonic() + max(5, self.pending_timeout + 2)
            while ((self.control_owned and self.control.busy)
                   or (self.camera_owned and self.camera.camera_busy)) and time.monotonic() < deadline:
                time.sleep(0.02)
            if run is not None:
                # JSONL 可能已提交而 run.json 更新失败；最终计数只认完整行。
                try:
                    recovered, records, issues = read_run(self.run_dir, verify_images=False)
                    committed = recovered["committed_count"]
                    if issues:
                        status, error = "failed", error or "；".join(issues)
                    # 索引已落盘但 Run 计数更新失败时，仍展示该已提交点，然后报告失败。
                    for record in records:
                        if record["sample_id"] not in self.emitted_ids:
                            frame = cv.imdecode(np.fromfile(self.run_dir / record["image_path"], dtype=np.uint8), cv.IMREAD_COLOR)
                            if frame is None:
                                raise OSError("已提交点的原图无法恢复")
                            self.publish_sample(record, frame)
                    run.update(status=status, committed_count=committed, error=error,
                               cleanup_error=cleanup_error, ended_at=datetime.now(timezone.utc).isoformat())
                    write_json(self.run_dir / "run.json", run)
                except Exception as exception:
                    status = "failed"
                    error = (error or "") + f"；保存收尾信息失败：{exception}"
            # 完成最后的文件收尾后才解锁，保存阶段也不能插入其他任务。
            cleanup_errors = []
            if self.camera_owned:
                try:
                    if not self.camera.running and not self.camera.camera_busy:
                        self.camera_owned = False
                    else:
                        self.request("camera", lambda req: self.camera.release_capture(self.owner, req), 3, cleanup=True)
                        self.camera_owned = False
                except Exception as exception:
                    cleanup_errors.append(str(exception))
            if self.control_owned:
                try:
                    if not self.control.release_control(self.owner):
                        raise RuntimeError("运动尚未结束，控制权保持锁定")
                    self.control_owned = False
                except Exception as exception:
                    cleanup_errors.append(str(exception))
            if cleanup_errors:
                cleanup_error = "；".join(cleanup_errors)
                status = "failed"
                if run is not None:
                    try:
                        run.update(status=status, error=error, cleanup_error=cleanup_error)
                        write_json(self.run_dir / "run.json", run)
                    except Exception as exception:
                        error = (error or "") + f"；收尾状态保存失败：{exception}"
            self.state_changed.emit(status.upper())
            self.task_result.emit({"run_id": self.owner, "status": status, "error": error,
                "cleanup_error": cleanup_error, "committed_count": committed,
                "run_dir": str(self.run_dir) if self.run_dir else None})
