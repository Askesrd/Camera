# ScanStorage.py
# 普通文件函数；由相机/扫描 worker 调用，不打开设备，不操作 GUI。
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2 as cv
import numpy as np


def write_json(path, data):
    """同目录临时文件替换，避免写到一半破坏原来的配置。"""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_session(output_dir, sample_name=""):
    now = datetime.now(timezone(timedelta(hours=8)))
    session_id = f"session_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    path = Path(output_dir) / session_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "session.json", {"schema_version": 1, "session_id": session_id,
               "created_at": now.isoformat(), "sample_name": sample_name, "runs": []})
    return path


def create_run(session_dir, config):
    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if session.get("schema_version") != 1:
        raise ValueError("不支持的 Session 格式版本")
    # JSON 往返复制，后续修改输入框不会改掉已经保存的配置。
    config = json.loads(json.dumps(config, allow_nan=False))
    run_id = f"run_{len(session['runs']) + 1:03d}_{uuid.uuid4().hex[:12]}"
    path = session_dir / run_id
    path.mkdir(exist_ok=False)
    (path / "images").mkdir()
    (path / "thumbs").mkdir()
    (path / "samples.jsonl").touch(exist_ok=False)
    run = {**config, "schema_version": 1, "session_id": session["session_id"],
           "run_id": run_id, "status": "preparing", "committed_count": 0,
           "image_format": "PNG", "png_compression": 1,
           "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    write_json(path / "run.json", run)
    session["runs"].append({"run_id": run_id, "directory": run_id})
    write_json(session_dir / "session.json", session)
    return path


def write_image(path, frame):
    """中文路径使用 imencode；Windows rename 不覆盖已有的最终文件。"""
    path = Path(path)
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size):
        raise ValueError("图像必须是非空的 uint8 BGR 三通道数组")
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError("只支持 PNG/JPEG 图像")
    parameters = [cv.IMWRITE_PNG_COMPRESSION, 1] if suffix == ".png" else []
    ok, encoded = cv.imencode(suffix, frame, parameters)
    if not ok:
        raise OSError("图像编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"图像已经存在，拒绝覆盖：{path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_run(run_dir, verify_images=True):
    """只读恢复：JSONL 是采样事实来源，不恢复任何运动权限。"""
    path = Path(run_dir)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if run.get("schema_version") != 1:
        raise ValueError("不支持的 Run 格式版本")
    samples, issues, seen = [], [], set()
    with (path / "samples.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                issues.append(f"第 {number} 行未写完，已停止读取")
                break
            try:
                record = json.loads(line)
                sample_id = record["sample_id"]
                relative = Path(record["image_path"])
                if (record.get("run_id") != run["run_id"] or type(sample_id) is not int
                        or sample_id < 0 or sample_id in seen or relative.is_absolute()
                        or relative.drive or relative.root or ".." in relative.parts
                        or record.get("capture_status") != "committed"):
                    raise ValueError("重复点、错误身份或非法图像路径")
                seen.add(sample_id)
                if not (path / relative).is_file():
                    issues.append(f"点 {sample_id} 的图像缺失")
                elif verify_images:
                    encoded = np.fromfile(path / relative, dtype=np.uint8)
                    pixels = cv.imdecode(encoded, cv.IMREAD_COLOR) if encoded.size else None
                    if pixels is None:
                        issues.append(f"点 {sample_id} 的图像损坏")
                samples.append(record)
            except (ValueError, KeyError, TypeError, OSError, cv.error) as error:
                issues.append(f"第 {number} 行记录异常：{error}")
                break
    run["committed_count"] = len(samples)
    if run.get("status") in ("preparing", "running"):
        run["status"] = "interrupted"
    return run, samples, issues


def commit_sample(run_dir, frame, record):
    """先落图、再同步完整 JSONL 行，最后更新可重建的 Run 计数。"""
    path = Path(run_dir)
    current = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if current.get("status") not in ("preparing", "running"):
        raise ValueError("Run 已结束，不能再追加采样点")
    run, samples, issues = read_run(path, verify_images=False)
    if issues:
        raise ValueError("Run 需要先处理不完整记录：" + "; ".join(issues))
    record = json.loads(json.dumps(record, allow_nan=False))
    sample_id, target = record.get("sample_id"), record.get("target_steps")
    score = record.get("focus_score")
    if (type(sample_id) is not int or sample_id < 0 or type(target) is not int
            or any(item["sample_id"] == sample_id for item in samples)
            or type(score) not in (int, float) or not math.isfinite(score)):
        raise ValueError("采样点编号、位置或分数无效")
    if record.get("run_id", run["run_id"]) != run["run_id"]:
        raise ValueError("采样点不属于此 Run")
    if record.get("freshness_method") not in ("continuous_validated", "software_trigger_verified"):
        raise ValueError("未验证的新帧只能作为调试数据，不能提交为正式采样点")
    if record.get("acknowledged_steps", target) != target:
        raise ValueError("已确认累计与目标不一致，不能提交采样点")
    if not isinstance(frame, np.ndarray) or shutil.disk_usage(path).free < frame.nbytes * 1.1 + 1048576:
        raise OSError("剩余空间不足以写入原图及记录")
    relative = f"images/s{sample_id:06d}_z{target:+08d}.png"
    record.update(run_id=run["run_id"], image_path=relative, capture_status="committed")
    write_image(path / relative, frame)
    with (path / "samples.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    run["status"] = "running"
    run["committed_count"] = len(samples) + 1
    write_json(path / "run.json", run)
    return record
