# ScanStorage.py
# 普通文件函数；由相机/扫描 worker 调用，不打开设备，不操作 GUI。
import csv
import copy
import json
import math
import os
import re
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


def image_path(run_dir, relative):
    """历史记录只允许引用本 Run 内的文件，包括解析后的链接目标。"""
    if not isinstance(relative, str) or not relative:
        raise ValueError("图像相对路径为空")
    root, relative = Path(run_dir).resolve(), Path(relative)
    if relative.is_absolute() or relative.drive or relative.root or ".." in relative.parts:
        raise ValueError("非法图像路径")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("图像路径超出了 Run 目录")
    return path


def read_image(path, width=None, height=None):
    """后台按需读取原图；不向历史目录写入或修补文件。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    frame = cv.imdecode(encoded, cv.IMREAD_COLOR) if encoded.size else None
    if frame is None:
        raise ValueError("图像损坏或无法解码")
    if width and height and frame.shape[:2] != (height, width):
        raise ValueError("图像尺寸与 Run 记录不符")
    return frame


def read_run(run_dir, verify_images=True, image_details=False, cancelled=None):
    """只读恢复：JSONL 是采样事实来源，不恢复任何运动权限。"""
    path = Path(run_dir)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if (not isinstance(run, dict) or run.get("schema_version") != 1
            or not isinstance(run.get("run_id"), str) or not run["run_id"]):
        raise ValueError("不支持的 Run 格式版本")
    samples, issues, seen = [], [], set()
    with (path / "samples.jsonl").open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if cancelled and cancelled():
                raise InterruptedError("已取消读取历史数据")
            if not line.endswith(b"\n"):
                issues.append(f"第 {number} 行未写完，已停止读取")
                break
            try:
                record = json.loads(line)
                sample_id = record["sample_id"]
                relative = record["image_path"]
                original = image_path(path, relative)
                score, target = record.get("focus_score"), record.get("target_steps")
                if (record.get("run_id") != run["run_id"] or type(sample_id) is not int
                        or sample_id < 0 or sample_id in seen or type(target) is not int
                        or type(score) not in (int, float) or not math.isfinite(score)
                        or record.get("capture_status") != "committed"):
                    raise ValueError("重复点、错误身份或无效的位置/分数")
                candidates = record.get("candidates", [])
                representative = record.get("representative_index")
                estimate = record.get("estimated_z_um")
                if (not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates)
                        or (representative is not None and (type(representative) is not int or not 0 <= representative < len(candidates)))
                        or not isinstance(record.get("quality_status", "not_assessed"), str)
                        or (estimate is not None and (type(estimate) not in (int, float) or not math.isfinite(estimate)))):
                    raise ValueError("候选帧、代表图、质量或理论位置元数据无效")
                seen.add(sample_id)
                image_status, image_error = "ok", ""
                if not original.is_file():
                    image_status, image_error = "missing", "图像缺失"
                    issues.append(f"点 {sample_id} 的图像缺失")
                elif verify_images:
                    try:
                        pixels = read_image(original, run.get("width"), run.get("height"))
                        del pixels
                    except (ValueError, OSError, cv.error) as error:
                        image_status, image_error = "corrupt", str(error)
                        issues.append(f"点 {sample_id} 的图像损坏：{error}")
                if image_details:
                    record.update(image_status=image_status, image_error=image_error)
                samples.append(record)
            except (ValueError, KeyError, TypeError, OSError, cv.error) as error:
                issues.append(f"第 {number} 行记录异常：{error}")
                break
    run["committed_count"] = len(samples)
    if run.get("status") in ("preparing", "running"):
        run["status"] = "interrupted"
    return run, samples, issues


def read_history(directory, cancelled=None, progress=None):
    """读取 Session、单个 Run 或其上一级数据目录；恢复仅供浏览。"""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("历史数据目录不存在")
    if (root / "session.json").is_file() or (root / "run.json").is_file():
        folders = [root]
    else:
        folders = sorted((p for p in root.iterdir() if p.is_dir()
                          and ((p / "session.json").is_file() or (p / "run.json").is_file())), reverse=True)
    if not folders:
        raise ValueError("未找到扫描记录，请选择 Session、Run 或包含它们的数据目录")
    runs, issues, seen, visited = [], [], set(), set()
    for folder in folders:
        if cancelled and cancelled():
            raise InterruptedError("已取消读取历史数据")
        entries = []
        session_id = None
        if (folder / "session.json").is_file():
            try:
                session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
                if (not isinstance(session, dict) or session.get("schema_version") != 1
                        or not isinstance(session.get("runs"), list)
                        or not isinstance(session.get("session_id"), str) or not session["session_id"]):
                    raise ValueError("不支持的 Session 格式")
                session_id = session["session_id"]
                for item in session["runs"]:
                    candidate = image_path(folder, item["directory"])
                    entries.append((candidate, item["run_id"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(f"{folder.name}：Session 索引异常：{error}")
            # 索引更新中断时，已创建的 Run 仍可只读查看，不能续扫。
            for candidate in sorted(folder.iterdir(), reverse=True):
                if candidate.is_dir() and (candidate / "run.json").is_file():
                    try:
                        entries.append((image_path(folder, candidate.name), None))
                    except ValueError as error:
                        issues.append(f"{candidate.name}：{error}")
        else:
            entries = [(folder, None)]
        for path, expected_id in entries:
            if path in visited:
                continue
            visited.add(path)
            if progress:
                progress(f"正在读取并检查原图：{path.name}")
            try:
                run, records, run_issues = read_run(path, image_details=True, cancelled=cancelled)
                if (expected_id is not None and expected_id != run["run_id"]
                        or session_id is not None and session_id != run.get("session_id")):
                    raise ValueError("Run 身份与 Session 索引不符")
                if run["run_id"] in seen:
                    raise ValueError("重复 Run 身份，未合并")
                seen.add(run["run_id"])
                runs.append({"run": run, "run_dir": str(path), "records": records, "issues": run_issues})
            except (OSError, ValueError, KeyError, TypeError, cv.error) as error:
                issues.append(f"{path.name}：{error}")
    return {"runs": runs, "issues": issues, "directory": str(root)}


def export_run(run_dir, run, records, output_dir, interval=None, cancelled=None, progress=None):
    """复制冻结的点集合；失败保留结果目录和明细，绝不覆盖或移动原图。"""
    run, records = copy.deepcopy(run), copy.deepcopy(records)
    if not records or any(r.get("run_id") != run["run_id"] for r in records):
        raise ValueError("没有可导出的点，或所选点不属于当前 Run")
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("导出点身份重复")
    originals = set()
    for record in records:
        original = image_path(run_dir, record["image_path"])
        relative = Path(record["image_path"])
        if relative.parts[0] != "images" or relative.suffix.lower() != ".png" or original in originals:
            raise ValueError("导出只接受 images 目录下不重复的原始 PNG 路径")
        originals.add(original)
    now = datetime.now(timezone(timedelta(hours=8)))
    path = Path(output_dir) / f"export_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "incomplete", "created_at": now.isoformat(),
              "source_run_id": run["run_id"], "source_session_id": run.get("session_id"),
              "source_run_dir": str(Path(run_dir).resolve()), "interval": interval,
              "selected_sample_ids": ids, "copied_sample_ids": [], "failures": []}
    exported = {**run, "status": "interrupted", "committed_count": 0,
                "export_source_status": run.get("status"), "export_selected_count": len(records)}
    write_json(path / "export.json", report)
    write_json(path / "source_run.json", run)
    write_json(path / "run.json", exported)
    fields = ["run_id", "sample_id", "target_steps", "acknowledged_steps", "estimated_z_um", "focus_score",
              "z_reference_id", "position_source", "representative_host_frame_id", "camera_stream_id",
              "camera_seq", "camera_timestamp_us", "captured_at_utc", "exposure_us", "gain_percent", "roi", "quality_status",
              "freshness_method", "freshness_verified", "fresh_guard_ms", "image_path",
              "candidate_score_1", "candidate_score_2", "candidate_score_3", "export_status", "export_error"]
    processed_count = 0
    try:
        with (path / "samples.jsonl").open("x", encoding="utf-8", newline="\n") as index:
            for record in records:
                if cancelled and cancelled():
                    raise InterruptedError("已取消导出")
                sample_id = record["sample_id"]
                target = None
                target_created = False
                item_status, item_error = "copied", ""
                try:
                    source = image_path(run_dir, record["image_path"])
                    frame = read_image(source, run.get("width"), run.get("height"))
                    del frame
                    target = image_path(path, record["image_path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open("rb") as incoming, target.open("xb") as outgoing:
                        target_created = True
                        while True:
                            if cancelled and cancelled():
                                raise InterruptedError("已取消导出")
                            block = incoming.read(1048576)
                            if not block:
                                break
                            outgoing.write(block)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                except InterruptedError:
                    if target_created:
                        target.unlink(missing_ok=True)
                    raise
                except (OSError, ValueError, KeyError, cv.error) as error:
                    if target_created:
                        target.unlink(missing_ok=True)
                    item_status, item_error = "failed", str(error)
                    report["failures"].append({"sample_id": sample_id, "error": item_error})
                if item_status == "copied":
                    # 索引写入失败即结束，不再把下一条拼到可能的半行后面。
                    saved = {key: value for key, value in record.items() if key not in ("image_status", "image_error")}
                    index.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                    index.flush()
                    os.fsync(index.fileno())
                    report["copied_sample_ids"].append(sample_id)
                processed_count += 1
                if progress:
                    progress(f"正在导出 {processed_count} / {len(records)} 点")
        report["status"] = "failed" if report["failures"] else "completed"
    except Exception as error:
        report["status"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
        report["error"] = str(error)
    finally:
        # 即使最后一次同步失败，也按完整 JSONL 行恢复事实，不能继续追加。
        try:
            _, committed, index_issues = read_run(path, verify_images=False)
            report["copied_sample_ids"] = [r["sample_id"] for r in committed]
            if index_issues:
                report["index_issues"] = index_issues
                if report["status"] == "completed":
                    report["status"] = "failed"
        except (OSError, ValueError) as error:
            report.update(status="failed", error=str(error))
        processed = set(report["copied_sample_ids"]) | {item["sample_id"] for item in report["failures"]}
        report["unprocessed_sample_ids"] = [sample_id for sample_id in ids if sample_id not in processed]
        failures = {item["sample_id"]: item["error"] for item in report["failures"]}
        rows = []
        for record in records:
            sample_id = record["sample_id"]
            candidates = record.get("candidates") or []
            selected = record.get("representative_index")
            representative = candidates[selected] if type(selected) is int and 0 <= selected < len(candidates) else {}
            row = {field: record.get(field) for field in fields}
            row.update(representative_host_frame_id=representative.get("host_frame_id"),
                       camera_stream_id=representative.get("camera_stream_id"),
                       camera_seq=representative.get("camera_seq"), camera_timestamp_us=representative.get("camera_timestamp_us"),
                       captured_at_utc=representative.get("host_received_utc"),
                       exposure_us=representative.get("exposure_us", run.get("exposure_readback_us")),
                       gain_percent=representative.get("gain_percent", run.get("gain_readback_percent")),
                       roi=json.dumps(record.get("roi", run.get("roi")), ensure_ascii=False),
                       export_status="copied" if sample_id in report["copied_sample_ids"] else ("failed" if sample_id in failures else "unprocessed"),
                       export_error=failures.get(sample_id, "" if sample_id in report["copied_sample_ids"] else report.get("error", "未处理")))
            for i in range(3):
                row[f"candidate_score_{i + 1}"] = candidates[i].get("score") if i < len(candidates) else None
            rows.append(row)
        try:
            with (path / "samples.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            exported.update(status="completed" if report["status"] == "completed" else "interrupted",
                            committed_count=len(report["copied_sample_ids"]))
            write_json(path / "run.json", exported)
            write_json(path / "export.json", report)
        except (OSError, ValueError) as error:
            raise OSError(f"导出收尾失败，已保留目录 {path}：{error}") from error
    return {**report, "export_dir": str(path)}


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
    method = record.get("freshness_method")
    if method == "continuous_unverified":
        if (run.get("capture_policy") != "continuous_auto" or run.get("freshness_method") != method
                or run.get("freshness_verified") is not False or record.get("freshness_verified") is not False
                or type(run.get("fresh_guard_ms")) is not int or not 200 <= run["fresh_guard_ms"] <= 60000
                or run.get("fresh_guard_source") != "exposure_estimate"
                or record.get("fresh_guard_source") != "exposure_estimate"
                or record.get("fresh_guard_ms") != run.get("fresh_guard_ms")):
            raise ValueError("自动连续采集必须保留一致的未验证状态和估算等待")
    elif method not in ("continuous_validated", "software_trigger_verified"):
        raise ValueError("未验证的新帧只能作为调试数据，不能提交为正式采样点")
    if record.get("acknowledged_steps", target) != target:
        raise ValueError("已确认累计与目标不一致，不能提交采样点")
    if not isinstance(frame, np.ndarray) or shutil.disk_usage(path).free < frame.nbytes * 1.1 + 1048576:
        raise OSError("剩余空间不足以写入原图及记录")
    # 使用本次 Run 固定的名称；原始名称仍完整保存在配置中。
    name = run.get("sample_name", "")
    name = name if isinstance(name, str) else ""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", name).strip(" .")[:40].rstrip(" .")
    # CON.txt 等也是 Windows 保留名，在前面加下划线后才能安全保存。
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]", name.split(".", 1)[0].rstrip().upper()):
        name = "_" + name
    prefix = name + "_" if name else ""
    relative = f"images/{prefix}s{sample_id:06d}_z{target:+08d}.png"
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
