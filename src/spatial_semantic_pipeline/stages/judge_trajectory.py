#!/usr/bin/env python3
"""Judge whether a pose trajectory CSV looks normal.

Prints 1 for a normal trajectory and 0 for an abnormal trajectory.
Use --verbose to print rejection reasons to stderr.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    DEFAULT_CONFIG,
    find_video,
    load_scenes,
    load_videos,
    load_yaml,
    metadata_root,
    pipeline_config,
    pipeline_lock,
    save_scenes,
    save_videos,
    stage_log,
)
from ..core.stages import Status


REQUIRED_COLUMNS = ["frame_idx", "timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
COLUMN_ALIASES = {
    "frame_idx": ("frame_idx", "scene_frame"),
    "timestamp": ("timestamp", "source_timestamp"),
    "tx": ("tx", "x"),
    "ty": ("ty", "y"),
    "tz": ("tz", "z"),
    "qw": ("qw",),
    "qx": ("qx",),
    "qy": ("qy",),
    "qz": ("qz",),
}
DEFAULT_JUDGE_CONFIG_PATH = DEFAULT_CONFIG
DEFAULT_THRESHOLDS = {
    "min_poses": 5,
    "quaternion_norm_tolerance": 0.2,
    "positive_step_epsilon": 1e-12,
    "iqr_epsilon": 1e-12,
    "jump_median_step_multiplier": 5.0,
    "jitter_median_step_multiplier": 4.0,
    "jitter_iqr_multiplier": 3.0,
    "jitter_ratio_limit": 0.07,
    "reject_no_movement": 1,
    "reject_non_positive_duration": 1,
    "static_step_median_multiplier": 0.05,
    "static_time_ratio_limit": 1.0 / 3.0,
    "min_turn_checks": 5,
    "sharp_turn_cosine_limit": -0.5,
    "reversal_cosine_limit": -0.85,
    "sharp_turn_ratio_limit": 0.16,
    "reversal_ratio_limit": 0.14,
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print 1 if a poses.csv trajectory is normal, else 0.")
    parser.add_argument("csv_path", type=Path, nargs="?", help="Pose CSV with frame_idx,timestamp,tx,ty,tz,qw,qx,qy,qz.")
    parser.add_argument("--video-id", help="Judge all pose chunks for one registered video.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_JUDGE_CONFIG_PATH,
        help="YAML file with judge thresholds. Defaults to configs/config.yaml.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print abnormal reasons to stderr.")
    return parser.parse_args()


def load_threshold_config(path: Path) -> dict[str, float]:
    config = dict(DEFAULT_THRESHOLDS)
    if not path.exists():
        return config

    data = load_yaml(path)
    values = data.get("judge") if isinstance(data.get("judge"), dict) else data
    for key, value in values.items():
        if key not in config:
            continue
        try:
            config[key] = int(value) if key == "min_poses" else float(value)
        except ValueError as exc:
            raise ValueError(f"invalid numeric value for judge.{key}: {value}") from exc
    return config


def load_video_judge_results(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and item.get("bad_chunks")]
    if isinstance(data, dict) and isinstance(data.get("videos"), dict):
        records = []
        for video_id, item in data["videos"].items():
            bad_chunks = item.get("bad_chunks") if isinstance(item, dict) else None
            if bad_chunks:
                records.append({"video_id": str(video_id), "bad_chunks": bad_chunks})
        return records
    return []


def write_video_judge_results(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("[]\n", encoding="utf-8")
        return
    lines = ["["]
    for index, record in enumerate(records):
        suffix = "," if index < len(records) - 1 else ""
        lines.append("  " + json.dumps(record, ensure_ascii=False, separators=(",", ":")) + suffix)
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_video_judge_results(path: Path, video_id: str, bad_chunks: list[dict]) -> None:
    records = [
        record for record in load_video_judge_results(path)
        if str(record.get("video_id", "")) != video_id
    ]
    if bad_chunks:
        records.append({"video_id": video_id, "bad_chunks": bad_chunks})
    else:
        records = [record for record in records if record.get("bad_chunks")]
    records = sorted(records, key=lambda item: str(item.get("video_id", "")))
    for index, record in enumerate(records, start=1):
        record["index"] = index
        record["bad_chunks"] = sorted(
            record["bad_chunks"],
            key=lambda item: (int(item.get("scene_index", 0)), int(item.get("chunk_index", 0))),
        )
    write_video_judge_results(path, records)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def scene_frame_window(scene: dict, chunk: dict) -> tuple[int, int] | None:
    if chunk.get("logical_start_frame") is None or chunk.get("logical_end_frame") is None:
        return None
    scene_start_frame = scene.get("vggt_start_frame")
    if scene_start_frame is None:
        scene_start_frame = scene.get("source_start_timestamp", scene.get("source_start_sec", 0))
    return int(chunk["logical_start_frame"]) - int(scene_start_frame), int(chunk["logical_end_frame"]) - int(scene_start_frame)


def read_pose_csv(
    path: Path,
    frame_window: tuple[int, int] | None = None,
) -> tuple[list[float], list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        column_for = {
            column: next((alias for alias in COLUMN_ALIASES[column] if alias in fieldnames), None)
            for column in REQUIRED_COLUMNS
        }
        missing = [column for column, actual in column_for.items() if actual is None]
        if missing:
            raise ValueError(f"missing columns: {missing}")

        timestamps: list[float] = []
        positions: list[tuple[float, float, float]] = []
        quaternions: list[tuple[float, float, float, float]] = []
        frame_start, frame_end = frame_window if frame_window is not None else (None, None)
        for row_number, row in enumerate(reader, start=2):
            try:
                scene_frame = int(float(row[column_for["frame_idx"]]))
                timestamp = float(row[column_for["timestamp"]])
                position = (float(row[column_for["tx"]]), float(row[column_for["ty"]]), float(row[column_for["tz"]]))
                quaternion = (
                    float(row[column_for["qw"]]),
                    float(row[column_for["qx"]]),
                    float(row[column_for["qy"]]),
                    float(row[column_for["qz"]]),
                )
            except ValueError as exc:
                raise ValueError(f"non-numeric value at row {row_number}") from exc

            values = [timestamp, *position, *quaternion]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"NaN/Inf value at row {row_number}")
            if frame_start is not None and not (frame_start <= scene_frame < frame_end):
                continue

            timestamps.append(timestamp)
            positions.append(position)
            quaternions.append(quaternion)
    return timestamps, positions, quaternions


def judge(
    path: Path,
    config: dict[str, float] | None = None,
    frame_window: tuple[int, int] | None = None,
) -> tuple[bool, list[str]]:
    config = dict(DEFAULT_THRESHOLDS if config is None else config)
    reasons: list[str] = []
    try:
        timestamps, positions, quaternions = read_pose_csv(path, frame_window)
    except Exception as exc:
        return False, [str(exc)]

    min_poses = int(config["min_poses"])
    if len(positions) < min_poses:
        reasons.append(f"too few poses: {len(positions)} < {min_poses}")
        return False, reasons

    dts = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if any(dt <= 0.0 or not math.isfinite(dt) for dt in dts):
        reasons.append("timestamps are not strictly increasing")

    quat_norms = [math.sqrt(sum(component * component for component in quat)) for quat in quaternions]
    bad_quats = [value for value in quat_norms if abs(value - 1.0) > config["quaternion_norm_tolerance"]]
    if bad_quats:
        reasons.append(f"quaternion norm out of range: {len(bad_quats)} rows")

    vectors = [sub3(positions[i + 1], positions[i]) for i in range(len(positions) - 1)]
    steps = [norm3(vector) for vector in vectors]
    positive_steps = [step for step in steps if step > config["positive_step_epsilon"]]
    if not positive_steps:
        if config["reject_no_movement"] > 0:
            reasons.append("trajectory has no movement")
        return not reasons, reasons

    median_step = median(positive_steps)
    q1 = percentile(positive_steps, 0.25)
    q3 = percentile(positive_steps, 0.75)
    iqr = max(q3 - q1, config["iqr_epsilon"])

    severe_jump_threshold = config["jump_median_step_multiplier"] * median_step
    jump_count = sum(step > severe_jump_threshold for step in steps)
    if jump_count:
        reasons.append(f"adjacent displacement jump: {jump_count} step(s)")

    static_step_threshold = max(config["positive_step_epsilon"], config["static_step_median_multiplier"] * median_step)
    total_time = max(timestamps[-1] - timestamps[0], 0.0)
    static_time = sum(dt for step, dt in zip(steps, dts) if step < static_step_threshold and dt > 0.0)
    if total_time <= 0.0:
        if config["reject_non_positive_duration"] > 0:
            reasons.append("non-positive total duration")
    elif static_time / total_time > config["static_time_ratio_limit"]:
        reasons.append(f"static time ratio too high: {static_time / total_time:.3f}")

    accelerations = [
        norm3((vectors[i + 1][0] - vectors[i][0], vectors[i + 1][1] - vectors[i][1], vectors[i + 1][2] - vectors[i][2]))
        for i in range(len(vectors) - 1)
    ]
    jitter_threshold = max(config["jitter_median_step_multiplier"] * median_step, q3 + config["jitter_iqr_multiplier"] * iqr)
    jitter_count = sum(value > jitter_threshold for value in accelerations)
    if accelerations and jitter_count / len(accelerations) > config["jitter_ratio_limit"]:
        reasons.append(f"jitter ratio too high: {jitter_count / len(accelerations):.3f}")

    moving_vectors = [vector for vector, step in zip(vectors, steps) if step >= static_step_threshold]
    sharp_turns = 0
    reversals = 0
    turn_checks = 0
    for before, after in zip(moving_vectors, moving_vectors[1:]):
        before_norm = norm3(before)
        after_norm = norm3(after)
        if before_norm <= 0.0 or after_norm <= 0.0:
            continue
        cosine = max(-1.0, min(1.0, dot3(before, after) / (before_norm * after_norm)))
        turn_checks += 1
        if cosine < config["sharp_turn_cosine_limit"]:
            sharp_turns += 1
        if cosine < config["reversal_cosine_limit"]:
            reversals += 1

    if turn_checks >= int(config["min_turn_checks"]):
        sharp_ratio = sharp_turns / turn_checks
        reversal_ratio = reversals / turn_checks
        if sharp_ratio > config["sharp_turn_ratio_limit"]:
            reasons.append(f"sharp displacement turn ratio too high: {sharp_ratio:.3f}")
        if reversal_ratio > config["reversal_ratio_limit"]:
            reasons.append(f"displacement reversal ratio too high: {reversal_ratio:.3f}")

    return not reasons, reasons


def main() -> int:
    args = parse_args()
    try:
        config = load_threshold_config(args.config.expanduser())
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.video_id:
        pipeline = pipeline_config()
        judge_results_path = metadata_root(pipeline) / "judge_results.json"
        scenes = load_scenes(pipeline).get("scenes", [])
        judged = 0
        results: dict[tuple[int, int], tuple[str, list[str]]] = {}
        bad_chunks: list[dict] = []
        for scene in scenes:
            if str(scene.get("video_id", "")) != args.video_id:
                continue
            scene_index = int(scene["scene_index"])
            for chunk in scene.get("pose_chunks", []):
                pose_file = chunk.get("pose_file")
                if not pose_file:
                    continue
                normal, reasons = judge(ROOT / str(pose_file), config, scene_frame_window(scene, chunk))
                status = "good" if normal else "bad"
                chunk_index = int(chunk["chunk_index"])
                results[(scene_index, chunk_index)] = (status, reasons)
                judged += 1
                if status == "bad":
                    bad_chunks.append({"scene_index": scene_index, "chunk_index": chunk_index})
                print(
                    f"{status} video_id={args.video_id} scene={scene_index:04d} "
                    f"chunk={chunk_index:04d}",
                    flush=True,
                )

        def update_scenes(scenes_data: dict) -> None:
            for scene in scenes_data.get("scenes", []):
                if str(scene.get("video_id", "")) != args.video_id:
                    continue
                scene_index = int(scene["scene_index"])
                for chunk in scene.get("pose_chunks", []):
                    key = (scene_index, int(chunk["chunk_index"]))
                    if key not in results:
                        continue
                    status, reasons = results[key]
                    chunk["trajectory_judge"] = status
                    chunk["trajectory_judge_reasons"] = reasons
                    chunk["status"] = "trajectory_judged"

        with pipeline_lock(pipeline):
            scenes_data = load_scenes(pipeline)
            update_scenes(scenes_data)
            save_scenes(pipeline, scenes_data)

            videos_data = load_videos(pipeline)
            video = find_video(videos_data, args.video_id)
            video["status"] = Status.TRAJECTORY_JUDGED
            stage_log(video, "judge", "completed", chunks=judged)
            save_videos(pipeline, videos_data)

            update_video_judge_results(judge_results_path, args.video_id, bad_chunks)
        print(
            f"Judged {judged} pose chunk(s) for video_id={args.video_id}; "
            f"bad={len(bad_chunks)} metadata={judge_results_path}"
        )
        return 0

    if args.csv_path is None:
        print("csv_path or --video-id is required", file=sys.stderr)
        return 2
    normal, reasons = judge(args.csv_path.expanduser(), config)
    print("1" if normal else "0")
    if args.verbose and reasons:
        for reason in reasons:
            print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
