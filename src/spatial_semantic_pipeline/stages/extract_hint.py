#!/usr/bin/env python3
"""Label obvious navigation-relevant behavior segments in a pose CSV.

The pose CSV is expected to be one VGGT chunk at 1 fps. Overlap frames used for
chunk stitching are ignored automatically from the chunk file name:

* first chunk: drop the last 5 poses
* middle chunk: drop the first and last 5 poses
* last chunk: drop the first 5 poses

Output is JSON by default and contains time ranges with conservative labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    DEFAULT_CONFIG,
    chunk_is_good_full_minute,
    find_video,
    load_config,
    load_scenes,
    load_videos,
    metadata_root,
    pipeline_lock,
    resolve_path,
    save_scenes,
    save_videos,
    slug,
    stage_log,
)
from ..core.stages import Status


COLUMN_ALIASES = {
    "scene_frame": ("scene_frame", "frame_idx"),
    "source_frame": ("source_frame",),
    "timestamp": ("source_timestamp", "timestamp"),
    "x": ("x", "tx"),
    "y": ("y", "ty"),
    "z": ("z", "tz"),
    "qx": ("qx",),
    "qy": ("qy",),
    "qz": ("qz",),
    "qw": ("qw",),
}

CHUNK_RE = re.compile(r"chunk_(\d+)_poses\.csv$")

HINT_DEFAULTS: dict[str, int | float | str] = {
    "overlap_frames": 5,
    "min_used_poses": 60,
    "min_segment_sec": 3.0,
    "stop_speed_ratio": 0.15,
    "slow_speed_ratio": 0.55,
    "turn_rate_deg": 10.0,
    "min_turn_deg": 50.0,
    "turn_window_sec": 24.0,
    "turn_step_sec": 4.0,
    "turn_core_ratio": 0.30,
    "max_turn_sec": 12.0,
    "min_stop_sec": 4.0,
    "uturn_deg": 120.0,
    "left_sign": "positive",
}


def hint_settings(config: dict[str, Any]) -> dict[str, int | float | str]:
    values = config.get("hint") or {}
    if not isinstance(values, dict):
        raise ValueError("Config field 'hint' must be a mapping")
    settings = dict(HINT_DEFAULTS)
    for key, default in HINT_DEFAULTS.items():
        if key not in values:
            continue
        value = values[key]
        if isinstance(default, int):
            settings[key] = int(value)
        elif isinstance(default, float):
            settings[key] = float(value)
        else:
            settings[key] = str(value)
    if settings["left_sign"] not in {"positive", "negative"}:
        raise ValueError("hint.left_sign must be 'positive' or 'negative'")
    return settings


def apply_hint_settings(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for key, value in hint_settings(config).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label stop/turn/u-turn/speed-change segments from a VGGT pose chunk CSV."
    )
    parser.add_argument("csv_path", type=Path, nargs="?", help="Input chunk_XXXX_poses.csv file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline YAML configuration file.")
    parser.add_argument("--video-id", help="Extract hints for all good pose chunks of one registered video.")
    parser.add_argument("-o", "--output", type=Path, help="Write output to this path instead of stdout.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Output diagnostic fields as well as local second labels.",
    )
    parser.add_argument(
        "--chunk-position",
        choices=("auto", "single", "start", "middle", "end"),
        default="auto",
        help="How to trim overlap poses. Defaults to auto from chunk name and sibling files.",
    )
    parser.add_argument("--overlap-frames", type=int, help="Number of overlap poses to ignore.")
    parser.add_argument("--key-only", action="store_true", help="Omit forward and uncertain segments from output.")
    parser.add_argument("--min-used-poses", type=int, help="Skip chunks with fewer poses after overlap trimming.")
    parser.add_argument(
        "--judge-results",
        type=Path,
        default=Path(__file__).with_name("judge_results.json"),
        help="Only process pose files marked good in this judge_results.json.",
    )
    parser.add_argument("--allow-unjudged", action="store_true", help="Process files missing from judge_results.json.")
    parser.add_argument("--min-segment-sec", type=float, help="Drop/merge shorter behavior segments.")
    parser.add_argument("--stop-speed-ratio", type=float, help="Stop threshold as median speed ratio.")
    parser.add_argument("--slow-speed-ratio", type=float, help="Slow movement threshold as median speed ratio.")
    parser.add_argument("--turn-rate-deg", type=float, help="Per-second yaw rate threshold for turns.")
    parser.add_argument("--min-turn-deg", type=float, help="Minimum accumulated yaw for a turn segment.")
    parser.add_argument("--turn-window-sec", type=float, help="Wide window used to find accumulated turns.")
    parser.add_argument("--turn-step-sec", type=float, help="Step between wide turn windows.")
    parser.add_argument("--turn-core-ratio", type=float, help="Core share of accumulated path turn.")
    parser.add_argument("--max-turn-sec", type=float, help="Maximum output duration for a turn segment.")
    parser.add_argument("--min-stop-sec", type=float, help="Minimum duration for stop segments.")
    parser.add_argument("--uturn-deg", type=float, help="Accumulated yaw threshold for u-turn.")
    parser.add_argument(
        "--left-sign",
        choices=("positive", "negative"),
        help="Which yaw direction is labeled as left. Flip this if left/right are reversed.",
    )
    return parser.parse_args()


def resolve_column(fieldnames: list[str], canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in fieldnames:
            return alias
    raise ValueError(f"missing column for {canonical}: tried {COLUMN_ALIASES[canonical]}")


def read_poses(path: Path) -> list[dict[str, float | int | None]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        columns = {name: resolve_column(fieldnames, name) for name in COLUMN_ALIASES if name != "source_frame"}
        source_frame_col = next((alias for alias in COLUMN_ALIASES["source_frame"] if alias in fieldnames), None)

        poses: list[dict[str, float | int | None]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                pose = {
                    "scene_frame": int(float(row[columns["scene_frame"]])),
                    "source_frame": int(float(row[source_frame_col])) if source_frame_col else None,
                    "timestamp": float(row[columns["timestamp"]]),
                    "x": float(row[columns["x"]]),
                    "y": float(row[columns["y"]]),
                    "z": float(row[columns["z"]]),
                    "qx": float(row[columns["qx"]]),
                    "qy": float(row[columns["qy"]]),
                    "qz": float(row[columns["qz"]]),
                    "qw": float(row[columns["qw"]]),
                }
            except ValueError as exc:
                raise ValueError(f"non-numeric value at row {row_number}") from exc

            numeric_values = [value for value in pose.values() if isinstance(value, float)]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"NaN/Inf value at row {row_number}")
            poses.append(pose)
    return poses


def infer_chunk_position(path: Path) -> str:
    match = CHUNK_RE.search(path.name)
    if not match:
        return "single"

    chunk_index = int(match.group(1))
    sibling_indices = []
    for sibling in path.parent.glob("chunk_*_poses.csv"):
        sibling_match = CHUNK_RE.search(sibling.name)
        if sibling_match:
            sibling_indices.append(int(sibling_match.group(1)))

    if len(sibling_indices) <= 1:
        return "single"
    if chunk_index == min(sibling_indices):
        return "start"
    if chunk_index == max(sibling_indices):
        return "end"
    return "middle"


def trim_overlap(
    poses: list[dict[str, float | int | None]], position: str, overlap_frames: int
) -> tuple[list[dict[str, float | int | None]], dict[str, int | str]]:
    drop_start = overlap_frames if position in {"middle", "end"} else 0
    drop_end = overlap_frames if position in {"start", "middle"} else 0
    if position == "single":
        drop_start = 0
        drop_end = 0

    end = len(poses) - drop_end if drop_end else len(poses)
    trimmed = poses[drop_start:end]
    if len(trimmed) < 2:
        raise ValueError(
            f"too few poses after trimming: original={len(poses)}, drop_start={drop_start}, drop_end={drop_end}"
        )

    return trimmed, {
        "chunk_position": position,
        "original_poses": len(poses),
        "drop_start_poses": drop_start,
        "drop_end_poses": drop_end,
        "used_poses": len(trimmed),
    }


def quat_to_forward_heading(qx: float, qy: float, qz: float, qw: float) -> float:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # CSV quaternions are camera-to-world rotations. VGGT/OpenCV uses +Z as
    # camera forward, and this pipeline plots walking motion on the x-z plane
    # with -Y as up. Use the same yaw convention as Pipeline_VGGT/scripts/plot.py:
    # yaw = atan2(forward_z, forward_x). With x-right/z-forward, positive yaw
    # change corresponds to a left turn when viewed from above.
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(forward_z, forward_x)


def angle_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def moving_median(values: list[float], radius: int = 1) -> list[float]:
    if not values:
        return []
    smoothed = []
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        smoothed.append(median(values[lo:hi]))
    return smoothed


def build_intervals(poses: list[dict[str, float | int | None]]) -> list[dict[str, Any]]:
    headings = [
        quat_to_forward_heading(
            float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"])
        )
        for pose in poses
    ]

    intervals: list[dict[str, Any]] = []
    for index in range(len(poses) - 1):
        a = poses[index]
        b = poses[index + 1]
        dt = float(b["timestamp"]) - float(a["timestamp"])
        if dt <= 0.0:
            continue
        dx = float(b["x"]) - float(a["x"])
        dy = float(b["y"]) - float(a["y"])
        dz = float(b["z"]) - float(a["z"])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_delta = angle_delta(headings[index + 1], headings[index])
        intervals.append(
            {
                "start_pose": a,
                "end_pose": b,
                "dt": dt,
                "distance": distance,
                "planar_distance": math.hypot(dx, dz),
                "move_heading": math.atan2(dz, dx) if math.hypot(dx, dz) > 1e-12 else None,
                "speed": distance / dt,
                "yaw_delta": yaw_delta,
                "yaw_rate_deg": math.degrees(yaw_delta) / dt,
            }
        )
    return intervals


def raw_interval_labels(intervals: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    speeds = moving_median([interval["speed"] for interval in intervals])
    yaw_rates = moving_median([interval["yaw_rate_deg"] for interval in intervals])
    moving_speeds = [speed for speed in speeds if speed > 1e-9]
    baseline_speed = median(moving_speeds) if moving_speeds else 0.0
    stop_threshold = baseline_speed * args.stop_speed_ratio
    slow_threshold = baseline_speed * args.slow_speed_ratio
    left_multiplier = 1.0 if args.left_sign == "positive" else -1.0

    labels = []
    for speed, yaw_rate in zip(speeds, yaw_rates):
        signed_yaw_rate = yaw_rate * left_multiplier
        if baseline_speed > 0.0 and speed <= stop_threshold:
            labels.append("stop")
        elif signed_yaw_rate >= args.turn_rate_deg:
            labels.append("turn_left")
        elif signed_yaw_rate <= -args.turn_rate_deg:
            labels.append("turn_right")
        elif baseline_speed > 0.0 and speed <= slow_threshold:
            labels.append("slow")
        else:
            labels.append("forward")
    return labels


def merge_raw_segments(intervals: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    if not intervals:
        return []

    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(intervals)):
        if labels[index] != labels[start]:
            segments.append(make_segment(intervals, labels[start], start, index - 1, float(intervals[0]["start_pose"]["timestamp"])))
            start = index
    segments.append(make_segment(intervals, labels[start], start, len(intervals) - 1, float(intervals[0]["start_pose"]["timestamp"])))
    return segments


def make_segment(
    intervals: list[dict[str, Any]], label: str, start: int, end: int, local_zero_time: float
) -> dict[str, Any]:
    selected = intervals[start : end + 1]
    start_pose = selected[0]["start_pose"]
    end_pose = selected[-1]["end_pose"]
    duration = float(end_pose["timestamp"]) - float(start_pose["timestamp"])
    total_distance = sum(interval["distance"] for interval in selected)
    total_yaw = sum(interval["yaw_delta"] for interval in selected)
    mean_speed = total_distance / duration if duration > 0.0 else 0.0
    mean_yaw_rate = math.degrees(total_yaw) / duration if duration > 0.0 else 0.0

    return {
        "label": label,
        "start_sec": round(float(start_pose["timestamp"]) - local_zero_time, 3),
        "end_sec": round(float(end_pose["timestamp"]) - local_zero_time, 3),
        "duration": round(duration, 3),
        "start_scene_frame": start_pose["scene_frame"],
        "end_scene_frame": end_pose["scene_frame"],
        "start_source_frame": start_pose["source_frame"],
        "end_source_frame": end_pose["source_frame"],
        "distance": round(total_distance, 6),
        "mean_speed": round(mean_speed, 6),
        "yaw_change_deg": round(math.degrees(total_yaw), 3),
        "mean_yaw_rate_deg": round(mean_yaw_rate, 3),
    }


def relabel_turns_and_speed_changes(segments: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    relabeled = []
    for segment in segments:
        segment = dict(segment)
        if segment["label"] in {"turn_left", "turn_right"}:
            abs_yaw = abs(float(segment["yaw_change_deg"]))
            if abs_yaw < args.min_turn_deg:
                segment["label"] = "forward" if segment["mean_speed"] > 0.0 else "uncertain"
        relabeled.append(segment)

    for index, segment in enumerate(relabeled):
        if segment["label"] != "slow":
            continue
        prev_speed = relabeled[index - 1]["mean_speed"] if index > 0 else None
        next_speed = relabeled[index + 1]["mean_speed"] if index + 1 < len(relabeled) else None
        if prev_speed is not None and next_speed is not None:
            if segment["mean_speed"] < prev_speed * 0.8 and next_speed <= segment["mean_speed"] * 1.2:
                segment["label"] = "slow_down"
            elif prev_speed <= segment["mean_speed"] * 1.2 and next_speed > segment["mean_speed"] * 1.2:
                segment["label"] = "speed_up"
        if segment["label"] == "slow":
            segment["label"] = "forward"
    return relabeled


def merge_short_segments(segments: list[dict[str, Any]], min_segment_sec: float) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    for segment in segments:
        is_obvious_turn = segment["label"] in {"turn_left", "turn_right"} and abs(segment["yaw_change_deg"]) >= 25.0
        if segment["duration"] < min_segment_sec and not is_obvious_turn:
            segment = dict(segment)
            segment["label"] = "uncertain"
        if merged and merged[-1]["label"] == segment["label"]:
            merged[-1] = combine_segments(merged[-1], segment)
        else:
            merged.append(segment)
    return merged


def combine_segments(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    duration = float(b["end_sec"]) - float(a["start_sec"])
    distance = float(a["distance"]) + float(b["distance"])
    yaw_change = float(a["yaw_change_deg"]) + float(b["yaw_change_deg"])
    combined = dict(a)
    combined.update(
        {
            "end_sec": b["end_sec"],
            "duration": round(duration, 3),
            "end_scene_frame": b["end_scene_frame"],
            "end_source_frame": b["end_source_frame"],
            "distance": round(distance, 6),
            "mean_speed": round(distance / duration, 6) if duration > 0.0 else 0.0,
            "yaw_change_deg": round(yaw_change, 3),
            "mean_yaw_rate_deg": round(yaw_change / duration, 3) if duration > 0.0 else 0.0,
        }
    )
    return combined



def detect_stop_segments(intervals: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not intervals:
        return []
    speeds = moving_median([interval["speed"] for interval in intervals], radius=2)
    moving_speeds = [speed for speed in speeds if speed > 1e-9]
    baseline_speed = median(moving_speeds) if moving_speeds else 0.0
    if baseline_speed <= 0.0:
        return []

    stop_threshold = baseline_speed * args.stop_speed_ratio
    labels = ["stop" if speed <= stop_threshold else "moving" for speed in speeds]
    segments = merge_raw_segments(intervals, labels)
    return [segment for segment in segments if segment["label"] == "stop" and segment["duration"] >= args.min_stop_sec]


def pose_planar_heading(a: dict[str, float | int | None], b: dict[str, float | int | None]) -> tuple[float | None, float]:
    dx = float(b["x"]) - float(a["x"])
    dz = float(b["z"]) - float(a["z"])
    distance = math.hypot(dx, dz)
    if distance <= 1e-12:
        return None, distance
    return math.atan2(dz, dx), distance


def window_path_turn(intervals: list[dict[str, Any]], start: int, end: int) -> tuple[float | None, float]:
    count = end - start + 1
    if count < 6:
        return None, 0.0
    side = max(4, min(count // 3, 10))
    pre_start = intervals[start]["start_pose"]
    pre_end = intervals[min(end, start + side - 1)]["end_pose"]
    post_start = intervals[max(start, end - side + 1)]["start_pose"]
    post_end = intervals[end]["end_pose"]
    pre_heading, pre_dist = pose_planar_heading(pre_start, pre_end)
    post_heading, post_dist = pose_planar_heading(post_start, post_end)
    if pre_heading is None or post_heading is None:
        return None, min(pre_dist, post_dist)
    return math.degrees(angle_delta(post_heading, pre_heading)), min(pre_dist, post_dist)


def local_path_turns(intervals: list[dict[str, Any]]) -> list[float]:
    headings = [interval.get("move_heading") for interval in intervals]
    turns = [0.0]
    for index in range(1, len(headings)):
        if headings[index] is None or headings[index - 1] is None:
            turns.append(0.0)
        else:
            turns.append(math.degrees(angle_delta(float(headings[index]), float(headings[index - 1]))))
    return turns


def shrink_path_turn_window(
    intervals: list[dict[str, Any]],
    local_turns: list[float],
    start: int,
    end: int,
    total_turn: float,
    args: argparse.Namespace,
) -> tuple[int, int]:
    sign = 1.0 if total_turn > 0.0 else -1.0
    energy = [max(0.0, local_turns[index] * sign) for index in range(start, end + 1)]
    total_energy = sum(energy)
    if total_energy <= 1e-9:
        return start, end
    tail_ratio = min(max(args.turn_core_ratio, 0.0), 0.45)
    lo_target = total_energy * tail_ratio
    hi_target = total_energy * (1.0 - tail_ratio)
    acc = 0.0
    core_start = start
    core_end = end
    for offset, value in enumerate(energy):
        acc += value
        if acc >= lo_target:
            core_start = max(start, start + offset - 1)
            break
    acc = 0.0
    for offset, value in enumerate(energy):
        acc += value
        if acc >= hi_target:
            core_end = min(end, start + offset + 1)
            break
    return core_start, max(core_start, core_end)




def cap_turn_window_duration(
    intervals: list[dict[str, Any]],
    local_turns: list[float],
    start: int,
    end: int,
    total_turn: float,
    max_turn_sec: float,
) -> tuple[int, int]:
    if max_turn_sec <= 0.0:
        return start, end
    duration = float(intervals[end]["end_pose"]["timestamp"]) - float(intervals[start]["start_pose"]["timestamp"])
    if duration <= max_turn_sec:
        return start, end

    sign = 1.0 if total_turn > 0.0 else -1.0
    energy = [max(0.0, local_turns[index] * sign) for index in range(start, end + 1)]
    if not energy or max(energy) <= 0.0:
        return start, end
    peak_index = start + max(range(len(energy)), key=lambda offset: energy[offset])
    best_start = start
    best_end = start
    best_energy = -1.0
    for candidate_start in range(start, end + 1):
        candidate_end = candidate_start
        while candidate_end < end and float(intervals[candidate_end]["end_pose"]["timestamp"]) - float(intervals[candidate_start]["start_pose"]["timestamp"]) < max_turn_sec:
            candidate_end += 1
        if not (candidate_start <= peak_index <= candidate_end):
            continue
        score = sum(energy[candidate_start - start : candidate_end - start + 1])
        if score > best_energy:
            best_energy = score
            best_start = candidate_start
            best_end = candidate_end
    return best_start, best_end

def detect_wide_turn_segments(intervals: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not intervals:
        return []

    window_sec = max(args.turn_window_sec, args.min_segment_sec)
    step_sec = max(args.turn_step_sec, 1.0)
    local_zero_time = float(intervals[0]["start_pose"]["timestamp"])
    local_turns = local_path_turns(intervals)

    candidates: list[dict[str, Any]] = []
    start = 0
    while start < len(intervals):
        window_start_time = float(intervals[start]["start_pose"]["timestamp"])
        end = start
        while end + 1 < len(intervals) and float(intervals[end]["end_pose"]["timestamp"]) - window_start_time < window_sec:
            end += 1
        total_turn, support_distance = window_path_turn(intervals, start, end)
        if total_turn is not None and support_distance > 0.03 and abs(total_turn) >= args.min_turn_deg:
            direction = "turn_left" if total_turn > 0.0 else "turn_right"
            core_start, core_end = shrink_path_turn_window(intervals, local_turns, start, end, total_turn, args)
            core_start, core_end = cap_turn_window_duration(intervals, local_turns, core_start, core_end, total_turn, args.max_turn_sec)
            refined_turn, _ = window_path_turn(intervals, core_start, core_end)
            if refined_turn is None or abs(refined_turn) < args.min_turn_deg * 0.6:
                refined_turn = total_turn
            label = "u_turn" if abs(refined_turn) >= args.uturn_deg else direction
            segment = make_segment(intervals, direction, core_start, core_end, local_zero_time)
            segment["label"] = label
            segment["yaw_change_deg"] = round(refined_turn, 3)
            segment["mean_yaw_rate_deg"] = round(refined_turn / segment["duration"], 3) if segment["duration"] > 0 else 0.0
            candidates.append(segment)
            start = core_end + 1
            continue
        next_time = window_start_time + step_sec
        while start < len(intervals) and float(intervals[start]["start_pose"]["timestamp"]) < next_time:
            start += 1
    return merge_overlapping_same_label(candidates)

def shrink_turn_window(
    intervals: list[dict[str, Any]],
    signed_yaws: list[float],
    abs_rates: list[float],
    start: int,
    end: int,
    total_yaw: float,
    args: argparse.Namespace,
) -> tuple[int, int]:
    sign = 1.0 if total_yaw > 0.0 else -1.0
    turn_energy = [max(0.0, signed_yaws[index] * sign) for index in range(start, end + 1)]
    peak = max(turn_energy) if turn_energy else 0.0
    if peak <= 0.0:
        return start, end

    threshold = max(peak * args.turn_core_ratio, args.turn_rate_deg * 0.25)
    strong = [start + index for index, value in enumerate(turn_energy) if value >= threshold]
    if not strong:
        return start, end

    core_start = max(start, strong[0] - 1)
    core_end = min(end, strong[-1] + 1)
    while core_start > start and abs_rates[core_start - 1] >= threshold:
        core_start -= 1
    while core_end < end and abs_rates[core_end + 1] >= threshold:
        core_end += 1
    return core_start, core_end


def merge_overlapping_same_label(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: (segment["start_sec"], segment["end_sec"]))
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        if segment["label"] == previous["label"] and segment["start_sec"] <= previous["end_sec"] + 2:
            merged[-1] = combine_segments(previous, segment)
        elif segment["start_sec"] <= previous["end_sec"]:
            if abs(segment["yaw_change_deg"]) > abs(previous["yaw_change_deg"]):
                merged[-1] = segment
        else:
            merged.append(segment)
    return merged


def merge_event_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"stop": 3, "u_turn": 2, "turn_left": 1, "turn_right": 1}
    ordered = sorted(segments, key=lambda segment: (segment["start_sec"], -priority.get(segment["label"], 0)))
    accepted: list[dict[str, Any]] = []
    for segment in ordered:
        overlaps = [existing for existing in accepted if segment["start_sec"] < existing["end_sec"] and segment["end_sec"] > existing["start_sec"]]
        if not overlaps:
            accepted.append(segment)
            continue
        if all(priority.get(segment["label"], 0) > priority.get(existing["label"], 0) for existing in overlaps):
            accepted = [existing for existing in accepted if existing not in overlaps]
            accepted.append(segment)
    return sorted(accepted, key=lambda segment: segment["start_sec"])


def pose_relative_to_results(path: Path, results_root: Path | None = None) -> str:
    resolved = path.resolve()
    if results_root is not None:
        try:
            return resolved.relative_to(results_root.resolve()).as_posix()
        except ValueError:
            pass
    try:
        return resolved.relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        return path.as_posix()


def load_judge_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"judge results must be a list: {path}")
    result: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        pose_file = item.get("pose_file")
        judge = item.get("judge")
        if isinstance(pose_file, str) and isinstance(judge, str):
            result[pose_file] = judge
    return result


def judge_status(path: Path, args: argparse.Namespace) -> tuple[bool, str]:
    judge_map = load_judge_map(args.judge_results)
    results_root = getattr(args, "results_root", args.judge_results.parent)
    rel_path = pose_relative_to_results(path, results_root)
    judge = judge_map.get(rel_path)
    if judge == "good":
        return True, "good"
    if judge is None and args.allow_unjudged:
        return True, "unjudged_allowed"
    if judge is None:
        return False, f"not_in_judge_results: {rel_path}"
    return False, f"judge_not_good: {judge}"


def scene_frame_window(scene: dict[str, Any], chunk: dict[str, Any]) -> tuple[int, int] | None:
    if chunk.get("logical_start_frame") is None or chunk.get("logical_end_frame") is None:
        return None
    scene_start_frame = scene.get("vggt_start_frame")
    if scene_start_frame is None:
        scene_start_frame = scene.get("source_start_timestamp", scene.get("source_start_sec", 0))
    return int(chunk["logical_start_frame"]) - int(scene_start_frame), int(chunk["logical_end_frame"]) - int(scene_start_frame)


def filter_poses_to_window(
    poses: list[dict[str, float | int | None]],
    frame_window: tuple[int, int] | None,
) -> tuple[list[dict[str, float | int | None]], dict[str, int | str]]:
    if frame_window is None:
        return poses, {
            "chunk_position": "logical_unknown",
            "original_poses": len(poses),
            "drop_start_poses": 0,
            "drop_end_poses": 0,
            "used_poses": len(poses),
        }
    start_frame, end_frame = frame_window
    filtered = [
        pose
        for pose in poses
        if start_frame <= int(pose["scene_frame"]) < end_frame
    ]
    return filtered, {
        "chunk_position": "logical_window",
        "logical_start_scene_frame": start_frame,
        "logical_end_scene_frame": end_frame,
        "original_poses": len(poses),
        "drop_start_poses": max(0, len([pose for pose in poses if int(pose["scene_frame"]) < start_frame])),
        "drop_end_poses": max(0, len([pose for pose in poses if int(pose["scene_frame"]) >= end_frame])),
        "used_poses": len(filtered),
    }


def format_hint_text(result: dict[str, Any]) -> str:
    display_labels = {"turn_left": "turn left", "turn_right": "turn right", "u_turn": "u-turn", "stop": "stop"}
    lines = [
        f"{round(segment['start_sec'])}-{round(segment['end_sec'])} {display_labels[segment['label']]}"
        for segment in result["labels"]
    ]
    return "\n".join(lines)


def load_hint_results(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"videos": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"videos": []}
    if isinstance(data, list):
        return {"videos": data}
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return data
    return {"videos": []}


def write_hint_results(records: list[dict[str, Any]], path: Path) -> None:
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


def update_video_hint_results(path: Path, video_id: str, records: list[dict[str, Any]]) -> None:
    data = load_hint_results(path)
    existing = [item for item in data.get("videos", []) if str(item.get("video_id", "")) != video_id]
    hint_chunks = [
        {
            "scene_index": int(item["scene_index"]),
            "chunk_index": int(item["chunk_index"]),
            "hint": str(item["hint"]),
        }
        for item in records
        if item.get("hint")
    ]
    if hint_chunks:
        existing.append(
            {
                "video_id": video_id,
                "hint_chunks": sorted(hint_chunks, key=lambda item: (item["scene_index"], item["chunk_index"])),
            }
        )
    existing = sorted(existing, key=lambda item: str(item.get("video_id", "")))
    for index, item in enumerate(existing, start=1):
        item["index"] = index
    write_hint_results(existing, path)


def analyze(
    path: Path,
    args: argparse.Namespace,
    scene: dict[str, Any] | None = None,
    chunk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_good, judge_reason = judge_status(path, args)
    if not is_good:
        return {
            "input": str(path),
            "judge_results": str(args.judge_results),
            "skipped": True,
            "skip_reason": judge_reason,
            "labels": [],
        }

    poses = read_poses(path)
    if scene is not None and chunk is not None:
        poses, trim_info = filter_poses_to_window(poses, scene_frame_window(scene, chunk))
    else:
        position = infer_chunk_position(path) if args.chunk_position == "auto" else args.chunk_position
        poses, trim_info = trim_overlap(poses, position, args.overlap_frames)
    if len(poses) < 2:
        return {
            "input": str(path),
            "judge": judge_reason,
            "trim": trim_info,
            "time_range": {"start_sec": 0.0, "end_sec": 0.0},
            "skipped": True,
            "skip_reason": f"too_few_poses_after_logical_window: used_poses={len(poses)}",
            "labels": [],
        }
    time_range = {
        "start_sec": 0.0,
        "end_sec": round(float(poses[-1]["timestamp"]) - float(poses[0]["timestamp"]), 3),
    }
    if len(poses) < args.min_used_poses:
        return {
            "input": str(path),
            "judge": judge_reason,
            "trim": trim_info,
            "time_range": time_range,
            "skipped": True,
            "skip_reason": f"not_full_minute_after_trimming: used_poses={len(poses)} < {args.min_used_poses}",
            "labels": [],
        }

    intervals = build_intervals(poses)
    segments = merge_event_segments(detect_stop_segments(intervals, args) + detect_wide_turn_segments(intervals, args))

    return {
        "input": str(path),
        "judge": judge_reason,
        "trim": trim_info,
        "time_range": time_range,
        "skipped": False,
        "labels": segments,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config.expanduser())
    apply_hint_settings(args, config)
    if args.video_id:
        hint_results_path = metadata_root(config) / "hint_results.json"
        args.allow_unjudged = True
        scenes = load_scenes(config).get("scenes", [])
        results: dict[tuple[int, int], tuple[str | None, str]] = {}
        hint_result_records: list[dict[str, Any]] = []
        written_count = 0
        for scene in scenes:
            if str(scene.get("video_id", "")) != args.video_id:
                continue
            scene_dir = f"scene_{int(scene['scene_index']):04d}"
            out_root = resolve_path(config, "hints_root") / slug(args.video_id) / scene_dir
            for chunk in scene.get("pose_chunks", []):
                scene_index = int(scene["scene_index"])
                chunk_index = int(chunk["chunk_index"])
                if not chunk_is_good_full_minute(config, chunk):
                    results[(scene_index, chunk_index)] = (None, "")
                    hint_result_records.append(
                        {
                            "video_id": args.video_id,
                            "scene_index": scene_index,
                            "chunk_index": chunk_index,
                            "hint": "",
                            "hint_file": None,
                            "skipped": True,
                            "skip_reason": "not_good_full_minute",
                        }
                    )
                    continue
                pose_file_value = chunk.get("pose_file")
                if not pose_file_value:
                    results[(scene_index, chunk_index)] = (None, "")
                    hint_result_records.append(
                        {
                            "video_id": args.video_id,
                            "scene_index": scene_index,
                            "chunk_index": chunk_index,
                            "hint": "",
                            "hint_file": None,
                            "skipped": True,
                            "skip_reason": "missing_pose_file",
                        }
                    )
                    continue
                pose_file = ROOT / str(pose_file_value)
                out_path = out_root / f"chunk_{chunk_index:04d}_hint.txt"
                result = analyze(pose_file, args, scene, chunk)
                hint = format_hint_text(result)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(hint + ("\n" if hint else ""), encoding="utf-8")
                hint_file = str(out_path.relative_to(ROOT))
                results[(scene_index, chunk_index)] = (
                    hint_file,
                    hint.strip(),
                )
                hint_result_records.append(
                    {
                        "video_id": args.video_id,
                        "scene_index": scene_index,
                        "chunk_index": chunk_index,
                        "hint": hint.strip(),
                        "hint_file": hint_file,
                        "skipped": bool(result.get("skipped")),
                        "skip_reason": result.get("skip_reason"),
                    }
                )
                written_count += 1
                print(
                    f"hint video_id={args.video_id} scene={scene_index:04d} "
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
                    hint_file, hint = results[key]
                    if hint_file:
                        chunk["trajectory_hint_file"] = hint_file
                    chunk["trajectory_hint"] = hint
                    chunk["status"] = "hint_extracted"

        with pipeline_lock(config):
            scenes_data = load_scenes(config)
            update_scenes(scenes_data)
            save_scenes(config, scenes_data)

            videos_data = load_videos(config)
            video = find_video(videos_data, args.video_id)
            video["status"] = Status.HINTS_EXTRACTED
            stage_log(video, "extract_hint", "completed", chunks=written_count, considered=len(results))
            save_videos(config, videos_data)

            update_video_hint_results(hint_results_path, args.video_id, hint_result_records)
        print(f"Extracted hints for {written_count} chunk(s) for video_id={args.video_id}; considered={len(results)}")
        return 0

    if args.csv_path is None:
        print("csv_path or --video-id is required", file=sys.stderr)
        return 2
    try:
        result = analyze(args.csv_path, args)
    except Exception as exc:
        print(f"key.py: {exc}", file=sys.stderr)
        return 2

    if args.full:
        text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        text = format_hint_text(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if text else ""), encoding="utf-8")
    else:
        if text:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
