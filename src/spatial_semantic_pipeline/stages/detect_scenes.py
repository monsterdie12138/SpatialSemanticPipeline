#!/usr/bin/env python3
"""Detect scene boundaries with the bundled TransNetV2 and store scene records."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    activate_pipeline_environment,
    ffprobe_duration,
    ffprobe_frame_count,
    find_video,
    load_scenes,
    load_videos,
    mutate_scenes,
    mutate_videos,
    optional_positive_float,
    optional_positive_int,
    pipeline_config,
    save_scenes,
    save_videos,
    stage_log,
    utc_now,
)
from ..providers.base import SceneDetectionRequest
from ..providers.registry import get_scene_detector
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect scenes for one registered video.")
    parser.add_argument("--video-id", required=True)
    return parser.parse_args()


def cut_edges_to_timestamps(cut_edges: list[list[int]], fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError("Video FPS must be > 0 when converting scene cuts to timestamps")
    return [int(next_frame) / fps for _prev_frame, next_frame in cut_edges]


def dedupe_cut_timestamps(cut_timestamps: list[float], min_gap_sec: float) -> list[float]:
    deduped: list[float] = []
    seen_cut_seconds: set[int] = set()
    for timestamp in sorted(float(value) for value in cut_timestamps):
        cut_second = int(math.floor(timestamp))
        if cut_second in seen_cut_seconds:
            continue
        if deduped and timestamp - deduped[-1] < min_gap_sec:
            continue
        deduped.append(timestamp)
        seen_cut_seconds.add(cut_second)
    return deduped


def integer_second_scene_ranges(cut_timestamps: list[float], duration_sec: float) -> list[tuple[int, int]]:
    duration_end_sec = int(math.floor(duration_sec))
    if duration_end_sec <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    start_sec = 0
    for timestamp in sorted(float(value) for value in cut_timestamps):
        if timestamp <= 0 or timestamp >= duration_sec:
            continue
        end_sec = int(math.floor(timestamp))
        if end_sec > start_sec:
            ranges.append((start_sec, end_sec))
        start_sec = max(start_sec, int(math.ceil(timestamp)))
    if duration_end_sec > start_sec:
        ranges.append((start_sec, duration_end_sec))
    return [(start, end) for start, end in ranges if end > start]


def build_scenes(
    video_id: str,
    cut_timestamps: list[float],
    duration_sec: float,
    fps: float,
    min_scene_sec: float,
) -> list[dict]:
    scenes = []
    valid_scene_index = 0
    if duration_sec <= 0 or fps <= 0:
        return scenes

    scene_ranges = integer_second_scene_ranges(cut_timestamps, duration_sec)

    for start_sec, end_sec in scene_ranges:
        duration_sec_int = end_sec - start_sec
        if duration_sec_int < min_scene_sec:
            continue
        start_frame = int(math.ceil(start_sec * fps))
        end_frame = int(math.ceil(end_sec * fps))
        source_frame_count = end_frame - start_frame
        scene_id = f"{video_id}_scene_{valid_scene_index:04d}"
        scenes.append(
            {
                "scene_id": scene_id,
                "video_id": video_id,
                "scene_index": valid_scene_index,
                "source_start_timestamp": start_sec,
                "source_end_timestamp": end_sec,
                "source_start_frame": start_frame,
                "source_end_frame": end_frame,
                "source_frame_count": source_frame_count,
                "pose_chunks": [],
                "created_at": utc_now(),
            }
        )
        valid_scene_index += 1
    return scenes

def main() -> int:
    args = parse_args()
    config = pipeline_config()
    activate_pipeline_environment(config)
    videos = load_videos(config)
    video = find_video(videos, args.video_id)
    video_path = ROOT / video["cache_video_path"]
    duration = float(video.get("duration_sec") or ffprobe_duration(video_path))
    frame_count = int(video.get("frame_count") or ffprobe_frame_count(video_path))
    video_fps = float(video.get("fps") or (frame_count / duration if duration > 0 else 0))
    scene_height = optional_positive_int(config.get("scene_detection_height"), 480)
    scene_fps = optional_positive_float(config.get("scene_detection_fps"), None)
    boundary_fps = scene_fps or video_fps
    scene_threshold = float(config.get("scene_detection_threshold", 0.5))
    scene_cut_dedup_sec = max(float(config.get("scene_cut_dedup_sec", 0.5)), 0.0)
    detector = get_scene_detector(config)
    detection = detector.detect(
        SceneDetectionRequest(
            config=config,
            video_path=video_path,
            height=scene_height,
            fps=scene_fps,
            threshold=scene_threshold,
        )
    )
    cut_edges = detection.cut_edges
    detector_name = detection.detector
    cut_timestamps = cut_edges_to_timestamps(cut_edges, boundary_fps)
    cut_timestamps = dedupe_cut_timestamps(cut_timestamps, scene_cut_dedup_sec)
    total_scene_count = len(integer_second_scene_ranges(cut_timestamps, duration))
    min_scene_sec = float(config.get("scene_min_sec", 2.0))
    new_scenes = build_scenes(args.video_id, cut_timestamps, duration, video_fps, min_scene_sec)

    def update_scenes(scenes_manifest: dict) -> None:
        scenes_manifest["scenes"] = [
            scene for scene in scenes_manifest["scenes"] if str(scene.get("video_id", "")) != args.video_id
        ]
        scenes_manifest["scenes"].extend(new_scenes)
        scenes_manifest["detector"] = detector_name

    mutate_scenes(config, update_scenes)

    def update_video(videos_manifest: dict) -> None:
        video_record = find_video(videos_manifest, args.video_id)
        has_valid_scenes = bool(new_scenes)
        video_record["status"] = Status.SCENES_DETECTED if has_valid_scenes else Status.SKIPPED_NO_VALID_SCENES
        video_record["total_scene_count"] = total_scene_count
        video_record["valid_scene_count"] = len(new_scenes)
        video_record["scene_boundaries"] = cut_timestamps
        video_record["scene_boundary_type"] = "timestamp_sec"
        video_record["scene_detector"] = detector_name
        video_record["scene_detection_height"] = scene_height or "original"
        video_record["scene_detection_fps"] = scene_fps or "original"
        video_record["scene_detection_threshold"] = scene_threshold
        video_record["scene_cut_dedup_sec"] = scene_cut_dedup_sec
        stage_log(
            video_record,
            "scene_detection",
            "completed" if has_valid_scenes else "skipped",
            total_scenes=total_scene_count,
            valid_scenes=len(new_scenes),
            boundaries=len(cut_timestamps),
            detector=detector_name,
            threshold=scene_threshold,
            cut_dedup_sec=scene_cut_dedup_sec,
        )

    mutate_videos(config, update_video)
    print(f"Registered {len(new_scenes)} scenes for video_id={args.video_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
