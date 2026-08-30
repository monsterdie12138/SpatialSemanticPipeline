#!/usr/bin/env python3
"""Run VGGT on scene windows split into overlapping frame-based chunks."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    activate_pipeline_environment,
    canonical_chunk_video_path,
    ensure_video_view,
    ffprobe_video_stream,
    find_video,
    load_scenes,
    load_videos,
    mutate_scenes,
    mutate_videos,
    optional_positive_float,
    pipeline_env,
    pipeline_config,
    pose_cache_dir,
    read_json,
    video_view_frame_map_path,
    stage_log,
)
from ..providers.base import PoseEstimationJob, PoseEstimationRequest
from ..providers.registry import get_pose_estimator
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate scene trajectories with VGGT.")
    parser.add_argument("--video-id", required=True)
    return parser.parse_args()


def scene_frame_range(start_sec: float, end_sec: float, fps: float) -> tuple[int, int]:
    if fps <= 0:
        raise ValueError("VGGT input fps must be > 0")
    start_frame = int(math.ceil(start_sec * fps))
    end_frame = int(math.ceil(end_sec * fps))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def frame_to_sec(frame_index: int, fps: float) -> float:
    return frame_index / fps


def scene_chunks(
    start_frame: int,
    end_frame: int,
    chunk_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int, int, int]]:
    if chunk_frames <= 0:
        raise ValueError("vggt_chunk_frames must be > 0")
    if overlap_frames < 0:
        raise ValueError("vggt_overlap_frames must be >= 0")
    if end_frame <= start_frame:
        return []

    chunks = []
    cursor = start_frame
    while cursor < end_frame:
        logical_start = cursor
        logical_end = min(cursor + chunk_frames, end_frame)
        input_start = max(start_frame, logical_start - overlap_frames)
        input_end = min(end_frame, logical_end + overlap_frames)
        chunks.append((logical_start, logical_end, input_start, input_end))
        cursor = logical_end
    return chunks


def parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    return float(value)


def infer_video_fps(video_path: Path) -> float:
    stream = ffprobe_video_stream(video_path)
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError(f"Could not infer FPS for {video_path}")
    return fps


def ffprobe_frame_count(video_path: Path) -> int:
    config = pipeline_config()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=pipeline_env(config),
    )
    value = result.stdout.strip()
    return int(value) if value and value != "N/A" else 0


def sampled_frame_to_source_frame(sampled_frame: int, sample_fps: float, source_fps: float) -> int:
    if sample_fps <= 0 or source_fps <= 0:
        raise ValueError("sample_fps and source_fps must be > 0")
    return int(math.ceil(sampled_frame_to_source_timestamp(sampled_frame, sample_fps) * source_fps))


def sampled_frame_to_source_timestamp(sampled_frame: int, sample_fps: float) -> float:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    return sampled_frame / sample_fps


def load_sampled_frame_map(map_path: Path) -> dict[int, dict[str, float | int]]:
    payload = read_json(map_path, {})
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Missing sampled frame map: {map_path}")
    result: dict[int, dict[str, float | int]] = {}
    for entry in frames:
        sampled_frame = int(entry["sampled_frame"])
        result[sampled_frame] = {
            "source_frame": int(entry["source_frame"]),
            "source_timestamp": float(entry["source_timestamp"]),
        }
    return result


def rewrite_pose_csv_with_frame_stamps(
    pose_file: Path,
    input_start_frame: int,
    scene_start_frame: int,
    sampled_frame_map: dict[int, dict[str, float | int]],
) -> tuple[int, set[int]]:
    with pose_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"x", "y", "z", "qx", "qy", "qz", "qw"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Pose CSV missing required columns {sorted(missing)}: {pose_file}")
    if "frame" not in (reader.fieldnames or []) and "scene_frame" not in (reader.fieldnames or []):
        raise ValueError(f"Pose CSV missing frame or scene_frame column: {pose_file}")

    output_fields = ["scene_frame", "source_frame", "source_timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    tmp = pose_file.with_suffix(pose_file.suffix + ".tmp")
    scene_frames: set[int] = set()
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            if "frame" in row and row["frame"] not in (None, ""):
                local_frame = int(row["frame"])
                sampled_video_frame = input_start_frame + local_frame
                scene_frame = sampled_video_frame - scene_start_frame
            else:
                scene_frame = int(float(row["scene_frame"]))
                sampled_video_frame = scene_start_frame + scene_frame
            source_entry = sampled_frame_map.get(sampled_video_frame)
            if source_entry is None:
                raise ValueError(
                    f"Missing original timestamp mapping for sampled frame {sampled_video_frame}: {pose_file}"
                )
            source_timestamp = float(source_entry["source_timestamp"])
            source_frame = int(source_entry["source_frame"])
            scene_frames.add(scene_frame)
            writer.writerow(
                {
                    "scene_frame": scene_frame,
                    "source_frame": source_frame,
                    "source_timestamp": f"{source_timestamp:.9f}",
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "qx": row["qx"],
                    "qy": row["qy"],
                    "qz": row["qz"],
                    "qw": row["qw"],
                }
            )
    tmp.replace(pose_file)
    return len(rows), scene_frames


def read_pose_scene_frames(pose_file: Path) -> set[int]:
    with pose_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "scene_frame" in fieldnames:
            scene_frame_column = "scene_frame"
        elif "frame_idx" in fieldnames:
            scene_frame_column = "frame_idx"
        else:
            raise ValueError(f"Pose CSV has no scene frame column: {pose_file}")
        return {int(float(row[scene_frame_column])) for row in reader if row.get(scene_frame_column) is not None}


def save_scene_progress(config: dict, video_id: str, scene: dict) -> None:
    def update_scenes(scenes_data: dict) -> None:
        scenes_data["scenes"] = [
            existing
            for existing in scenes_data["scenes"]
            if not (
                str(existing.get("video_id", "")) == video_id
                and int(existing.get("scene_index", -1)) == int(scene.get("scene_index", -2))
            )
        ]
        scenes_data["scenes"].append(dict(scene))

    mutate_scenes(config, update_scenes)


def main() -> int:
    args = parse_args()
    config = pipeline_config()
    activate_pipeline_environment(config)
    videos = load_videos(config)
    scenes_manifest = load_scenes(config)
    video = find_video(videos, args.video_id)
    original_video = ROOT / video["cache_video_path"]
    source_fps = float(video.get("fps") or infer_video_fps(original_video))
    chunk_frames = int(config.get("vggt_chunk_frames", config.get("vggt_chunk_sec", 60)))
    overlap_frames = int(config.get("vggt_overlap_frames", config.get("vggt_overlap_sec", 3)))
    configured_fps = optional_positive_float(config.get("sampled_fps", config.get("vggt_input_fps")), 1.0)
    sample_fps = infer_video_fps(original_video) if configured_fps is None else configured_fps
    sampled_height = config.get("sampled_height", config.get("vggt_input_height", 480))
    sampled_height = int(sampled_height) if sampled_height is not None else None
    sampled_video = ensure_video_view(
        config,
        args.video_id,
        original_video,
        height=sampled_height,
        fps=sample_fps,
        no_audio=True,
    )
    sampled_frame_map = load_sampled_frame_map(video_view_frame_map_path(sampled_video))
    pose_estimator = get_pose_estimator(config)

    work_root = pose_cache_dir(config, args.video_id)
    video_scenes = [
        scene for scene in scenes_manifest["scenes"] if str(scene.get("video_id", "")) == args.video_id
    ]

    total_chunks = 0
    for scene in video_scenes:
        scene_dir = work_root / f"scene_{int(scene['scene_index']):04d}"
        existing_chunks = {
            int(chunk.get("chunk_index", 0)): dict(chunk)
            for chunk in scene.get("pose_chunks", [])
        }
        scene["pose_chunks"] = []
        scene_start_timestamp = scene.get("source_start_timestamp", scene.get("source_start_sec"))
        scene_end_timestamp = scene.get("source_end_timestamp", scene.get("source_end_sec"))
        if scene_start_timestamp is None or scene_end_timestamp is None:
            raise ValueError(
                f"Scene missing timestamp range: video_id={args.video_id} "
                f"scene_index={scene.get('scene_index')}"
            )
        scene_start_frame, scene_end_frame = scene_frame_range(
            float(scene_start_timestamp),
            float(scene_end_timestamp),
            sample_fps,
        )
        if scene_end_frame <= scene_start_frame:
            raise ValueError(
                f"Scene has no frames after mapping to {sample_fps}fps view: "
                f"video_id={args.video_id} scene_index={scene.get('scene_index')} "
                f"range={scene_start_frame}-{scene_end_frame}"
            )
        expected_scene_frames = set(range(scene_end_frame - scene_start_frame))
        covered_scene_frames: set[int] = set()
        scene["vggt_start_frame"] = scene_start_frame
        scene["vggt_end_frame"] = scene_end_frame
        chunk_records = []
        for chunk_index, (logical_start, logical_end, input_start, input_end) in enumerate(
            scene_chunks(scene_start_frame, scene_end_frame, chunk_frames, overlap_frames)
        ):
            chunk_video = scene_dir / f"chunk_{chunk_index:04d}.mp4"
            pose_file = scene_dir / f"chunk_{chunk_index:04d}_poses.csv"
            existing_chunk = existing_chunks.get(chunk_index, {})
            existing_pose_file = ROOT / str(existing_chunk.get("pose_file", "")) if existing_chunk.get("pose_file") else None
            if existing_pose_file is None and pose_file.is_file():
                existing_pose_file = pose_file
                existing_chunk["pose_file"] = str(pose_file.relative_to(ROOT))
            if existing_pose_file is not None and existing_pose_file.is_file():
                _pose_frame_count, pose_scene_frames = rewrite_pose_csv_with_frame_stamps(
                    existing_pose_file,
                    input_start,
                    scene_start_frame,
                    sampled_frame_map,
                )
                covered_scene_frames.update(
                    frame for frame in pose_scene_frames if 0 <= frame < scene_end_frame - scene_start_frame
                )
                existing_chunk["chunk_video_file"] = str(
                    canonical_chunk_video_path(config, video, scene, {"chunk_index": chunk_index}).relative_to(ROOT)
                )
                existing_chunk["chunk_start_sec"] = round(frame_to_sec(logical_start, sample_fps), 3)
                existing_chunk["chunk_end_sec"] = round(frame_to_sec(logical_end, sample_fps), 3)
                existing_chunk["logical_start_frame"] = logical_start
                existing_chunk["logical_end_frame"] = logical_end
                existing_chunk["input_start_frame"] = input_start
                existing_chunk["input_end_frame"] = input_end
                existing_chunk["logical_start_sec"] = round(frame_to_sec(logical_start, sample_fps), 3)
                existing_chunk["logical_end_sec"] = round(frame_to_sec(logical_end, sample_fps), 3)
                existing_chunk["input_start_sec"] = round(frame_to_sec(input_start, sample_fps), 3)
                existing_chunk["input_end_sec"] = round(frame_to_sec(input_end, sample_fps), 3)
                existing_chunk["logical_frame_count"] = logical_end - logical_start
                existing_chunk["theoretical_input_frame_count"] = input_end - input_start
                existing_chunk["overlap_frames"] = overlap_frames
                scene["pose_chunks"].append(existing_chunk)
                total_chunks += 1
                continue
            if not chunk_video.is_file():
                raise FileNotFoundError(f"Missing prepared VGGT chunk from 03_split_chunks.py: {chunk_video}")
            actual_input_frame_count = ffprobe_frame_count(chunk_video)
            theoretical_input_frame_count = input_end - input_start
            is_full_middle_chunk = (
                logical_end - logical_start == chunk_frames
                and input_start < logical_start
                and input_end > logical_end
            )
            if is_full_middle_chunk and actual_input_frame_count != theoretical_input_frame_count:
                raise ValueError(
                    f"Full chunk frame count mismatch for {chunk_video}: got {actual_input_frame_count}, "
                    f"expected {theoretical_input_frame_count}"
                )
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "existing_chunk": existing_chunk,
                    "logical_start": logical_start,
                    "logical_end": logical_end,
                    "input_start": input_start,
                    "input_end": input_end,
                    "chunk_video": chunk_video,
                    "pose_file": pose_file,
                    "actual_input_frame_count": actual_input_frame_count,
                    "theoretical_input_frame_count": theoretical_input_frame_count,
                }
            )

        if chunk_records:
            pose_estimator.estimate(
                PoseEstimationRequest(
                    config=config,
                    jobs=tuple(
                        PoseEstimationJob(
                            input_video=Path(record["chunk_video"]),
                            output_pose=Path(record["pose_file"]),
                            sample_fps=sample_fps,
                        )
                        for record in chunk_records
                    ),
                    work_dir=scene_dir,
                )
            )

        for record in chunk_records:
            chunk_index = int(record["chunk_index"])
            existing_chunk = dict(record.get("existing_chunk") or {})
            logical_start = int(record["logical_start"])
            logical_end = int(record["logical_end"])
            input_start = int(record["input_start"])
            input_end = int(record["input_end"])
            chunk_video = Path(record["chunk_video"])
            pose_file = Path(record["pose_file"])
            actual_input_frame_count = int(record["actual_input_frame_count"])
            theoretical_input_frame_count = int(record["theoretical_input_frame_count"])
            pose_frame_count, pose_scene_frames = rewrite_pose_csv_with_frame_stamps(
                pose_file,
                input_start,
                scene_start_frame,
                sampled_frame_map,
            )
            covered_scene_frames.update(
                frame for frame in pose_scene_frames if 0 <= frame < scene_end_frame - scene_start_frame
            )
            expected_frame_count = actual_input_frame_count
            if pose_frame_count != expected_frame_count:
                raise ValueError(
                    f"Pose row count mismatch for {pose_file}: got {pose_frame_count}, "
                    f"expected {expected_frame_count}"
            )
            chunk_video.unlink(missing_ok=True)
            chunk_record = {
                "chunk_index": chunk_index,
                "chunk_video_file": str(
                    canonical_chunk_video_path(config, video, scene, {"chunk_index": chunk_index}).relative_to(ROOT)
                ),
                "chunk_start_sec": round(frame_to_sec(logical_start, sample_fps), 3),
                "chunk_end_sec": round(frame_to_sec(logical_end, sample_fps), 3),
                "logical_start_frame": logical_start,
                "logical_end_frame": logical_end,
                "input_start_frame": input_start,
                "input_end_frame": input_end,
                "logical_start_sec": round(frame_to_sec(logical_start, sample_fps), 3),
                "logical_end_sec": round(frame_to_sec(logical_end, sample_fps), 3),
                "input_start_sec": round(frame_to_sec(input_start, sample_fps), 3),
                "input_end_sec": round(frame_to_sec(input_end, sample_fps), 3),
                "logical_frame_count": logical_end - logical_start,
                "input_frame_count": actual_input_frame_count,
                "theoretical_input_frame_count": theoretical_input_frame_count,
                "overlap_frames": overlap_frames,
                "pose_file": str(pose_file.relative_to(ROOT)),
                "status": "completed",
            }
            chunk_record = {key: value for key, value in chunk_record.items() if value is not None}
            scene["pose_chunks"].append(chunk_record)
            save_scene_progress(config, args.video_id, scene)
            total_chunks += 1
        missing_scene_frames = sorted(expected_scene_frames - covered_scene_frames)
        if missing_scene_frames:
            preview = missing_scene_frames[:20]
            raise ValueError(
                f"Missing VGGT poses for video_id={args.video_id} scene_index={scene.get('scene_index')}: "
                f"{len(missing_scene_frames)} missing scene_frame values, first={preview}"
            )
        scene["scene_frame_count"] = scene_end_frame - scene_start_frame
        shutil.rmtree(scene_dir / "frames", ignore_errors=True)

    def update_scenes(scenes_data: dict) -> None:
        scenes_data["scenes"] = [
            scene for scene in scenes_data["scenes"] if str(scene.get("video_id", "")) != args.video_id
        ]
        scenes_data["scenes"].extend(video_scenes)

    mutate_scenes(config, update_scenes)

    def update_video(videos_data: dict) -> None:
        video_record = find_video(videos_data, args.video_id)
        video_record["status"] = Status.POSES_ESTIMATED
        video_record["sampled_fps"] = sample_fps
        stage_log(video_record, "vggt", "completed", chunks=total_chunks)

    mutate_videos(config, update_video)
    print(f"Wrote {total_chunks} pose chunks for video_id={args.video_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
