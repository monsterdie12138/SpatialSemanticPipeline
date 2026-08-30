#!/usr/bin/env python3
"""Prepare sampled chunks for VGGT, VLM, and final local output."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    activate_pipeline_environment,
    ensure_video_view,
    ffprobe_frame_count,
    find_video,
    load_config,
    load_scenes,
    load_videos,
    mutate_scenes,
    mutate_videos,
    optional_positive_float,
    optional_positive_int,
    pipeline_env,
    pose_cache_dir,
    resolve_path,
    slug,
    stage_log,
)
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register one video's VLM/output chunks, cutting only missing files."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def chunk_dir(config: dict[str, Any], video: dict[str, Any], scene: dict[str, Any]) -> Path:
    channel = slug(video.get("hf_channel") or video.get("channel") or "unknown")
    return (
        resolve_path(config, "videos_root")
        / channel
        / slug(str(video["video_id"]))
        / f"scene_{int(scene['scene_index']):04d}"
    )


def scene_frame_range(start_sec: float, end_sec: float, fps: float) -> tuple[int, int]:
    if fps <= 0:
        raise ValueError("chunk fps must be > 0")
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


def cut_frame_chunk(
    source_video: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    *,
    sample_fps: float,
    crf: int | None,
    force: bool,
) -> int:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        existing_frame_count = ffprobe_frame_count(output_path)
        if existing_frame_count == end_frame - start_frame:
            return existing_frame_count
    if end_frame <= start_frame:
        raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
    if sample_fps <= 0:
        raise ValueError(f"Invalid sample_fps: {sample_fps}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    tmp.unlink(missing_ok=True)
    start_sec = start_frame / sample_fps
    expected_frames = end_frame - start_frame
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
        "-an",
        "-frames:v",
        str(expected_frames),
        "-c:v",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
    ]
    cmd.extend([str(tmp)])
    subprocess.run(cmd, check=True, env=pipeline_env(config=None))
    tmp.replace(output_path)
    return ffprobe_frame_count(output_path)


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    frame_count = cut_frame_chunk(
        task["source"],
        task["out_path"],
        int(task["start_frame"]),
        int(task["end_frame"]),
        sample_fps=float(task["sample_fps"]),
        crf=task.get("crf"),
        force=bool(task.get("force")),
    )
    task["frame_count"] = frame_count
    return task


def update_chunk_records(
    config: dict[str, Any],
    video: dict[str, Any],
    prepared_records: dict[tuple[int, int], dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    updated = 0

    def update(scenes_data: dict[str, Any]) -> None:
        nonlocal updated
        for scene in scenes_data.get("scenes", []):
            if str(scene.get("video_id", "")) != args.video_id:
                continue
            if args.scene_index is not None and int(scene["scene_index"]) != args.scene_index:
                continue
            existing = {
                int(chunk.get("chunk_index", 0)): dict(chunk)
                for chunk in scene.get("pose_chunks", [])
            }
            records = []
            scene_index = int(scene["scene_index"])
            for (record_scene_index, chunk_index), prepared in sorted(prepared_records.items()):
                if record_scene_index != scene_index:
                    continue
                record = existing.get(chunk_index, {"chunk_index": chunk_index})
                record.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_video_file": prepared["chunk_video_file"],
                        "chunk_start_sec": prepared["chunk_start_sec"],
                        "chunk_end_sec": prepared["chunk_end_sec"],
                        "logical_start_frame": prepared["logical_start_frame"],
                        "logical_end_frame": prepared["logical_end_frame"],
                        "input_start_frame": prepared["input_start_frame"],
                        "input_end_frame": prepared["input_end_frame"],
                        "logical_start_sec": prepared["logical_start_sec"],
                        "logical_end_sec": prepared["logical_end_sec"],
                        "input_start_sec": prepared["input_start_sec"],
                        "input_end_sec": prepared["input_end_sec"],
                        "logical_frame_count": prepared["logical_frame_count"],
                        "theoretical_input_frame_count": prepared["theoretical_input_frame_count"],
                        "overlap_frames": prepared["overlap_frames"],
                        "status": record.get("status", "chunk_split"),
                    }
                )
                records.append(record)
                updated += 1
            if args.chunk_index is None:
                scene["pose_chunks"] = records
            else:
                kept = [
                    chunk
                    for chunk in scene.get("pose_chunks", [])
                    if int(chunk.get("chunk_index", 0)) != args.chunk_index
                ]
                scene["pose_chunks"] = sorted(kept + records, key=lambda item: int(item.get("chunk_index", 0)))

    mutate_scenes(config, update)
    return updated


def main() -> int:
    args = parse_args()
    config = load_config()
    activate_pipeline_environment(config)
    video = find_video(load_videos(config), args.video_id)
    scenes = [
        scene
        for scene in load_scenes(config).get("scenes", [])
        if str(scene.get("video_id", "")) == args.video_id
    ]
    source_video = ROOT / str(video["cache_video_path"])
    chunk_frames = int(config.get("vggt_chunk_frames", config.get("vggt_chunk_sec", 60)))
    overlap_frames = int(config.get("vggt_overlap_frames", config.get("vggt_overlap_sec", 3)))
    sample_fps = optional_positive_float(config.get("sampled_fps", config.get("vggt_input_fps")), 1.0)
    if sample_fps is None:
        raise ValueError("sampled_fps must be fixed when preparing reusable chunks")
    chunk_duration_sec = float(config.get("chunk_duration_sec", chunk_frames / sample_fps))
    if int(round(chunk_duration_sec * sample_fps)) != chunk_frames:
        raise ValueError(
            f"chunk_duration_sec * sampled_fps must equal vggt_chunk_frames: "
            f"{chunk_duration_sec} * {sample_fps} != {chunk_frames}"
        )
    sampled_height = optional_positive_int(config.get("sampled_height", config.get("vggt_input_height")), 480)
    sampled_video = ensure_video_view(
        config,
        args.video_id,
        source_video,
        height=sampled_height,
        fps=sample_fps,
        no_audio=True,
    )
    sampled_frame_count = ffprobe_frame_count(sampled_video)
    prepared_records: dict[tuple[int, int], dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for scene in scenes:
        scene_index = int(scene["scene_index"])
        if args.scene_index is not None and scene_index != args.scene_index:
            continue
        scene_start_timestamp = scene.get("source_start_timestamp", scene.get("source_start_sec"))
        scene_end_timestamp = scene.get("source_end_timestamp", scene.get("source_end_sec"))
        if scene_start_timestamp is None or scene_end_timestamp is None:
            raise ValueError(f"Scene missing timestamp range: video_id={args.video_id} scene_index={scene_index}")
        scene_start_frame, scene_end_frame = scene_frame_range(
            float(scene_start_timestamp),
            float(scene_end_timestamp),
            sample_fps,
        )
        scene_start_frame = min(scene_start_frame, sampled_frame_count)
        scene_end_frame = min(scene_end_frame, sampled_frame_count)
        for chunk_index, (logical_start, logical_end, input_start, input_end) in enumerate(
            scene_chunks(scene_start_frame, scene_end_frame, chunk_frames, overlap_frames)
        ):
            if args.chunk_index is not None and chunk_index != args.chunk_index:
                continue
            vlm_chunk = chunk_dir(config, video, scene) / f"chunk_{chunk_index:04d}.mp4"
            vggt_chunk = pose_cache_dir(config, args.video_id) / f"scene_{scene_index:04d}" / f"chunk_{chunk_index:04d}.mp4"
            prepared_records[(scene_index, chunk_index)] = {
                "chunk_video_file": str(vlm_chunk.relative_to(ROOT)),
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
                "theoretical_input_frame_count": input_end - input_start,
                "overlap_frames": overlap_frames,
            }
            tasks.extend(
                [
                    {
                        "kind": "vlm",
                        "source": sampled_video,
                        "out_path": vlm_chunk,
                        "start_frame": logical_start,
                        "end_frame": logical_end,
                        "expected_frames": logical_end - logical_start,
                        "sample_fps": sample_fps,
                        "crf": None,
                        "force": args.force,
                    },
                    {
                        "kind": "vggt",
                        "source": sampled_video,
                        "out_path": vggt_chunk,
                        "start_frame": input_start,
                        "end_frame": input_end,
                        "expected_frames": input_end - input_start,
                        "sample_fps": sample_fps,
                        "crf": None,
                        "force": args.force,
                    },
                ]
            )
    workers = max(int(config.get("chunk_split_workers", 4)), 1)
    print(
        f"Preparing {len(tasks)} chunk file(s) for video_id={args.video_id} with {workers} worker(s).",
        flush=True,
    )
    if workers == 1:
        for task in tasks:
            finished = run_task(task)
            if int(finished["frame_count"]) != int(finished["expected_frames"]):
                raise ValueError(
                    f"{finished['kind']} chunk frame count mismatch for {finished['out_path']}: "
                    f"got {finished['frame_count']} expected {finished['expected_frames']}"
                )
            print(f"prepared {finished['kind']} {finished['out_path'].relative_to(ROOT)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="split") as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                finished = future.result()
                if int(finished["frame_count"]) != int(finished["expected_frames"]):
                    raise ValueError(
                        f"{finished['kind']} chunk frame count mismatch for {finished['out_path']}: "
                        f"got {finished['frame_count']} expected {finished['expected_frames']}"
                    )
                print(f"prepared {finished['kind']} {finished['out_path'].relative_to(ROOT)}", flush=True)

    chunk_count = update_chunk_records(config, video, prepared_records, args)

    def update_video(videos_data: dict[str, Any]) -> None:
        video_record = find_video(videos_data, args.video_id)
        video_record["status"] = Status.CHUNKS_SPLIT
        stage_log(video_record, "split_chunks", "completed", chunks=chunk_count)

    mutate_videos(config, update_video)
    print(f"Registered {chunk_count} chunk record(s) for video_id={args.video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
