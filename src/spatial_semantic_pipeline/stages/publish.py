#!/usr/bin/env python3
"""Export one completed video package to the local output tree.

This stage preserves the former publishing layout and filtering rules, but it
performs no network operations.  Files are first assembled below
``output/.staging`` and then atomically moved into the final output tree.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    activate_pipeline_environment,
    chunk_is_full_minute,
    clean_manifest_files,
    cleanup_completed_video,
    ffprobe_duration,
    ffprobe_frame_count,
    ffprobe_video_stream,
    find_video,
    load_config,
    load_scenes,
    load_videos,
    output_root,
    pipeline_lock,
    resolve_chunk_video_path,
    save_scenes,
    save_videos,
    slug,
    stage_log,
    write_clean_scenes_manifest,
    write_clean_videos_manifest,
)
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one processed video package to output/.")
    parser.add_argument("--video-id", required=True)
    return parser.parse_args()


def parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def video_properties(source: Path, video: dict[str, Any]) -> tuple[int | str, int | str, float, int, float]:
    stream = None
    width = video.get("width")
    height = video.get("height")
    fps = video.get("fps")
    frame_count = video.get("frame_count")
    duration = video.get("duration_sec")
    if width is None or height is None or fps is None:
        stream = ffprobe_video_stream(source)
    width = width if width is not None else (stream or {}).get("width", "original")
    height = height if height is not None else (stream or {}).get("height", "original")
    if fps is None:
        fps = parse_fraction((stream or {}).get("avg_frame_rate")) or parse_fraction(
            (stream or {}).get("r_frame_rate")
        )
    if frame_count is None:
        frame_count = ffprobe_frame_count(source)
    if duration is None:
        duration = ffprobe_duration(source)
    return width or "original", height or "original", float(fps), int(frame_count), float(duration)


def stage_copy(source: Path, staging_root: Path, relative_path: str | Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Output source does not exist: {source}")
    destination = staging_root / Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def merge_staging_tree(staging_root: Path, destination_root: Path) -> None:
    for source in sorted(staging_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(staging_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)


def export_standard(config: dict[str, Any], video_id: str) -> dict[str, int]:
    videos = load_videos(config)
    scenes_state = load_scenes(config)
    video = find_video(videos, video_id)
    channel = slug(str(video.get("hf_channel") or video.get("channel") or "unknown"))
    final_root = output_root(config)
    staging_root = final_root / ".staging" / slug(video_id)
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    source_video = ROOT / str(video["cache_video_path"])
    width, height, fps, frame_count, duration = video_properties(source_video, video)
    info_relative = Path("info") / channel / f"{slug(video_id)}.json"
    stage_copy(ROOT / str(video["cache_info_path"]), staging_root, info_relative)

    video_scenes = [
        scene for scene in scenes_state.get("scenes", []) if str(scene.get("video_id", "")) == video_id
    ]

    pose_count = 0
    semantic_count = 0
    chunk_count = 0
    for scene in video_scenes:
        scene_index = int(scene["scene_index"])
        for chunk in scene.get("pose_chunks", []):
            pose_value = chunk.get("pose_file")
            if pose_value:
                pose_source = ROOT / str(pose_value)
                pose_relative = (
                    Path("scenes")
                    / channel
                    / slug(video_id)
                    / f"scene_{scene_index:04d}"
                    / pose_source.name
                )
                stage_copy(pose_source, staging_root, pose_relative)
                chunk["output_pose_file"] = pose_relative.as_posix()
                pose_count += 1

            semantic_value = chunk.get("semantic_file")
            if chunk_is_full_minute(config, chunk) and semantic_value:
                semantic_source = ROOT / str(semantic_value)
                semantic_relative = (
                    Path("semantics")
                    / channel
                    / slug(video_id)
                    / f"scene_{scene_index:04d}"
                    / semantic_source.name
                )
                stage_copy(semantic_source, staging_root, semantic_relative)
                chunk["output_semantic_file"] = semantic_relative.as_posix()
                semantic_count += 1

            chunk_source = resolve_chunk_video_path(config, video, scene, chunk)
            if chunk_source is not None:
                chunk_relative = (
                    Path("chunks")
                    / channel
                    / slug(video_id)
                    / f"scene_{scene_index:04d}"
                    / chunk_source.name
                )
                stage_copy(chunk_source, staging_root, chunk_relative)
                chunk["output_chunk_video_file"] = chunk_relative.as_posix()
                chunk_count += 1

    def update_scenes(data: dict[str, Any]) -> None:
        data["scenes"] = [
            scene for scene in data.get("scenes", []) if str(scene.get("video_id", "")) != video_id
        ]
        data["scenes"].extend(video_scenes)

    def update_video(data: dict[str, Any]) -> None:
        record = find_video(data, video_id)
        record["status"] = Status.EXPORTED
        record["output_info_path"] = info_relative.as_posix()
        record["output_width"] = width
        record["output_height"] = height
        record["output_fps"] = fps
        record["output_frame_count"] = frame_count
        record["output_duration_sec"] = duration
        stage_log(
            record,
            "export_output",
            "completed",
            pose_files=pose_count,
            semantic_files=semantic_count,
            chunk_videos=chunk_count,
        )

    with pipeline_lock(config):
        scenes = load_scenes(config)
        update_scenes(scenes)
        save_scenes(config, scenes)
        state = load_videos(config)
        update_video(state)
        save_videos(config, state)
        write_clean_videos_manifest(config)
        write_clean_scenes_manifest(config)
        for manifest, relative in clean_manifest_files(config):
            stage_copy(manifest, staging_root, relative)
        merge_staging_tree(staging_root, final_root)
        cleanup_completed_video(config, video_id)

    shutil.rmtree(staging_root, ignore_errors=True)
    try:
        staging_root.parent.rmdir()
    except OSError:
        # Another output worker may still be using the shared staging parent.
        pass
    return {
        "poses": pose_count,
        "semantics": semantic_count,
        "chunks": chunk_count,
    }


def main() -> int:
    args = parse_args()
    config = load_config()
    activate_pipeline_environment(config)
    counts = export_standard(config, args.video_id)
    print(
        f"Exported video_id={args.video_id} to {output_root(config)} "
        + " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
