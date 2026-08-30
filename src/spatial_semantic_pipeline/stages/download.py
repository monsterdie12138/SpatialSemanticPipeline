#!/usr/bin/env python3
"""Download one video source into cache and register it by YouTube video_id."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    activate_pipeline_environment,
    ffprobe_duration,
    ffprobe_frame_count,
    ffprobe_video_stream,
    find_video,
    load_videos,
    mutate_videos,
    pipeline_config,
    pipeline_env,
    slug,
    source_is_registered,
    stable_video_id,
    optional_positive_int,
    optional_positive_float,
    stage_log,
    utc_now,
    video_cache_dir,
)
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one video source.")
    parser.add_argument("--source", required=True)
    return parser.parse_args()


def infer_frame_count(duration: float, fps: float | None, video_path: Path) -> int:
    if duration > 0 and fps is not None and fps > 0:
        return int(round(duration * fps))
    return ffprobe_frame_count(video_path)


def build_yt_dlp_cmd(
    source: str,
    output_template: Path,
    height: int | None,
    fps: float | None = None,
    js_runtimes: list[str] | None = None,
    format_selector: str | None = None,
    write_description: bool = True,
    write_thumbnail: bool = True,
) -> list[str]:
    fps_text = str(int(fps)) if fps is not None and fps.is_integer() else str(fps)
    fps_filter = f"[fps<={fps_text}]" if fps is not None else ""
    fallback_height = 480 if height and height >= 720 else None
    configured_format = str(format_selector or "").strip()
    if configured_format.lower() in {"", "auto", "default"}:
        configured_format = ""
    if configured_format:
        fmt = configured_format
    elif height is None:
        fmt = f"bv{fps_filter}[ext=mp4]/bv{fps_filter}/bv[ext=mp4]/bv" if fps_filter else "bv[ext=mp4]/bv"
    elif fps_filter:
        formats = [
            f"bv[height={height}]{fps_filter}[ext=mp4]",
            f"bv[height={height}]{fps_filter}",
            f"bv[height={height}][ext=mp4]",
            f"bv[height={height}]",
        ]
        if fallback_height:
            formats.extend(
                [
                    f"bv[height={fallback_height}]{fps_filter}[ext=mp4]",
                    f"bv[height={fallback_height}]{fps_filter}",
                    f"bv[height={fallback_height}][ext=mp4]",
                    f"bv[height={fallback_height}]",
                    f"bv[height<={fallback_height}]{fps_filter}[ext=mp4]",
                    f"bv[height<={fallback_height}]{fps_filter}",
                    f"bv[height<={fallback_height}][ext=mp4]",
                    f"bv[height<={fallback_height}]",
                ]
            )
        formats.extend([f"bv{fps_filter}[ext=mp4]", f"bv{fps_filter}", "bv[ext=mp4]", "bv"])
        fmt = "/".join(formats)
    else:
        formats = [f"bv[height={height}][ext=mp4]", f"bv[height={height}]"]
        if fallback_height:
            formats.extend(
                [
                    f"bv[height={fallback_height}][ext=mp4]",
                    f"bv[height={fallback_height}]",
                    f"bv[height<={fallback_height}][ext=mp4]",
                    f"bv[height<={fallback_height}]",
                ]
            )
        formats.extend(["bv[ext=mp4]", "bv"])
        fmt = "/".join(formats)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--write-info-json",
    ]
    if write_description:
        cmd.append("--write-description")
    if write_thumbnail:
        cmd.extend(["--write-thumbnail", "--convert-thumbnails", "webp"])
    cmd.extend([
        "-f", fmt,
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "-o",
        str(output_template),
    ])
    for runtime in js_runtimes or []:
        cmd.extend(["--js-runtimes", runtime])
    cmd.append(source)
    return cmd


def parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    return float(value)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def bool_config(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_yt_dlp_once(
    source: str,
    output_template: Path,
    height: int | None,
    fps: float | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run exactly one direct yt-dlp attempt and propagate any failure."""
    cmd = build_yt_dlp_cmd(
        source,
        output_template,
        height,
        fps,
        as_list(config.get("youtube_js_runtimes")),
        str(config.get("youtube_format") or "auto"),
        bool_config(config, "youtube_write_description", True),
        bool_config(config, "youtube_write_thumbnail", True),
    )
    subprocess.run(cmd, check=True, env=pipeline_env(config))
    return {"proxy": None, "attempts": [{"attempt": 1, "returncode": 0}]}


def find_package(work_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    info_paths = sorted(work_dir.glob("*.info.json"))
    if not info_paths:
        raise FileNotFoundError(f"yt-dlp did not write an info json in {work_dir}")
    info_path = info_paths[0]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stem = info_path.name[: -len(".info.json")]
    for suffix in (".mp4", ".mkv", ".webm"):
        media = work_dir / f"{stem}{suffix}"
        if media.exists():
            return media, info_path, info
    raise FileNotFoundError(f"yt-dlp did not write a media file next to {info_path}")


def main() -> int:
    args = parse_args()
    config = pipeline_config()
    activate_pipeline_environment(config)
    video_id = stable_video_id(args.source)
    skipped = False

    def append_record(videos: dict[str, Any]) -> None:
        nonlocal skipped
        if source_is_registered(videos, args.source):
            skipped = True
            return
        videos["videos"].append(
            {
                "video_id": video_id,
                "source_url": args.source,
                "status": Status.DOWNLOADING,
                "created_at": utc_now(),
            }
        )

    mutate_videos(config, append_record)
    if skipped:
        print(f"Skip duplicate source: {args.source}")
        return 0

    work_dir = video_cache_dir(config, video_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    height = optional_positive_int(config.get("youtube_download_height"), None)
    download_fps = optional_positive_float(config.get("youtube_download_fps"), None)
    output_template = work_dir / "%(id)s.%(ext)s"

    try:
        canonical_video = work_dir / "original.mp4"
        canonical_info = work_dir / "info.json"
        reused_cache = canonical_video.is_file() and canonical_info.is_file()
        if reused_cache:
            info = json.loads(canonical_info.read_text(encoding="utf-8"))
            download_result = {"proxy": "cache", "attempts": []}
        else:
            download_result = run_yt_dlp_once(args.source, output_template, height, download_fps, config)
            media_path, info_path, info = find_package(work_dir)
            if media_path != canonical_video:
                shutil.move(str(media_path), canonical_video)
            if info_path != canonical_info:
                shutil.move(str(info_path), canonical_info)

        downloaded_id = str(info.get("id") or video_id)
        duration = ffprobe_duration(canonical_video)
        stream = ffprobe_video_stream(canonical_video)
        fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
        frame_count = infer_frame_count(duration, fps, canonical_video)

        def mark_downloaded(videos: dict[str, Any]) -> None:
            video = find_video(videos, video_id)
            if downloaded_id != video_id:
                video["requested_video_id"] = video_id
                video["video_id"] = downloaded_id
            video.update(
                {
                    "status": Status.DOWNLOADED,
                    "youtube_download_height": height or "original",
                    "youtube_download_fps": download_fps or "original",
                    "download_proxy": download_result.get("proxy"),
                    "download_attempts": download_result.get("attempts", []),
                    "title": info.get("title"),
                    "uploader": info.get("uploader") or info.get("channel"),
                    "webpage_url": info.get("webpage_url") or args.source,
                    "duration_sec": duration,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "cache_video_path": str(canonical_video.relative_to(ROOT)),
                    "cache_info_path": str(canonical_info.relative_to(ROOT)),
                    "channel": info.get("uploader_id") or info.get("channel_id") or info.get("uploader") or info.get("channel"),
                    "hf_channel": slug(info.get("uploader_id") or info.get("channel_id") or info.get("uploader") or info.get("channel")),
                    "hf_stem": slug(downloaded_id),
                }
            )
            stage_log(video, "download", "reused_cache" if reused_cache else "completed")

        mutate_videos(config, mark_downloaded)
        video_id = downloaded_id
    except Exception as exc:
        if bool_config(config, "download_rollback_on_failure", True):
            def rollback_download(videos: dict[str, Any]) -> None:
                videos["videos"] = [
                    video
                    for video in videos.get("videos", [])
                    if video.get("source_url") != args.source and video.get("video_id") != video_id
                ]

            mutate_videos(config, rollback_download)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        def mark_failed(videos: dict[str, Any]) -> None:
            try:
                video = find_video(videos, video_id)
            except KeyError:
                return
            video["status"] = Status.DOWNLOAD_FAILED
            video["error"] = str(exc)
            stage_log(video, "download", "failed", error=str(exc))

        mutate_videos(config, mark_failed)
        raise

    action = "Reused cached download" if reused_cache else "Downloaded"
    print(f"{action} video_id={video_id}: {canonical_video.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
