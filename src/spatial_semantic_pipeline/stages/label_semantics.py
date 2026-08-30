#!/usr/bin/env python3
"""Call Gemini for one prepared prompt/video clip."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from ..context import (
    DEFAULT_CONFIG,
    chunk_is_full_minute,
    find_video,
    load_config,
    load_scenes,
    load_videos,
    mutate_scenes,
    mutate_videos,
    path_from_config,
    resolve_chunk_video_path,
    resolve_path,
    slug,
    stage_log,
    vlm_response_has_inner_monologue,
    vlm_response_missing_video,
    write_text,
)
from ..providers.base import SemanticLabelRequest, SemanticLabeler
from ..providers.registry import get_semantic_labeler
from ..core.stages import Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Gemini on one video clip.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video-id", help="Call Gemini for all chunk clips of one registered video.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-without-hint-file", type=Path)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--text-output", type=Path)
    return parser.parse_args()


def prompt_with_hint(prompt_template: str, hint: str) -> str:
    replacement = hint.strip()
    marker = "{Insert VGGT data here: e.g., 12-15s: Turn Left / 28-30s: Stationary}"
    return prompt_template.replace(marker, replacement)


def prompt_for_chunk(prompt_template: str, prompt_without_hint_template: str, chunk: dict[str, Any]) -> str:
    hint = str(chunk.get("trajectory_hint") or "").strip()
    if hint:
        return prompt_with_hint(prompt_template, hint)
    return prompt_without_hint_template


def semantic_output_path(config: dict[str, Any], video_id: str, scene_index: int, chunk_index: int) -> Path:
    return (
        resolve_path(config, "semantics_root")
        / slug(video_id)
        / f"scene_{scene_index:04d}"
        / f"chunk_{chunk_index:04d}_semantics.txt"
    )


def valid_semantic_file(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return bool(text.strip()) and not vlm_response_missing_video(text) and vlm_response_has_inner_monologue(text)


def process_chunk(
    config: dict[str, Any],
    prompt_template: str,
    prompt_without_hint_template: str,
    video_id: str,
    scene: dict[str, Any],
    chunk: dict[str, Any],
    clip: Path,
    labeler: SemanticLabeler | None = None,
) -> dict[str, Any]:
    scene_index = int(scene["scene_index"])
    chunk_index = int(chunk["chunk_index"])
    output = semantic_output_path(config, video_id, scene_index, chunk_index)
    labeler = labeler or get_semantic_labeler(config)
    response = labeler.label(
        SemanticLabelRequest(
            config=config,
            prompt=prompt_for_chunk(prompt_template, prompt_without_hint_template, chunk),
            clip=clip,
        )
    )
    text = response.text
    write_text(output, text)
    return {
        "scene_index": scene_index,
        "chunk_index": chunk_index,
        "chunk_video_file": str(clip.relative_to(ROOT)),
        "semantic_file": str(output.relative_to(ROOT)),
        "semantic_text": text,
    }


def process_video(config: dict[str, Any], video_id: str, prompt_file: Path, prompt_without_hint_file: Path) -> int:
    prompt_template = prompt_file.read_text(encoding="utf-8")
    prompt_without_hint_template = prompt_without_hint_file.read_text(encoding="utf-8")
    video = find_video(load_videos(config), video_id)
    scenes = [
        scene
        for scene in load_scenes(config).get("scenes", [])
        if str(scene.get("video_id", "")) == video_id
    ]
    tasks: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    results: dict[tuple[int, int], dict[str, Any]] = {}
    for scene in scenes:
        scene_index = int(scene["scene_index"])
        for chunk in sorted(scene.get("pose_chunks", []), key=lambda item: int(item.get("chunk_index", 0))):
            if not chunk.get("chunk_video_file") or not chunk_is_full_minute(config, chunk):
                continue
            chunk_index = int(chunk["chunk_index"])
            clip = resolve_chunk_video_path(config, video, scene, chunk)
            if clip is None:
                continue
            output = semantic_output_path(config, video_id, scene_index, chunk_index)
            semantic_file = chunk.get("semantic_file")
            existing = ROOT / str(semantic_file) if semantic_file else output
            if existing.is_file():
                if valid_semantic_file(existing):
                    text = existing.read_text(encoding="utf-8")
                    results[(scene_index, chunk_index)] = {
                        "scene_index": scene_index,
                        "chunk_index": chunk_index,
                        "chunk_video_file": str(clip.relative_to(ROOT)),
                        "semantic_file": str(existing.relative_to(ROOT)),
                        "semantic_text": text,
                    }
                    continue
                existing.unlink(missing_ok=True)
            tasks.append((scene, chunk, clip))
    workers = max(int(config.get("vlm_workers", 4)), 1)
    failures: list[tuple[tuple[int, int], Exception]] = []

    def update_completed_chunk(scenes_data: dict[str, Any], completed: dict[str, Any]) -> None:
        for scene in scenes_data.get("scenes", []):
            if str(scene.get("video_id", "")) != video_id:
                continue
            if int(scene["scene_index"]) != int(completed["scene_index"]):
                continue
            for chunk in scene.get("pose_chunks", []):
                if int(chunk["chunk_index"]) == int(completed["chunk_index"]):
                    chunk.update(
                        {
                            "chunk_video_file": completed["chunk_video_file"],
                            "semantic_file": completed["semantic_file"],
                            "semantic_text": completed["semantic_text"],
                            "status": "semantics_labeled",
                        }
                    )

    for completed in results.values():
        mutate_scenes(config, lambda scenes_data, completed=completed: update_completed_chunk(scenes_data, completed))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm") as executor:
        futures = {
            executor.submit(
                process_chunk,
                config,
                prompt_template,
                prompt_without_hint_template,
                video_id,
                scene,
                chunk,
                clip,
            ): (
                int(scene["scene_index"]),
                int(chunk["chunk_index"]),
            )
            for scene, chunk, clip in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                completed = future.result()
            except Exception as exc:
                failures.append((key, exc))
                print(f"gemini_failed video_id={video_id} scene={key[0]:04d} chunk={key[1]:04d}: {exc}", flush=True)
                continue
            results[key] = completed
            mutate_scenes(config, lambda scenes_data, completed=completed: update_completed_chunk(scenes_data, completed))
            print(f"gemini video_id={video_id} scene={key[0]:04d} chunk={key[1]:04d}", flush=True)

    def update_scenes(scenes_data: dict[str, Any]) -> None:
        for scene in scenes_data.get("scenes", []):
            if str(scene.get("video_id", "")) != video_id:
                continue
            scene_index = int(scene["scene_index"])
            for chunk in scene.get("pose_chunks", []):
                key = (scene_index, int(chunk["chunk_index"]))
                if key in results:
                    chunk.update(
                        {
                            "chunk_video_file": results[key]["chunk_video_file"],
                            "semantic_file": results[key]["semantic_file"],
                            "semantic_text": results[key]["semantic_text"],
                            "status": "semantics_labeled",
                        }
                    )

    if results:
        mutate_scenes(config, update_scenes)

    if failures:
        details = ", ".join(f"scene={key[0]:04d} chunk={key[1]:04d}" for key, _exc in failures[:20])
        if len(failures) > 20:
            details += f", ... +{len(failures) - 20} more"
        raise RuntimeError(f"VLM failed for {len(failures)} chunk(s) of video_id={video_id}: {details}")

    def update_video(videos_data: dict[str, Any]) -> None:
        video = find_video(videos_data, video_id)
        video["status"] = Status.SEMANTICS_LABELED
        stage_log(video, "call_gemini", "completed", chunks=len(results))

    mutate_videos(config, update_video)
    print(f"Wrote Gemini semantics for {len(results)} chunk(s) for video_id={video_id}")
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.video_id:
        prompt_file = args.prompt_file or path_from_config(config, "prompt_file", ROOT / "input" / "prompt.txt")
        prompt_without_hint_file = args.prompt_without_hint_file or path_from_config(
            config,
            "prompt_without_hint_file",
            ROOT / "input" / "prompt_without_hint.txt",
        )
        return process_video(config, args.video_id, prompt_file.expanduser(), prompt_without_hint_file.expanduser())
    if args.prompt_file is None or args.clip is None or args.text_output is None:
        print("--prompt-file, --clip, and --text-output are required without --video-id", file=sys.stderr)
        return 2
    prompt = args.prompt_file.read_text(encoding="utf-8")
    response = get_semantic_labeler(config).label(
        SemanticLabelRequest(config=config, prompt=prompt, clip=args.clip)
    )
    write_text(args.text_output, response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
