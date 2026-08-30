#!/usr/bin/env python3
"""Shared helpers for the staged video-to-pose pipeline."""

from __future__ import annotations

import contextlib
import base64
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_CONFIG = Path(
    os.environ.get("SPATIAL_SEMANTIC_PIPELINE_CONFIG", PROJECT_CONFIG)
).expanduser().resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = parse_scalar(raw_value)
        parent[key.strip()] = value
        if isinstance(value, dict):
            stack.append((indent, value))
    return root


def load_yaml(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def pipeline_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_yaml()
    return config.get("new_pipeline", config)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return pipeline_config(load_yaml(path))


def path_from_config(config: dict[str, Any], key: str, default: Path) -> Path:
    value = config.get(key) or config.get("paths", {}).get(key)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def metadata_root(config: dict[str, Any]) -> Path:
    """Mutable pipeline state and intermediate metadata."""
    return path_from_config(config, "metadata_root", ROOT / "metadata")


def output_root(config: dict[str, Any]) -> Path:
    """Final, self-contained dataset-style output tree."""
    return path_from_config(config, "output_root", ROOT / "output")


def resolve_path(config: dict[str, Any], key: str) -> Path:
    defaults = {
        "videos_root": cache_root(config) / "chunks",
        "video_cache_dir": cache_root(config) / "videos",
        "vggt_results_root": metadata_root(config),
        "semantics_root": cache_root(config) / "semantics",
        "hints_root": cache_root(config) / "hints",
    }
    if key not in defaults:
        raise KeyError(f"Unknown configured path: {key}")
    return path_from_config(config, key, defaults[key])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp = Path(tmp_name)
    try:
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        file_obj = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with file_obj as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def pipeline_tmp_dir(config: dict[str, Any] | None = None) -> Path:
    path = cache_root(config or {}) / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pipeline_env(config: dict[str, Any] | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    config = config or {}
    cache_dir = cache_root(config)
    tmp_dir = str(pipeline_tmp_dir(config))
    for key in ("TMPDIR", "TMP", "TEMP"):
        env[key] = tmp_dir
    env.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    env.setdefault("HF_HUB_CACHE", str(cache_dir / "hf_home" / "hub"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def activate_pipeline_environment(config: dict[str, Any] | None = None) -> None:
    config = config or {}
    os.environ["PYTHONNOUSERSITE"] = "1"
    cache_dir = cache_root(config)
    tmp_dir = str(pipeline_tmp_dir(config))
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = tmp_dir
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hf_home" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


@contextlib.contextmanager
def pipeline_lock(config: dict[str, Any]):
    lock_dir = cache_root(config)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def mutate_videos(config: dict[str, Any], mutator) -> dict[str, Any]:
    with pipeline_lock(config):
        data = load_videos(config)
        mutator(data)
        save_videos(config, data)
        return data


def mutate_scenes(config: dict[str, Any], mutator) -> dict[str, Any]:
    with pipeline_lock(config):
        data = load_scenes(config)
        mutator(data)
        save_scenes(config, data)
        return data


def read_sources(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")
    sources: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = normalize_source_url(raw_line)
            if not line or line.startswith("#") or line in seen:
                continue
            sources.append(line)
            seen.add(line)
    return sources


def normalize_source_url(source: str) -> str:
    source = source.strip()
    if not source:
        return ""
    parsed = urlparse(source)
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"youtube.com", "m.youtube.com"} and parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return source.rstrip("/")


def slug(value: str | None, fallback: str = "unknown") -> str:
    value = (value or fallback).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._")
    return value or fallback


def run(cmd: list[str], *, cwd: Path | None = None, config: dict[str, Any] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=pipeline_env(config))


def capture(cmd: list[str], *, cwd: Path | None = None, config: dict[str, Any] | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
        env=pipeline_env(config),
    )
    return result.stdout


def extract_clip(
    source: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    fps: float,
    height: int,
    crf: int = 30,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if fps > 0:
        fps_text = str(int(fps)) if float(fps).is_integer() else str(fps)
        filters.append(f"fps=fps={fps_text}:start_time=0")
    if height > 0:
        filters.append(f"scale=-2:{int(height)}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start_sec):.6f}",
        "-t",
        f"{float(duration_sec):.6f}",
        "-i",
        str(source),
        "-an",
    ]
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
    cmd.extend(["-c:v", "libopenh264", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", str(output_path)])
    run(cmd)


VIDEO_MISSING_PATTERNS = (
    "please provide the video",
    "please upload the video",
    "upload or link the",
    "provide the video file",
    "provide the visual",
    "once the video is provided",
    "once you upload",
    "need the visual data",
)


def vlm_response_missing_video(text: str) -> bool:
    lowered = text.strip().lower()
    return any(pattern in lowered for pattern in VIDEO_MISSING_PATTERNS)


def vlm_response_has_inner_monologue(text: str) -> bool:
    return bool(re.search(r"\[\d{2}\]", text)) and all(
        label in text for label in ("Observation:", "Latent Thought:", "Action:")
    )


def extract_vlm_frame_data_urls(config: dict[str, Any], clip: Path) -> list[str]:
    frame_limit = max(int(config.get("vlm_frame_limit", 60)), 1)
    jpeg_quality = max(int(config.get("vlm_frame_jpeg_quality", 3)), 2)
    with tempfile.TemporaryDirectory(prefix="pipeline_vlm_frames_") as tmp:
        tmp_dir = Path(tmp)
        pattern = tmp_dir / "frame_%04d.jpg"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip),
                "-frames:v",
                str(frame_limit),
                "-q:v",
                str(jpeg_quality),
                str(pattern),
            ]
        )
        urls: list[str] = []
        for frame in sorted(tmp_dir.glob("frame_*.jpg")):
            frame_b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
            urls.append(f"data:image/jpeg;base64,{frame_b64}")
        return urls


def vlm_input_mode(config: dict[str, Any]) -> str:
    input_mode = str(config.get("vlm_input_mode", "frames")).strip().lower()
    if input_mode not in {"frames", "video"}:
        raise ValueError(f"Unsupported vlm_input_mode={input_mode!r}; expected 'frames' or 'video'")
    return input_mode


def vlm_request_content(config: dict[str, Any], prompt: str, clip: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    frame_urls = extract_vlm_frame_data_urls(config, clip)
    if not frame_urls:
        raise ValueError(f"No frames extracted for VLM clip: {clip}")
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in frame_urls)
    return content


def vlm_api_key(config: dict[str, Any]) -> str:
    env_name = str(config.get("api_key_env") or config.get("vlm_api_key_env") or "")
    value = os.environ.get(env_name) if env_name else None
    value = value or config.get("api_key") or config.get("vlm_api_key")
    if not value:
        hint = f" Set ${env_name}" if env_name else ""
        raise ValueError(f"api_key is required for VLM semantic labeling.{hint}")
    return str(value)


def build_gemini_video_payload(config: dict[str, Any], prompt: str, clip: Path) -> dict[str, Any]:
    clip_b64 = base64.b64encode(clip.read_bytes()).decode("ascii")
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "video/mp4", "data": clip_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": float(config.get("temperature", 0.2)),
            "maxOutputTokens": int(config.get("max_tokens", 4096)),
        },
    }


def extract_gemini_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str) and text.strip():
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError(f"Could not extract text from Gemini response: {json.dumps(payload, ensure_ascii=False)[:1000]}")


def normalize_gemini_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    if not base_url.endswith("/v1") and not base_url.endswith("/v1beta"):
        base_url = f"{base_url}/v1"
    return base_url


def gemini_api_root(base_url: str) -> str:
    base_url = normalize_gemini_base_url(base_url).rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    if base_url.endswith("/v1beta"):
        return base_url[:-7]
    return base_url


def call_gemini_video_api(config: dict[str, Any], prompt: str, clip: Path) -> tuple[str, dict[str, Any]]:
    model = str(config.get("model") or config.get("vlm_model") or "gemini-3-pro-preview")
    base_url = str(config.get("base_url") or "https://generativelanguage.googleapis.com/v1beta")
    api_version = str(config.get("api_version") or "v1beta").strip("/")
    url = f"{gemini_api_root(base_url)}/{api_version}/models/{model}:generateContent"
    body = json.dumps(build_gemini_video_payload(config, prompt, clip)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {vlm_api_key(config)}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    timeout = float(config.get("vlm_timeout_sec") or config.get("api_timeout_sec") or 300)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {error_body[:2000]}") from exc
    return extract_gemini_text(payload), payload


def call_openai_compatible_frames_api(config: dict[str, Any], prompt: str, clip: Path) -> tuple[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from exc

    model = str(config.get("model") or config.get("vlm_model") or "gemini-3-pro-preview")
    timeout_sec = float(config.get("vlm_timeout_sec", 180))
    client_kwargs: dict[str, Any] = {"api_key": vlm_api_key(config), "timeout": timeout_sec}
    if config.get("base_url"):
        client_kwargs["base_url"] = str(config.get("base_url"))
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": vlm_request_content(config, prompt, clip)}],
        max_tokens=int(config.get("max_tokens", 4096)),
    )
    return response.choices[0].message.content or "", response


def retry_call_vlm_api(config: dict[str, Any], prompt: str, clip: Path) -> tuple[str, Any]:
    max_attempts = max(int(config.get("vlm_retry_attempts", 3)), 1)
    retry_wait_sec = float(config.get("vlm_retry_wait_sec", 5))
    input_mode = vlm_input_mode(config)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if input_mode == "video":
                text, response = call_gemini_video_api(config, prompt, clip)
            else:
                text, response = call_openai_compatible_frames_api(config, prompt, clip)
            if vlm_response_missing_video(text) or not vlm_response_has_inner_monologue(text):
                raise RuntimeError(f"VLM returned invalid semantic response: {text[:240]!r}")
            return text, response
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(retry_wait_sec)
    raise RuntimeError(f"VLM call failed after {max_attempts} attempts: {last_error}") from last_error


def ffprobe_duration(path: Path) -> float:
    out = capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    ).strip()
    return float(out)


def ffprobe_video_stream(path: Path) -> dict[str, Any]:
    out = capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(out)
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def parse_frame_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    return float(value)


def ffprobe_frame_count(path: Path) -> int:
    out = capture(
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
            str(path),
        ]
    ).strip()
    return int(out) if out and out != "N/A" else 0


def ffprobe_has_audio(path: Path) -> bool:
    out = capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(out)
    return bool(payload.get("streams"))


def consolidation_manifest_path(config: dict[str, Any]) -> Path:
    return metadata_root(config) / "consolidation.json"


def videos_manifest_path(config: dict[str, Any]) -> Path:
    return consolidation_manifest_path(config)


def pipeline_state_path(config: dict[str, Any]) -> Path:
    return metadata_root(config) / "pipeline_state.json"


def scenes_manifest_path(config: dict[str, Any]) -> Path:
    return consolidation_manifest_path(config)


def pipeline_scenes_path(config: dict[str, Any]) -> Path:
    return metadata_root(config) / "pipeline_scenes.json"


def sources_path(config: dict[str, Any]) -> Path:
    return path_from_config(config, "sources_file", ROOT / "input" / "sources.txt")


def cache_root(config: dict[str, Any]) -> Path:
    return path_from_config(config, "cache_root", ROOT / "cache")


def optional_positive_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "original"}:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"Expected positive integer or 'original', got {value!r}")
    return result


def optional_positive_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "original"}:
        return None
    result = float(value)
    if result <= 0:
        raise ValueError(f"Expected positive number or 'original', got {value!r}")
    return result


def media_value_label(value: int | float | None, suffix: str) -> str:
    if value is None:
        return "original"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{suffix}"


def configured_full_chunk_duration_sec(config: dict[str, Any]) -> float:
    return float(config.get("full_chunk_duration_sec", config.get("chunk_duration_sec", 60.0)))


def chunk_duration_sec(chunk: dict[str, Any]) -> float:
    for start_key, end_key in (
        ("logical_start_sec", "logical_end_sec"),
        ("chunk_start_sec", "chunk_end_sec"),
        ("input_start_sec", "input_end_sec"),
    ):
        start = chunk.get(start_key)
        end = chunk.get(end_key)
        if start is None or end is None:
            continue
        try:
            return float(end) - float(start)
        except (TypeError, ValueError):
            continue
    logical_frame_count = chunk.get("logical_frame_count")
    if logical_frame_count is not None:
        try:
            return float(logical_frame_count)
        except (TypeError, ValueError):
            pass
    return 0.0


def chunk_is_good_full_minute(config: dict[str, Any], chunk: dict[str, Any]) -> bool:
    if chunk.get("trajectory_judge") != "good":
        return False
    return chunk_is_full_minute(config, chunk)


def chunk_is_full_minute(config: dict[str, Any], chunk: dict[str, Any]) -> bool:
    required_sec = configured_full_chunk_duration_sec(config)
    return chunk_duration_sec(chunk) + 1e-6 >= required_sec


def canonical_chunk_video_path(config: dict[str, Any], video: dict[str, Any], scene: dict[str, Any], chunk: dict[str, Any]) -> Path:
    channel = slug(video.get("hf_channel") or video.get("channel") or "unknown")
    return (
        resolve_path(config, "videos_root")
        / channel
        / slug(str(video["video_id"]))
        / f"scene_{int(scene['scene_index']):04d}"
        / f"chunk_{int(chunk['chunk_index']):04d}.mp4"
    )


def resolve_chunk_video_path(
    config: dict[str, Any],
    video: dict[str, Any],
    scene: dict[str, Any],
    chunk: dict[str, Any],
) -> Path | None:
    canonical = canonical_chunk_video_path(config, video, scene, chunk)
    value = chunk.get("chunk_video_file")
    if value:
        candidate = ROOT / str(value)
        if candidate.is_file() and candidate.name == canonical.name:
            return candidate
    if canonical.is_file():
        return canonical
    if value:
        candidate = ROOT / str(value)
        if candidate.is_file():
            return candidate
    return None


def manifest_has_records(path: Path, key: str) -> bool:
    return bool(read_json(path, {key: []}).get(key))


def channel_manifest_path(config: dict[str, Any], kind: str, channel: str) -> Path:
    channel_slug = slug(channel)
    metadata_root = consolidation_manifest_path(config).parent
    if kind == "videos":
        return metadata_root / "info" / channel_slug / f"info_{channel_slug}.json"
    return metadata_root / "scenes" / channel_slug / f"scenes_{channel_slug}.json"


def channel_manifest_repo_path(kind: str, channel: str) -> str:
    channel_slug = slug(channel)
    if kind == "videos":
        return f"info/{channel_slug}/info_{channel_slug}.json"
    return f"scenes/{channel_slug}/scenes_{channel_slug}.json"


def index_manifest_repo_path(kind: str) -> str:
    return "consolidation.json"


def legacy_channel_manifest_path(config: dict[str, Any], kind: str, channel: str) -> Path:
    metadata_root = consolidation_manifest_path(config).parent
    filename = "videos.json" if kind == "videos" else "scenes.json"
    root = "videos" if kind == "videos" else "scenes"
    return metadata_root / root / slug(channel) / filename


def legacy_index_manifest_path(config: dict[str, Any], kind: str) -> Path:
    metadata_root = consolidation_manifest_path(config).parent
    root = "videos" if kind == "videos" else "scenes"
    return metadata_root / root / "index.json"


def clean_manifest_repo_path(config: dict[str, Any], path: Path, kind: str) -> str:
    if path == consolidation_manifest_path(config):
        return index_manifest_repo_path(kind)
    channel = path.parent.name
    return channel_manifest_repo_path(kind, channel)


def clean_manifest_files(config: dict[str, Any]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    consolidation_path = consolidation_manifest_path(config)
    if consolidation_path.exists():
        files.append((consolidation_path, index_manifest_repo_path("metadata")))
    judge_results_path = consolidation_path.parent / "judge_results.json"
    if judge_results_path.exists():
        files.append((judge_results_path, "judge_results.json"))
    hint_results_path = consolidation_path.parent / "hint_results.json"
    if hint_results_path.exists():
        files.append((hint_results_path, "hint_results.json"))
    index = read_json(consolidation_path, {"channels": []})
    for kind, manifest_key in (("videos", "videos_manifest"), ("scenes", "scenes_manifest")):
        for item in index.get("channels", []):
            local_path = channel_manifest_path(config, kind, str(item.get("channel", "")))
            if local_path.exists():
                files.append((local_path, str(item.get(manifest_key) or channel_manifest_repo_path(kind, str(item.get("channel", ""))))))
    return files


def load_clean_videos(config: dict[str, Any]) -> list[dict[str, Any]]:
    index = read_json(consolidation_manifest_path(config), {"channels": []})
    if not index.get("channels"):
        index = read_json(legacy_index_manifest_path(config, "videos"), {"channels": []})
    if index.get("channels"):
        videos: list[dict[str, Any]] = []
        for item in index.get("channels", []):
            channel = str(item.get("channel", ""))
            manifest = read_json(channel_manifest_path(config, "videos", channel), {})
            if "videos" not in manifest:
                manifest = read_json(legacy_channel_manifest_path(config, "videos", channel), {"videos": []})
            videos.extend(manifest.get("videos", []))
        return videos
    return read_json(legacy_videos_manifest_path(config), {"videos": []}).get("videos", [])


def load_videos(config: dict[str, Any]) -> dict[str, Any]:
    data = read_json(pipeline_state_path(config), {"schema_version": 1, "videos": []})
    data.setdefault("schema_version", 1)
    data.setdefault("videos", [])
    return data


def save_videos(config: dict[str, Any], data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    write_json(pipeline_state_path(config), data)


def load_scenes(config: dict[str, Any]) -> dict[str, Any]:
    data = read_json(pipeline_scenes_path(config), {"schema_version": 1, "scenes": []})
    data.setdefault("schema_version", 1)
    data.setdefault("scenes", [])
    return data


def save_scenes(config: dict[str, Any], data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    write_json(pipeline_scenes_path(config), data)


def source_video_id(source: str) -> str | None:
    source = source.strip()
    if not source:
        return None
    parsed = urlparse(source)
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"youtube.com", "m.youtube.com"} and parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        return video_id or None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id or None
    return None


def find_video(videos: dict[str, Any], video_id: str) -> dict[str, Any]:
    for video in videos.get("videos", []):
        if str(video.get("video_id", "")) == video_id:
            return video
    raise KeyError(f"Video id not found: {video_id}")


def find_video_by_source(videos: dict[str, Any], source: str) -> dict[str, Any] | None:
    normalized = normalize_source_url(source)
    parsed_video_id = source_video_id(source)
    for video in videos.get("videos", []):
        if parsed_video_id and str(video.get("video_id", "")) == parsed_video_id:
            return video
        if normalize_source_url(str(video.get("source_url", ""))) == normalized:
            return video
        if normalize_source_url(str(video.get("webpage_url", ""))) == normalized:
            return video
    return None


def find_next_source(config: dict[str, Any], videos: dict[str, Any]) -> str | None:
    done_urls = {
        normalize_source_url(str(value))
        for video in videos.get("videos", [])
        for value in (video.get("source_url"), video.get("webpage_url"))
        if value
    }
    done_ids = {str(video.get("video_id")) for video in videos.get("videos", []) if video.get("video_id")}
    for source in read_sources(sources_path(config)):
        parsed_video_id = source_video_id(source)
        if parsed_video_id and parsed_video_id in done_ids:
            continue
        if normalize_source_url(source) not in done_urls:
            return source
    return None


def source_is_registered(videos: dict[str, Any], source: str) -> bool:
    return find_video_by_source(videos, source) is not None


def stable_video_id(source: str) -> str:
    return source_video_id(source) or slug(source, fallback="video")


def video_cache_dir(config: dict[str, Any], video_id: str) -> Path:
    return cache_root(config) / "videos" / slug(video_id)


def video_cache_dirs(config: dict[str, Any], video_id: str, video: dict[str, Any] | None = None) -> list[Path]:
    dirs: list[Path] = []

    def add(path: Path) -> None:
        if path not in dirs:
            dirs.append(path)

    add(video_cache_dir(config, video_id))
    if video is not None:
        requested_video_id = video.get("requested_video_id")
        if requested_video_id:
            add(video_cache_dir(config, str(requested_video_id)))
        for key in ("cache_video_path", "cache_info_path"):
            value = video.get(key)
            if value:
                add((ROOT / str(value)).parent)
    return dirs


def processed_video_path(config: dict[str, Any], video_id: str) -> Path:
    height = optional_positive_int(config.get("sampled_height", config.get("vggt_input_height")), 480)
    fps = optional_positive_float(config.get("sampled_fps", config.get("vggt_input_fps")), 1.0)
    return (
        cache_root(config)
        / "processed"
        / f"{slug(video_id)}_{media_value_label(height, 'p')}_{media_value_label(fps, 'fps')}_noaudio.mp4"
    )


def hf_upload_video_path(config: dict[str, Any], video_id: str) -> Path:
    return cache_root(config) / "upload" / f"{slug(video_id)}_legacy_noaudio.mp4"


def video_view_path(
    config: dict[str, Any],
    video_id: str,
    height: int | None,
    fps: float | None,
    *,
    no_audio: bool = True,
) -> Path:
    audio_label = "noaudio" if no_audio else "audio"
    return (
        cache_root(config)
        / "views"
        / f"{slug(video_id)}_{media_value_label(height, 'p')}_{media_value_label(fps, 'fps')}_{audio_label}.mp4"
    )


def video_view_frame_map_path(view_path: Path) -> Path:
    return view_path.with_suffix(view_path.suffix + ".frame_map.json")


def ffprobe_frame_timestamps(path: Path) -> list[float]:
    out = capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time,pkt_pts_time",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(out)
    timestamps: list[float] = []
    for index, frame in enumerate(payload.get("frames") or []):
        value = (
            frame.get("best_effort_timestamp_time")
            or frame.get("pts_time")
            or frame.get("pkt_pts_time")
        )
        if value is None or value == "N/A":
            raise ValueError(f"Missing source timestamp for frame {index} in {path}")
        timestamps.append(float(value))
    return timestamps


def integer_second_sampling(fps: float | None) -> bool:
    return fps is not None and abs(float(fps) - 1.0) < 1e-9


def video_view_sampling_method(fps: float | None) -> str:
    if integer_second_sampling(fps):
        return "source_integer_second_first_frame_v1"
    return "source_frame_step_keyframe_v1" if fps is not None else "copy_v1"


def first_source_frames_by_integer_second(source_timestamps: list[float]) -> list[int]:
    source_frames: list[int] = []
    seen_seconds: set[int] = set()
    for source_frame, timestamp in enumerate(source_timestamps):
        second = math.floor(timestamp)
        if second in seen_seconds:
            continue
        seen_seconds.add(second)
        source_frames.append(source_frame)
    return source_frames


def ensure_video_view_frame_map(
    config: dict[str, Any],
    view_path: Path,
    source_video: Path,
    *,
    fps: float | None,
) -> Path:
    map_path = video_view_frame_map_path(view_path)
    sampling_method = video_view_sampling_method(fps)
    payload = read_json(map_path, {}) if map_path.exists() else {}
    if (
        isinstance(payload, dict)
        and payload.get("source_video") == str(source_video)
        and payload.get("view_video") == str(view_path)
        and payload.get("sampling_method") == sampling_method
        and isinstance(payload.get("frames"), list)
        and payload["frames"]
    ):
        return map_path

    source_timestamps = ffprobe_frame_timestamps(source_video)
    source_stream = ffprobe_video_stream(source_video)
    source_fps = parse_frame_rate(source_stream.get("avg_frame_rate")) or parse_frame_rate(
        source_stream.get("r_frame_rate")
    )
    if integer_second_sampling(fps):
        frame_step = None
        source_frames = first_source_frames_by_integer_second(source_timestamps)
    elif fps is not None:
        frame_step = round(source_fps / fps) if source_fps > fps > 0 else 0
        if frame_step <= 0 or abs(source_fps / frame_step - fps) >= 1e-3:
            raise ValueError(
                f"Strict timestamp mapping requires source-frame-step sampling: "
                f"source_fps={source_fps}, target_fps={fps}"
            )
        source_frames = list(range(0, len(source_timestamps), frame_step))
    else:
        frame_step = None
        source_frames = list(range(len(source_timestamps)))

    frames = [
        {
            "sampled_frame": sampled_frame,
            "source_frame": source_frame,
            "source_timestamp": source_timestamps[source_frame],
        }
        for sampled_frame, source_frame in enumerate(source_frames)
    ]
    write_json(
        map_path,
        {
            "source_video": str(source_video),
            "view_video": str(view_path),
            "fps": fps,
            "source_fps": source_fps,
            "frame_step": frame_step,
            "sampling_method": sampling_method,
            "timestamp_source": "ffprobe_best_effort_timestamp_time",
            "frames": frames,
        },
    )
    return map_path


def ensure_video_view(
    config: dict[str, Any],
    video_id: str,
    source_video: Path,
    *,
    height: int | None,
    fps: float | None,
    no_audio: bool = True,
) -> Path:
    dst = video_view_path(config, video_id, height, fps, no_audio=no_audio)
    metadata_path = dst.with_suffix(dst.suffix + ".json")
    sampling_method = video_view_sampling_method(fps)
    metadata = read_json(metadata_path, {}) if metadata_path.exists() else {}
    if (
        dst.exists()
        and dst.stat().st_size > 0
        and isinstance(metadata, dict)
        and metadata.get("sampling_method") == sampling_method
    ):
        ensure_video_view_frame_map(config, dst, source_video, fps=fps)
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    tmp.unlink(missing_ok=True)

    filters = []
    if fps is not None:
        fps_text = str(int(fps)) if fps.is_integer() else str(fps)
        source_stream = ffprobe_video_stream(source_video)
        source_fps = parse_frame_rate(source_stream.get("avg_frame_rate")) or parse_frame_rate(
            source_stream.get("r_frame_rate")
        )
        if integer_second_sampling(fps):
            filters.append("select=isnan(prev_selected_t)+not(eq(floor(t)\\,floor(prev_selected_t)))")
        else:
            frame_step = round(source_fps / fps) if source_fps > fps > 0 else 0
            if frame_step > 0 and abs(source_fps / frame_step - fps) < 1e-3:
                filters.append(f"select=not(mod(n\\,{frame_step}))")
            else:
                filters.append(f"fps=fps={fps_text}:start_time=0")
        filters.append(f"setpts=N/({fps_text}*TB)")
    if height is not None:
        filters.append(f"scale=-2:{height}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
    ]
    if no_audio:
        cmd.append("-an")
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
        cmd.extend(
            [
                "-c:v",
                "libopenh264",
                "-g",
                "1",
                "-keyint_min",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        )
    else:
        cmd.extend(["-c:v", "copy"])
    if not no_audio:
        cmd.extend(["-c:a", "copy"])
    cmd.extend(["-movflags", "+faststart", str(tmp)])
    run(cmd)
    tmp.replace(dst)
    ensure_video_view_frame_map(config, dst, source_video, fps=fps)
    write_json(
        metadata_path,
        {
            "source_video": str(source_video),
            "height": height,
            "fps": fps,
            "no_audio": no_audio,
            "sampling_method": sampling_method,
        },
    )
    return dst


def remove_video_views(config: dict[str, Any], video_id: str) -> None:
    views_dir = cache_root(config) / "views"
    if not views_dir.exists():
        return
    for path in views_dir.glob(f"{slug(video_id)}_*.mp4*"):
        path.unlink(missing_ok=True)


def pose_cache_dir(config: dict[str, Any], video_id: str) -> Path:
    return cache_root(config) / "poses" / slug(video_id)


def scene_boundaries_for_video(scenes: list[dict[str, Any]], video_id: str) -> list[float]:
    video_scenes = sorted(
        [scene for scene in scenes if str(scene.get("video_id", "")) == video_id],
        key=lambda scene: int(scene.get("scene_index", 0)),
    )
    boundaries: list[float] = []
    for prev_scene, next_scene in zip(video_scenes, video_scenes[1:]):
        prev_end_timestamp = prev_scene.get("source_end_timestamp", prev_scene.get("source_end_sec"))
        next_start_timestamp = next_scene.get("source_start_timestamp", next_scene.get("source_start_sec"))
        if prev_end_timestamp is not None and next_start_timestamp is not None:
            cut_second = int(prev_end_timestamp)
            if int(next_start_timestamp) == cut_second + 1:
                boundaries.append(float(cut_second))
    return boundaries

def legacy_videos_manifest_path(config: dict[str, Any]) -> Path:
    configured = videos_manifest_path(config)
    legacy = metadata_root(config) / "videos.json"
    return legacy if configured != legacy else configured


def legacy_scenes_manifest_path(config: dict[str, Any]) -> Path:
    configured = scenes_manifest_path(config)
    legacy = metadata_root(config) / "scenes.json"
    return legacy if configured != legacy else configured


def channel_from_repo_path(path: Any, prefix: str) -> str:
    parts = str(path or "").split("/")
    if len(parts) >= 2 and parts[0] == prefix:
        return slug(parts[1])
    return "unknown"


def video_channel(video: dict[str, Any]) -> str:
    return slug(
        video.get("channel")
        or video.get("hf_channel")
        or channel_from_repo_path(video.get("hf_video_path") or video.get("hf_raw_video"), "videos")
    )


def scene_channel(scene: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> str:
    video_id = str(scene.get("video_id", ""))
    if video_id in videos_by_id:
        return video_channel(videos_by_id[video_id])
    pose_files = scene.get("pose_files") or scene.get("hf_pose_files") or []
    if pose_files:
        return channel_from_repo_path(pose_files[0], "scenes")
    return "unknown"


def read_existing_channel_rows(config: dict[str, Any], kind: str, key: str) -> list[dict[str, Any]]:
    index_path = videos_manifest_path(config) if kind == "videos" else scenes_manifest_path(config)
    rows: list[dict[str, Any]] = []
    index = read_json(index_path, {"channels": []})
    if not index.get("channels"):
        index = read_json(legacy_index_manifest_path(config, kind), {"channels": []})
    for item in index.get("channels", []):
        channel = str(item.get("channel", ""))
        manifest = read_json(channel_manifest_path(config, kind, channel), {})
        if key not in manifest:
            manifest = read_json(legacy_channel_manifest_path(config, kind, channel), {key: []})
        for row in manifest.get(key, []):
            row = dict(row)
            row.setdefault("channel", channel)
            rows.append(row)
    legacy_path = legacy_videos_manifest_path(config) if kind == "videos" else legacy_scenes_manifest_path(config)
    legacy = read_json(legacy_path, {key: []})
    rows.extend(legacy.get(key, []))
    return rows


def build_clean_videos_by_channel(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    clean_by_id = {
        str(video.get("video_id", "")): video
        for video in read_existing_channel_rows(config, "videos", "videos")
        if video.get("video_id")
    }
    state = load_videos(config)
    scenes = load_scenes(config).get("scenes", [])
    for video in state.get("videos", []):
        video_id = str(video.get("video_id", ""))
        info_path = video.get("output_info_path") or video.get("hf_info")
        if not video_id or not info_path:
            continue
        valid_scene_count = len([scene for scene in scenes if str(scene.get("video_id", "")) == video_id])
        boundaries = video.get("scene_boundaries") or scene_boundaries_for_video(scenes, video_id)
        total_scene_count = video.get("total_scene_count")
        if total_scene_count is None:
            total_scene_count = len(boundaries) + 1 if boundaries else valid_scene_count
        clean_video = {
            "video_id": video_id,
            "source_url": video.get("source_url"),
            "width": video.get("output_width", video.get("hf_upload_width", video.get("width"))),
            "height": video.get("output_height", video.get("hf_upload_height", video.get("height"))),
            "duration_sec": video.get("output_duration_sec", video.get("hf_upload_duration_sec", video.get("duration_sec"))),
            "frame_count": video.get("output_frame_count", video.get("hf_upload_frame_count", video.get("frame_count"))),
            "fps": video.get("output_fps", video.get("hf_upload_fps", video.get("fps"))),
            "info_path": info_path,
            "total_scene_count": total_scene_count,
            "valid_scene_count": valid_scene_count,
            "scene_boundaries": boundaries,
            "channel": video_channel(video),
        }
        clean_by_id[video_id] = clean_video
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for video in clean_by_id.values():
        channel = video_channel(video)
        video = dict(video)
        video["channel"] = channel
        by_channel.setdefault(channel, []).append(video)
    for channel, videos in by_channel.items():
        by_channel[channel] = sorted(videos, key=lambda item: str(item.get("video_id", "")))
    return dict(sorted(by_channel.items()))


def build_clean_videos_manifest(config: dict[str, Any]) -> dict[str, Any]:
    by_channel = build_clean_videos_by_channel(config)
    channels = []
    for channel, videos in by_channel.items():
        channels.append(
            {
                "channel": channel,
                "videos_manifest": channel_manifest_repo_path("videos", channel),
                "scenes_manifest": channel_manifest_repo_path("scenes", channel),
                "video_count": len(videos),
            }
        )
    return {"channels": channels}


def build_clean_scenes_by_channel(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    clean_by_key = {
        (str(scene.get("video_id", "")), int(scene.get("scene_index", 0))): scene
        for scene in read_existing_channel_rows(config, "scenes", "scenes")
        if scene.get("video_id") is not None
    }
    state = load_videos(config)
    videos_by_id = {str(video.get("video_id", "")): video for video in state.get("videos", [])}
    for channel, videos in build_clean_videos_by_channel(config).items():
        for video in videos:
            videos_by_id.setdefault(str(video.get("video_id", "")), video)
    for scene in sorted(
        load_scenes(config).get("scenes", []),
        key=lambda item: (str(item.get("video_id", "")), int(item.get("scene_index", 0))),
    ):
        video_id = str(scene.get("video_id", ""))
        video = videos_by_id.get(video_id, {})
        if not (video.get("output_info_path") or video.get("info_path") or video.get("hf_info") or video.get("hf_info_path")):
            continue
        pose_files = []
        for chunk in sorted(scene.get("pose_chunks", []), key=lambda item: int(item.get("chunk_index", 0))):
            pose_file = chunk.get("output_pose_file") or chunk.get("hf_pose_file")
            if pose_file:
                pose_files.append(pose_file)
        clean_by_key[(video_id, int(scene.get("scene_index", 0)))] = {
            "video_id": video_id,
            "scene_index": int(scene.get("scene_index", 0)),
            "source_start_timestamp": scene.get("source_start_timestamp", scene.get("source_start_sec")),
            "source_end_timestamp": scene.get("source_end_timestamp", scene.get("source_end_sec")),
            "source_start_frame": scene.get("source_start_frame"),
            "source_end_frame": scene.get("source_end_frame"),
            "source_frame_count": scene.get("source_frame_count", scene.get("duration_frames")),
            "scene_frame_count": scene.get("scene_frame_count"),
            "chunk_count": len(pose_files),
            "pose_files": pose_files,
            "channel": scene_channel(scene, videos_by_id),
        }
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for scene in clean_by_key.values():
        channel = scene_channel(scene, videos_by_id)
        scene = dict(scene)
        scene["channel"] = channel
        by_channel.setdefault(channel, []).append(scene)
    for channel, scenes in by_channel.items():
        by_channel[channel] = sorted(
            scenes,
            key=lambda item: (str(item.get("video_id", "")), int(item.get("scene_index", 0))),
        )
    return dict(sorted(by_channel.items()))


def build_clean_scenes_manifest(config: dict[str, Any]) -> dict[str, Any]:
    by_channel = build_clean_scenes_by_channel(config)
    video_index = build_clean_videos_manifest(config)
    video_counts = {item["channel"]: item.get("video_count", 0) for item in video_index.get("channels", [])}
    channels_with_metadata = sorted(set(video_counts) | set(by_channel))
    channels = []
    for channel in channels_with_metadata:
        scenes = by_channel.get(channel, [])
        channels.append(
            {
                "channel": channel,
                "videos_manifest": channel_manifest_repo_path("videos", channel),
                "scenes_manifest": channel_manifest_repo_path("scenes", channel),
                "video_count": video_counts.get(channel, 0),
                "scene_count": len(scenes),
            }
        )
    return {"channels": channels}


def strip_channel_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row.pop("channel", None)
        stripped.append(row)
    return stripped


def write_clean_scenes_manifest(config: dict[str, Any]) -> dict[str, Any]:
    by_channel = build_clean_scenes_by_channel(config)
    for channel, scenes in by_channel.items():
        write_json(
            channel_manifest_path(config, "scenes", channel),
            {"channel": channel, "scenes": strip_channel_from_rows(scenes)},
        )
    manifest = build_clean_scenes_manifest(config)
    write_json(consolidation_manifest_path(config), manifest)
    return manifest


def write_clean_videos_manifest(config: dict[str, Any]) -> dict[str, Any]:
    by_channel = build_clean_videos_by_channel(config)
    for channel, videos in by_channel.items():
        write_json(
            channel_manifest_path(config, "videos", channel),
            {"channel": channel, "videos": strip_channel_from_rows(videos)},
        )
    manifest = build_clean_videos_manifest(config)
    return manifest


def remove_video_from_local_state(config: dict[str, Any], video_id: str) -> None:
    state = load_videos(config)
    state["videos"] = [
        item for item in state.get("videos", []) if str(item.get("video_id", "")) != video_id
    ]
    save_videos(config, state)

    scene_state = load_scenes(config)
    scene_state["scenes"] = [
        item for item in scene_state.get("scenes", []) if str(item.get("video_id", "")) != video_id
    ]
    if not scene_state["scenes"]:
        scene_state.pop("detector", None)
    save_scenes(config, scene_state)


def cleanup_completed_video(config: dict[str, Any], video_id: str) -> None:
    video = None
    try:
        video = find_video(load_videos(config), video_id)
    except KeyError:
        pass
    for cache_dir in video_cache_dirs(config, video_id, video):
        shutil.rmtree(cache_dir, ignore_errors=True)
    processed_video_path(config, video_id).unlink(missing_ok=True)
    hf_upload_video_path(config, video_id).unlink(missing_ok=True)
    remove_video_views(config, video_id)
    shutil.rmtree(pose_cache_dir(config, video_id), ignore_errors=True)
    for key in ("videos_root", "hints_root", "semantics_root"):
        root = resolve_path(config, key)
        if root.exists():
            for path in root.rglob(slug(video_id)):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
    remove_video_from_local_state(config, video_id)


def cleanup_successful_runtime(config: dict[str, Any]) -> None:
    """Remove transient state after the entire workflow has succeeded.

    Per-video cleanup deliberately keeps the runtime roots available while
    other videos may still be running.  The workflow calls this final cleanup
    only after every selected video has finished and the output package is
    complete.  Failed or interrupted runs retain both roots for resuming.
    """
    shutil.rmtree(cache_root(config), ignore_errors=True)
    shutil.rmtree(metadata_root(config), ignore_errors=True)
    shutil.rmtree(output_root(config) / ".staging", ignore_errors=True)


# Backward-compatible name for state created by older pipeline versions.
cleanup_uploaded_video = cleanup_completed_video


def remove_cached_video_dirs(config: dict[str, Any], key: str, video_id: str) -> None:
    root = resolve_path(config, key)
    if not root.exists():
        return
    for path in root.rglob(slug(video_id)):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def remove_unreferenced_files(root: Path, referenced: set[Path]) -> None:
    if not root.exists():
        return
    referenced_resolved = {path.resolve() for path in referenced if path.exists()}
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and path.resolve() not in referenced_resolved:
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def prune_chunk_keys(chunk: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = chunk.pop(key, None)
        if key.endswith("_file") and isinstance(value, str):
            (ROOT / value).unlink(missing_ok=True)


POSE_DERIVED_CHUNK_KEYS = (
    "pose_file",
    "logical_start_frame",
    "logical_end_frame",
    "input_start_frame",
    "input_end_frame",
    "logical_start_sec",
    "logical_end_sec",
    "input_start_sec",
    "input_end_sec",
    "logical_frame_count",
    "input_frame_count",
    "theoretical_input_frame_count",
    "overlap_frames",
    "trajectory_judge",
    "trajectory_judge_reasons",
    "trajectory_hint_file",
    "trajectory_hint",
    "semantic_file",
    "semantic_text",
    "hf_pose_file",
    "hf_semantic_file",
    "hf_trajectory_hint_file",
    "hf_chunk_video_file",
    "hf_chunk_repo_id",
    "output_pose_file",
    "output_semantic_file",
    "output_chunk_video_file",
)

POST_VGGT_CHUNK_KEYS = (
    "trajectory_judge",
    "trajectory_judge_reasons",
    "trajectory_hint_file",
    "trajectory_hint",
    "semantic_file",
    "semantic_text",
    "hf_pose_file",
    "hf_semantic_file",
    "hf_trajectory_hint_file",
    "hf_chunk_video_file",
    "hf_chunk_repo_id",
    "output_pose_file",
    "output_semantic_file",
    "output_chunk_video_file",
)

SEMANTIC_UPLOAD_CHUNK_KEYS = (
    "hf_pose_file",
    "hf_semantic_file",
    "hf_trajectory_hint_file",
    "hf_chunk_video_file",
    "hf_chunk_repo_id",
    "output_pose_file",
    "output_semantic_file",
    "output_chunk_video_file",
)


def rollback_to_chunks_split_preserve_pose(config: dict[str, Any], video_id: str) -> bool:
    scenes = load_scenes(config)
    referenced_pose_files: set[Path] = set()
    referenced_chunk_files: set[Path] = set()

    def clean_scene(scene: dict[str, Any]) -> None:
        if str(scene.get("video_id", "")) != video_id:
            return
        kept_chunks = []
        for chunk in scene.get("pose_chunks", []):
            chunk = dict(chunk)
            pose_file = chunk.get("pose_file")
            pose_path = ROOT / str(pose_file) if pose_file else None
            if pose_path is not None and pose_path.is_file():
                referenced_pose_files.add(pose_path)
                chunk_video_file = chunk.get("chunk_video_file")
                chunk_video_path = ROOT / str(chunk_video_file) if chunk_video_file else None
                if chunk_video_path is not None and chunk_video_path.is_file():
                    referenced_chunk_files.add(chunk_video_path)
                for key in POST_VGGT_CHUNK_KEYS:
                    value = chunk.pop(key, None)
                    if key.endswith("_file") and isinstance(value, str):
                        (ROOT / value).unlink(missing_ok=True)
                chunk["status"] = "completed"
            else:
                for key in POSE_DERIVED_CHUNK_KEYS:
                    chunk.pop(key, None)
                chunk["status"] = "chunk_split"
            kept_chunks.append(chunk)
        scene["pose_chunks"] = kept_chunks
        scene.pop("scene_frame_count", None)

    for scene in scenes.get("scenes", []):
        clean_scene(scene)
    save_scenes(config, scenes)
    remove_unreferenced_files(pose_cache_dir(config, video_id), referenced_pose_files)
    chunks_root = resolve_path(config, "videos_root")
    if chunks_root.exists():
        for video_chunk_dir in chunks_root.rglob(slug(video_id)):
            if video_chunk_dir.is_dir():
                remove_unreferenced_files(video_chunk_dir, referenced_chunk_files)
    remove_cached_video_dirs(config, "hints_root", video_id)
    remove_cached_video_dirs(config, "semantics_root", video_id)

    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    video["status"] = "chunks_split"
    stage_log(video, "rollback_to_chunks_split", "completed")
    save_videos(config, videos)
    return True


def rollback_to_scenes_detected_preserve_pose(config: dict[str, Any], video_id: str) -> bool:
    if not rollback_to_chunks_split_preserve_pose(config, video_id):
        return False
    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    video["status"] = "scenes_detected"
    stage_log(video, "rollback_to_scenes_detected", "completed")
    save_videos(config, videos)
    return True


def rollback_to_poses_estimated(config: dict[str, Any], video_id: str) -> bool:
    scenes = load_scenes(config)
    for scene in scenes.get("scenes", []):
        if str(scene.get("video_id", "")) != video_id:
            continue
        for chunk in scene.get("pose_chunks", []):
            for key in POST_VGGT_CHUNK_KEYS:
                value = chunk.pop(key, None)
                if key.endswith("_file") and isinstance(value, str):
                    (ROOT / value).unlink(missing_ok=True)
            if chunk.get("pose_file"):
                chunk["status"] = "completed"
    save_scenes(config, scenes)
    remove_cached_video_dirs(config, "hints_root", video_id)
    remove_cached_video_dirs(config, "semantics_root", video_id)

    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    video["status"] = "poses_estimated"
    stage_log(video, "rollback_to_poses_estimated", "completed")
    save_videos(config, videos)
    return True


def rollback_to_hints_extracted_preserve_semantics(config: dict[str, Any], video_id: str) -> bool:
    scenes = load_scenes(config)
    referenced_semantic_files: set[Path] = set()
    for scene in scenes.get("scenes", []):
        if str(scene.get("video_id", "")) != video_id:
            continue
        for chunk in scene.get("pose_chunks", []):
            semantic_file = chunk.get("semantic_file")
            semantic_path = ROOT / str(semantic_file) if semantic_file else None
            if semantic_path is not None and semantic_path.is_file():
                referenced_semantic_files.add(semantic_path)
                chunk["status"] = "semantics_labeled"
            else:
                chunk.pop("semantic_file", None)
                chunk.pop("semantic_text", None)
                chunk["status"] = "hint_extracted"
            for key in SEMANTIC_UPLOAD_CHUNK_KEYS:
                chunk.pop(key, None)
    save_scenes(config, scenes)
    remove_unreferenced_files(resolve_path(config, "semantics_root") / slug(video_id), referenced_semantic_files)

    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    video["status"] = "hints_extracted"
    stage_log(video, "rollback_to_hints_extracted", "completed", completed_semantic_chunks=len(referenced_semantic_files))
    save_videos(config, videos)
    return True


def rollback_to_hints_extracted_remove_chunks(config: dict[str, Any], video_id: str) -> bool:
    scenes = load_scenes(config)
    for scene in scenes.get("scenes", []):
        if str(scene.get("video_id", "")) != video_id:
            continue
        for chunk in scene.get("pose_chunks", []):
            for key in (
                "chunk_video_file",
                "chunk_start_sec",
                "chunk_end_sec",
                "semantic_file",
                "semantic_text",
                "hf_semantic_file",
                "hf_trajectory_hint_file",
                "hf_chunk_video_file",
                "hf_chunk_repo_id",
            ):
                value = chunk.pop(key, None)
                if key.endswith("_file") and isinstance(value, str):
                    (ROOT / value).unlink(missing_ok=True)
            if chunk.get("trajectory_hint_file"):
                chunk["status"] = "hint_extracted"
    save_scenes(config, scenes)
    remove_cached_video_dirs(config, "videos_root", video_id)
    remove_cached_video_dirs(config, "semantics_root", video_id)

    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    video["status"] = "hints_extracted"
    stage_log(video, "rollback_to_hints_extracted", "completed")
    save_videos(config, videos)
    return True


def rollback_to_semantics_labeled(config: dict[str, Any], video_id: str) -> bool:
    scenes = load_scenes(config)
    for scene in scenes.get("scenes", []):
        if str(scene.get("video_id", "")) != video_id:
            continue
        for chunk in scene.get("pose_chunks", []):
            for key in SEMANTIC_UPLOAD_CHUNK_KEYS:
                chunk.pop(key, None)
    save_scenes(config, scenes)

    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False
    for key in (
        "hf_repo_id",
        "hf_repo_type",
        "hf_info",
        "hf_upload_width",
        "hf_upload_height",
        "hf_upload_fps",
        "hf_upload_frame_count",
        "hf_upload_duration_sec",
        "hf_chunks_repo_id",
        "output_info_path",
    ):
        video.pop(key, None)
    video["status"] = "semantics_labeled"
    stage_log(video, "rollback_to_semantics_labeled", "completed")
    save_videos(config, videos)
    return True


def rollback_video_to_stage(config: dict[str, Any], video_id: str, active_stage: str | None = None) -> bool:
    status = ""
    try:
        status = str(find_video(load_videos(config), video_id).get("status", ""))
    except KeyError:
        return False

    if active_stage == "download" or status == "downloading":
        rollback_video(config, video_id)
        return True
    if active_stage == "gpu_lane":
        if status == "chunks_split":
            return rollback_to_chunks_split_preserve_pose(config, video_id)
        if status in {"downloaded", "scenes_detected"}:
            return reset_video_to_downloaded(config, video_id)
        if status == "poses_estimated":
            return rollback_to_poses_estimated(config, video_id)
    if active_stage == "vggt":
        return rollback_to_chunks_split_preserve_pose(config, video_id)
    if active_stage == "vlm":
        return rollback_to_hints_extracted_preserve_semantics(config, video_id)
    if active_stage in {"upload", "output"}:
        return rollback_to_semantics_labeled(config, video_id)
    if active_stage in {"scene_detection", "split_chunks"} or status in {"downloaded", "scenes_detected", "chunks_split"}:
        return reset_video_to_downloaded(config, video_id)
    if active_stage in {"judge", "extract_hint"} or status in {"poses_estimated", "trajectory_judged", "hints_extracted"}:
        return rollback_to_poses_estimated(config, video_id)
    if status == "semantics_labeled":
        return rollback_to_semantics_labeled(config, video_id)
    return False


def reset_video_to_downloaded(config: dict[str, Any], video_id: str) -> bool:
    """Drop derived pipeline state while preserving a completed downloaded source video."""
    videos = load_videos(config)
    try:
        video = find_video(videos, video_id)
    except KeyError:
        return False

    cache_video = ROOT / str(video.get("cache_video_path", ""))
    cache_info = ROOT / str(video.get("cache_info_path", ""))
    if not cache_video.is_file() or not cache_info.is_file():
        return False

    processed_video_path(config, video_id).unlink(missing_ok=True)
    hf_upload_video_path(config, video_id).unlink(missing_ok=True)
    remove_video_views(config, video_id)
    shutil.rmtree(pose_cache_dir(config, video_id), ignore_errors=True)
    for key in ("videos_root", "hints_root", "semantics_root"):
        root = resolve_path(config, key)
        if root.exists():
            for path in root.rglob(slug(video_id)):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    scene_state = load_scenes(config)
    scene_state["scenes"] = [
        scene for scene in scene_state.get("scenes", []) if str(scene.get("video_id", "")) != video_id
    ]
    if not scene_state["scenes"]:
        scene_state.pop("detector", None)
    save_scenes(config, scene_state)

    for key in (
        "error",
        "total_scene_count",
        "valid_scene_count",
        "scene_boundaries",
        "scene_boundary_type",
        "scene_detector",
        "scene_detection_height",
        "scene_detection_fps",
        "scene_detection_threshold",
        "scene_cut_dedup_sec",
        "sampled_height",
        "sampled_fps",
        "vggt_input_height",
        "vggt_input_fps",
        "hf_repo_id",
        "hf_repo_type",
        "hf_raw_video",
        "hf_info",
        "hf_upload_height",
        "hf_upload_fps",
        "hf_upload_frame_count",
        "hf_upload_duration_sec",
        "hf_chunks_repo_id",
    ):
        video.pop(key, None)
    video["status"] = "downloaded"
    stage_log(video, "rollback_to_downloaded", "completed")
    save_videos(config, videos)
    write_clean_videos_manifest(config)
    write_clean_scenes_manifest(config)
    return True


def rollback_video(config: dict[str, Any], video_id: str) -> None:
    videos = load_videos(config)
    scenes = load_scenes(config)
    video: dict[str, Any] | None = None
    for candidate in videos.get("videos", []):
        if str(candidate.get("video_id", "")) == video_id:
            video = candidate
            break

    for cache_dir in video_cache_dirs(config, video_id, video):
        shutil.rmtree(cache_dir, ignore_errors=True)
    processed_video_path(config, video_id).unlink(missing_ok=True)
    hf_upload_video_path(config, video_id).unlink(missing_ok=True)
    remove_video_views(config, video_id)
    shutil.rmtree(pose_cache_dir(config, video_id), ignore_errors=True)

    scenes["scenes"] = [
        scene for scene in scenes.get("scenes", []) if str(scene.get("video_id", "")) != video_id
    ]
    if not scenes["scenes"]:
        scenes.pop("detector", None)
    videos["videos"] = [
        item for item in videos.get("videos", []) if str(item.get("video_id", "")) != video_id
    ]
    save_scenes(config, scenes)
    save_videos(config, videos)
    write_clean_videos_manifest(config)
    write_clean_scenes_manifest(config)

def stage_log(video: dict[str, Any], stage: str, status: str, **extra: Any) -> None:
    video.setdefault("stage_history", []).append({"stage": stage, "status": status})
