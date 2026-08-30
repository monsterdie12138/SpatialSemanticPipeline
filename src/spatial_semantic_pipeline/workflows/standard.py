#!/usr/bin/env python3
"""Run the staged pipeline with download/GPU/local-output overlap."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GPU_IDLE_MEMORY_USED_MB = 4096
GPU_IDLE_UTILIZATION_PERCENT = 20
DEFAULT_PREFERRED_GPU_IDS = [1, 2, 3, 4]
MAX_GPU_LANE_WORKERS = 4
from ..context import (
    activate_pipeline_environment,
    cleanup_successful_runtime,
    find_video,
    find_video_by_source,
    load_clean_videos,
    load_videos,
    mutate_videos,
    normalize_source_url,
    pipeline_config,
    pipeline_env,
    read_sources,
    remove_video_from_local_state,
    rollback_video_to_stage,
    rollback_video,
    source_is_registered,
    source_video_id,
    sources_path,
    cache_root,
    stage_log,
    video_cache_dir,
)
from ..core.stages import Status, stage_module


class GpuOomRetry(RuntimeError):
    def __init__(self, video_id: str, gpu_id: int, status: str, stage: str) -> None:
        super().__init__(f"{stage} CUDA OOM for video_id={video_id} on GPU {gpu_id}; status={status}")
        self.video_id = video_id
        self.gpu_id = gpu_id
        self.status = status
        self.stage = stage


def workflow_log_path() -> Path:
    return Path(os.environ.get("LOG_FILE", ROOT / "logs" / "workflow_latest.log"))


def recent_log_has_stage_oom(video_id: str, script: str, max_bytes: int = 512 * 1024) -> bool:
    try:
        path = workflow_log_path()
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", "ignore").lower()
    except OSError:
        return False

    marker = f"starting stage {script} --video-id={video_id}".lower()
    marker_index = text.rfind(marker)
    if marker_index >= 0:
        text = text[marker_index:]
    oom_markers = (
        "torch.cuda.outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "out of memory",
    )
    return any(oom_marker in text for oom_marker in oom_markers)


def run_gpu_stage_with_oom_retry(
    script: str,
    video_id: str,
    gpu_id: int,
    stage_name: str,
    rollback_stage: str,
    *,
    env_extra: dict[str, str] | None = None,
) -> None:
    config = pipeline_config()
    try:
        run_stage(script, f"--video-id={video_id}", env_extra=env_extra)
    except subprocess.CalledProcessError:
        if recent_log_has_stage_oom(video_id, script):
            rollback_video_to_stage(config, video_id, rollback_stage)
            status = video_status(config, video_id)
            print(
                f"{stage_name} CUDA OOM for {video_id} on GPU {gpu_id}; "
                f"rolled back to status={status} and will retry when an idle GPU is available.",
                file=sys.stderr,
                flush=True,
            )
            raise GpuOomRetry(video_id, gpu_id, status, stage_name)
        raise


def run_stage(script: str, *args: str, env_extra: dict[str, str] | None = None) -> None:
    module = stage_module(pipeline_config(), script)
    stage_args = " ".join(args)
    stage_started_at = time.monotonic()
    print(f"Starting stage {script} {stage_args}", flush=True)
    cmd = [sys.executable, "-m", module, *args]
    module_env = dict(env_extra or {})
    current_pythonpath = os.environ.get("PYTHONPATH")
    module_env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + current_pythonpath if current_pythonpath else "")
    env = pipeline_env(pipeline_config(), module_env)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        elapsed_sec = time.monotonic() - stage_started_at
        print(
            f"Failed stage {script} {stage_args} exit_code={exc.returncode} elapsed_sec={elapsed_sec:.1f}",
            file=sys.stderr,
            flush=True,
        )
        raise
    elapsed_sec = time.monotonic() - stage_started_at
    print(f"Finished stage {script} {stage_args} elapsed_sec={elapsed_sec:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pipeline workflow.")
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--skip-vggt", action="store_true")
    parser.add_argument(
        "--skip-output",
        "--skip-publish",
        dest="skip_output",
        action="store_true",
        help="Stop after semantic labeling without exporting the final output package.",
    )
    parser.add_argument("--ignore-clean-manifest", action="store_true")
    return parser.parse_args()


def pending_sources(config: dict[str, Any], limit: int, ignore_clean_manifest: bool = False) -> list[str]:
    if limit <= 0:
        return []
    runtime_videos = load_videos(config).get("videos", [])
    final_videos = [] if ignore_clean_manifest else load_clean_videos(config)
    videos = {"videos": [*runtime_videos, *final_videos]}
    selected: list[str] = []
    selected_keys: set[str] = set()
    for source in read_sources(sources_path(config)):
        if source_is_registered(videos, source):
            continue
        key = source_video_id(source) or normalize_source_url(source)
        if key in selected_keys:
            continue
        selected.append(source)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def download_source(source: str) -> str | None:
    config = pipeline_config()
    try:
        run_stage("01_download_video.py", "--source", source)
    except subprocess.CalledProcessError as exc:
        video = find_video_by_source(load_videos(config), source)
        if video is not None:
            video_id = str(video["video_id"])
            rollback_video(config, video_id)
            print(f"Rolled back failed download video_id={video_id}: {source}", file=sys.stderr, flush=True)
        else:
            print(f"Download failed before registration: {source}", file=sys.stderr, flush=True)
        print(f"01_download_video.py exited with code {exc.returncode}; stopping new downloads.", file=sys.stderr, flush=True)
        return None

    video = find_video_by_source(load_videos(config), source)
    if video is None:
        return None
    return str(video["video_id"])


def video_status(config: dict[str, Any], video_id: str) -> str:
    for video in load_videos(config).get("videos", []):
        if str(video.get("video_id", "")) == video_id:
            return str(video.get("status", ""))
    return ""


def parse_gpu_id_list(value: Any, default: list[int]) -> list[int]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [value]

    gpu_ids: list[int] = []
    seen: set[int] = set()
    for raw_item in raw_items:
        try:
            gpu_id = int(str(raw_item).strip())
        except ValueError:
            continue
        if gpu_id < 0 or gpu_id in seen:
            continue
        gpu_ids.append(gpu_id)
        seen.add(gpu_id)
    return gpu_ids or list(default)


def gpu_priority_key(gpu_id: int, preferred_gpu_ids: list[int]) -> tuple[int, int]:
    try:
        return (0, preferred_gpu_ids.index(gpu_id))
    except ValueError:
        return (1, gpu_id)


def query_idle_gpu_ids(preferred_gpu_ids: list[int]) -> list[int]:
    gpu_query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    compute_query_cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]

    def run_query() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            gpu_query_cmd,
            check=True,
            text=True,
            capture_output=True,
        )

    try:
        result = run_query()
    except Exception as first_exc:
        try:
            subprocess.run(["nvidia-modprobe", "-u", "-c=0"], check=True)
            result = run_query()
        except Exception as second_exc:
            raise RuntimeError(
                "Could not query GPUs with nvidia-smi. Check that the NVIDIA driver is running "
                "and visible in this environment before starting GPU stages."
            ) from second_exc
        else:
            print(
                f"Recovered NVIDIA device nodes with nvidia-modprobe after nvidia-smi failed: {first_exc}",
                file=sys.stderr,
                flush=True,
            )

    occupied_gpu_uuids: set[str] = set()
    compute_result = subprocess.run(
        compute_query_cmd,
        check=False,
        text=True,
        capture_output=True,
    )
    if compute_result.returncode == 0:
        for line in compute_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[1]:
                occupied_gpu_uuids.add(parts[0])

    idle_ids: list[int] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            gpu_uuid = parts[1]
            memory_used_mb = int(parts[2])
            utilization_percent = int(parts[3])
        except ValueError:
            continue
        is_preferred_gpu = index in preferred_gpu_ids
        if not is_preferred_gpu and gpu_uuid in occupied_gpu_uuids:
            continue
        if memory_used_mb <= GPU_IDLE_MEMORY_USED_MB and utilization_percent <= GPU_IDLE_UTILIZATION_PERCENT:
            idle_ids.append(index)
    return sorted(idle_ids, key=lambda gpu_id: gpu_priority_key(gpu_id, preferred_gpu_ids))


def run_gpu_lane(video_id: str, skip_vggt: bool, gpu_id: int) -> str:
    config = pipeline_config()
    gpu_env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    status = video_status(config, video_id)
    print(f"Running GPU lane for {video_id} from status={status} on GPU {gpu_id}", flush=True)
    if status == Status.DOWNLOADED:
        run_gpu_stage_with_oom_retry(
            "02_detect_scenes.py",
            video_id,
            gpu_id,
            "TransNetV2",
            "gpu_lane",
            env_extra=gpu_env,
        )
        status = video_status(config, video_id)
        print(f"After detect_scenes {video_id} status={status}", flush=True)
    if status == Status.SCENES_DETECTED:
        print(f"Holding GPU {gpu_id} while preparing chunks for {video_id}; 03 itself does not use GPU.", flush=True)
        run_stage("03_split_chunks.py", f"--video-id={video_id}")
        status = video_status(config, video_id)
        print(f"After split_chunks {video_id} status={status}", flush=True)
    if status == Status.CHUNKS_SPLIT and not skip_vggt:
        print(f"Ensuring prepared chunk files for {video_id} before VGGT.", flush=True)
        run_stage("03_split_chunks.py", f"--video-id={video_id}")
        status = video_status(config, video_id)
        print(f"After ensuring split_chunks {video_id} status={status}", flush=True)
        print(f"Running VGGT estimate for {video_id} on the same GPU {gpu_id}", flush=True)
        run_gpu_stage_with_oom_retry(
            "04_estimate_trajectories.py",
            video_id,
            gpu_id,
            "VGGT",
            "gpu_lane",
            env_extra=gpu_env,
        )
        status = video_status(config, video_id)
        print(f"After estimate_trajectories {video_id} status={status}", flush=True)
    if status == Status.CHUNKS_SPLIT and skip_vggt:
        print(f"VGGT skipped for {video_id}; GPU lane released after chunk preparation.", flush=True)
    elif status not in {Status.POSES_ESTIMATED, Status.CHUNKS_SPLIT}:
        print(f"GPU lane finished for {video_id} with status={status}", flush=True)
    return video_id


def run_postprocess_stage(video_id: str, skip_vggt: bool) -> str:
    config = pipeline_config()
    status = video_status(config, video_id)
    print(f"Running postprocess stage for {video_id} from status={status}", flush=True)
    if status == Status.POSES_ESTIMATED:
        run_stage("05_judge.py", f"--video-id={video_id}")
        status = video_status(config, video_id)
        print(f"After judge {video_id} status={status}", flush=True)
        return video_id
    if status == Status.TRAJECTORY_JUDGED:
        run_stage("06_extract_hint.py", f"--video-id={video_id}")
        status = video_status(config, video_id)
        print(f"After extract_hint {video_id} status={status}", flush=True)
        return video_id
    if status == Status.HINTS_EXTRACTED:
        run_stage(
            "07_call_gemini.py",
            f"--video-id={video_id}",
            env_extra={"PYTHONNOUSERSITE": "1"},
        )
        status = video_status(config, video_id)
        print(f"After call_gemini {video_id} status={status}", flush=True)
        return video_id
    if status == Status.CHUNKS_SPLIT and skip_vggt:
        def mark_skipped(videos_data: dict[str, Any]) -> None:
            video = find_video(videos_data, video_id)
            video["status"] = Status.SKIPPED_VGGT
            stage_log(video, "skip_vggt", "skipped", reason="no trajectory judge for good/full-minute filtering")

        mutate_videos(config, mark_skipped)
        print(f"Skipped VLM for {video_id}; --skip-vggt leaves no good/full-minute trajectory labels.", flush=True)
    return video_id


def export_video(video_id: str) -> str:
    run_stage("08_stage_output.py", f"--video-id={video_id}")
    return video_id


def free_disk_gb(config: dict[str, Any]) -> float:
    usage = shutil.disk_usage(cache_root(config))
    return usage.free / (1024 ** 3)


def can_submit_download(config: dict[str, Any]) -> bool:
    min_free_gb = float(config.get("min_free_disk_gb", 0) or 0)
    if min_free_gb <= 0:
        return True
    free_gb = free_disk_gb(config)
    if free_gb < min_free_gb:
        print(
            f"Paused new downloads: free disk {free_gb:.1f}GB < min_free_disk_gb {min_free_gb:.1f}GB.",
            flush=True,
        )
        return False
    return True


def rollback_known_videos(
    config: dict[str, Any],
    video_ids: set[str],
    completed: set[str],
    active_stages: dict[str, str] | None = None,
) -> None:
    active_stages = active_stages or {}
    for video_id in sorted(video_ids - completed):
        stage = active_stages.get(video_id)
        if rollback_video_to_stage(config, video_id, stage):
            print(
                f"Rolled back video_id={video_id} after workflow abort "
                f"(active_stage={stage or 'none'}, status={video_status(config, video_id)}).",
                file=sys.stderr,
            )
            continue
        rollback_video(config, video_id)
        print(f"Fully rolled back incomplete video_id={video_id} after workflow abort.", file=sys.stderr)


def recover_runtime_state(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    processing_statuses = {
        Status.DOWNLOADED,
        Status.SCENES_DETECTED,
        Status.POSES_ESTIMATED,
        Status.TRAJECTORY_JUDGED,
        Status.HINTS_EXTRACTED,
        Status.CHUNKS_SPLIT,
    }
    export_statuses = {Status.SEMANTICS_LABELED}
    process_ids: list[str] = []
    export_ids: list[str] = []
    for video in list(load_videos(config).get("videos", [])):
        video_id = str(video.get("video_id", ""))
        status = str(video.get("status", ""))
        if not video_id:
            continue
        if status == Status.DOWNLOADING:
            cached_dir = video_cache_dir(config, video_id)
            cached_video = cached_dir / "original.mp4"
            cached_info = cached_dir / "info.json"
            if cached_video.is_file() and cached_info.is_file():
                remove_video_from_local_state(config, video_id)
                print(
                    f"Dropped stale downloading state for video_id={video_id}; cached download will be reused.",
                    file=sys.stderr,
                )
            else:
                rollback_video(config, video_id)
                print(f"Rolled back stale downloading video_id={video_id} before workflow resume.", file=sys.stderr)
        elif status in processing_statuses:
            process_ids.append(video_id)
        elif status in export_statuses:
            export_ids.append(video_id)
    return process_ids, export_ids


def main() -> int:
    workflow_started_at = time.monotonic()
    args = parse_args()
    config = pipeline_config()
    activate_pipeline_environment(config)
    resume_process_ids, resume_export_ids = recover_runtime_state(config)
    sources = pending_sources(config, args.max_videos, args.ignore_clean_manifest)
    if not sources and not resume_process_ids and not resume_export_ids:
        if not args.skip_output:
            cleanup_successful_runtime(config)
        elapsed_sec = time.monotonic() - workflow_started_at
        print(f"No pending source URLs in sources.txt. elapsed_sec={elapsed_sec:.1f}")
        return 0

    download_workers = max(int(config.get("download_workers", 1)), 1)
    configured_gpu_workers = max(int(config.get("max_gpu_devices", config.get("gpu_workers", 1))), 1)
    max_gpu_devices = min(configured_gpu_workers, MAX_GPU_LANE_WORKERS)
    preferred_gpu_ids = parse_gpu_id_list(config.get("preferred_gpu_ids"), DEFAULT_PREFERRED_GPU_IDS)
    postprocess_workers = max(int(config.get("postprocess_workers", 1)), 1)
    output_workers = max(int(config.get("output_workers", 1)), 1)
    disk_check_interval_sec = max(int(config.get("disk_check_interval_sec", 300)), 1)
    gpu_check_interval_sec = max(int(config.get("gpu_check_interval_sec", 15)), 1)

    download_pool = ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix="download")
    gpu_pool = ThreadPoolExecutor(max_workers=max_gpu_devices, thread_name_prefix="gpu")
    post_pool = ThreadPoolExecutor(max_workers=postprocess_workers, thread_name_prefix="post")
    output_pool = ThreadPoolExecutor(max_workers=output_workers, thread_name_prefix="output")

    next_source_index = 0
    download_futures: dict[Future, str] = {}
    gpu_futures: dict[Future, tuple[str, int]] = {}
    post_futures: dict[Future, str] = {}
    output_futures: dict[Future, str] = {}
    gpu_queue: list[str] = []
    post_queue: list[str] = []
    scheduled_sources: set[str] = set()
    known_video_ids: set[str] = set()
    completed_video_ids: set[str] = set()
    finished_video_ids: set[str] = set()
    completed_count = 0
    target_count = len(sources) + len(resume_process_ids) + len(resume_export_ids)
    active_stages: dict[str, str] = {}
    failed = False
    download_failed = False
    failed_download_sources: set[str] = set()
    stop_new_downloads = False

    def submit_downloads() -> None:
        nonlocal next_source_index
        if stop_new_downloads:
            return
        while len(download_futures) < download_workers and next_source_index < len(sources):
            if not can_submit_download(config):
                return
            source = sources[next_source_index]
            next_source_index += 1
            scheduled_sources.add(source)
            future = download_pool.submit(download_source, source)
            download_futures[future] = source
            print(f"Queued download: {source}")

    def mark_completed(video_id: str, message: str, *, exported: bool = True) -> None:
        nonlocal completed_count
        finished_video_ids.add(video_id)
        if exported:
            completed_video_ids.add(video_id)
        completed_count += 1
        print(message, flush=True)

    def enqueue_next_stage(video_id: str) -> None:
        status = video_status(config, video_id)
        if status == Status.DOWNLOADED:
            gpu_queue.append(video_id)
            print(f"Queued {video_id} for GPU 02-03-04 lane.", flush=True)
        elif status == Status.SKIPPED_NO_VALID_SCENES:
            mark_completed(video_id, f"Skipped {video_id}; no valid scenes after scene detection.", exported=False)
        elif status == Status.SKIPPED_VGGT:
            mark_completed(video_id, f"Skipped {video_id}; VGGT was disabled so no good/full-minute chunks exist.", exported=False)
        elif status == Status.SCENES_DETECTED:
            gpu_queue.append(video_id)
            print(f"Queued {video_id} for GPU lane from chunk preparation.", flush=True)
        elif status == Status.CHUNKS_SPLIT:
            if args.skip_vggt:
                post_queue.append(video_id)
                print(f"Queued {video_id} for VLM because VGGT is skipped.", flush=True)
            else:
                gpu_queue.append(video_id)
                print(f"Queued {video_id} for GPU lane from chunks_split.", flush=True)
        elif status in {Status.POSES_ESTIMATED, Status.TRAJECTORY_JUDGED, Status.HINTS_EXTRACTED}:
            post_queue.append(video_id)
            print(f"Queued {video_id} for postprocess from status={status}.", flush=True)
        elif status == Status.SEMANTICS_LABELED:
            if args.skip_output:
                mark_completed(video_id, f"Processed {video_id}; output export skipped.", exported=False)
            else:
                output_future = output_pool.submit(export_video, video_id)
                output_futures[output_future] = video_id
                active_stages[video_id] = "output"
                print(f"Queued output export for {video_id}.", flush=True)
        else:
            raise RuntimeError(f"Cannot route video_id={video_id} from status={status!r}")

    def submit_gpu_jobs() -> None:
        assigned_gpu_ids = {gpu_id for _video_id, gpu_id in gpu_futures.values()}
        idle_gpu_ids = [
            gpu_id for gpu_id in query_idle_gpu_ids(preferred_gpu_ids)
            if gpu_id not in assigned_gpu_ids
        ]
        while gpu_queue and idle_gpu_ids:
            gpu_id = idle_gpu_ids.pop(0)
            video_id = gpu_queue.pop(0)
            future = gpu_pool.submit(run_gpu_lane, video_id, args.skip_vggt, gpu_id)
            gpu_futures[future] = (video_id, gpu_id)
            active_stages[video_id] = "gpu_lane"
            print(f"Queued GPU lane for {video_id} on GPU {gpu_id}.")

    def submit_post_jobs() -> None:
        while post_queue and len(post_futures) < postprocess_workers:
            video_id = post_queue.pop(0)
            status = video_status(config, video_id)
            future = post_pool.submit(run_postprocess_stage, video_id, args.skip_vggt)
            post_futures[future] = video_id
            active_stages[video_id] = {
                Status.CHUNKS_SPLIT: "vlm",
                Status.POSES_ESTIMATED: "judge",
                Status.TRAJECTORY_JUDGED: "extract_hint",
                Status.HINTS_EXTRACTED: "vlm",
            }.get(status, "postprocess")
            print(f"Queued postprocess stage for {video_id}.", flush=True)

    try:
        for video_id in resume_process_ids:
            known_video_ids.add(video_id)
            enqueue_next_stage(video_id)
            print(f"Queued resumed processing {video_id} from status={video_status(config, video_id)}.")
        for video_id in resume_export_ids:
            known_video_ids.add(video_id)
            if args.skip_output:
                mark_completed(video_id, f"Resumed processed {video_id}; output export skipped.", exported=False)
            else:
                output_future = output_pool.submit(export_video, video_id)
                output_futures[output_future] = video_id
                active_stages[video_id] = "output"
                print(f"Resumed output export {video_id}.")
        submit_downloads()
        submit_gpu_jobs()
        submit_post_jobs()
        while completed_count < target_count:
            submit_gpu_jobs()
            submit_post_jobs()
            active = set(download_futures) | set(gpu_futures) | set(post_futures) | set(output_futures)
            if not active:
                if gpu_queue:
                    print(
                        f"Waiting {gpu_check_interval_sec}s for an idle GPU before starting {len(gpu_queue)} queued GPU jobs.",
                        flush=True,
                    )
                    time.sleep(gpu_check_interval_sec)
                    submit_gpu_jobs()
                    continue
                if post_queue:
                    submit_post_jobs()
                    continue
                if next_source_index < len(sources) and not stop_new_downloads:
                    print(
                        f"Waiting {disk_check_interval_sec}s before retrying downloads.",
                        flush=True,
                    )
                    time.sleep(disk_check_interval_sec)
                    submit_downloads()
                    continue
                break
            done, _pending = wait(
                active,
                timeout=gpu_check_interval_sec if gpu_queue else None,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                if future in download_futures:
                    source = download_futures.pop(future)
                    video_id = future.result()
                    if video_id is None:
                        download_failed = True
                        failed_download_sources.add(source)
                        stop_new_downloads = True
                        completed_count += 1
                        print(f"Download failed and was rolled back: {source}", file=sys.stderr, flush=True)
                        continue
                    known_video_ids.add(video_id)
                    enqueue_next_stage(video_id)
                    submit_gpu_jobs()
                    submit_post_jobs()
                    print(f"Downloaded {video_id}; queued next stage.")
                    submit_downloads()
                elif future in gpu_futures:
                    video_id, gpu_id = gpu_futures.pop(future)
                    try:
                        future.result()
                    except GpuOomRetry as exc:
                        active_stages.pop(video_id, None)
                        if video_status(config, video_id) != Status.CHUNKS_SPLIT:
                            rollback_video_to_stage(config, video_id, "gpu_lane")
                        status = video_status(config, video_id)
                        gpu_queue.append(video_id)
                        print(
                            f"Requeued {video_id} after {exc.stage} CUDA OOM on GPU {gpu_id}; "
                            f"status={status}. Waiting for an idle GPU; preferred order is "
                            f"{preferred_gpu_ids}, then any other idle GPU.",
                            file=sys.stderr,
                            flush=True,
                        )
                        submit_gpu_jobs()
                        submit_post_jobs()
                        continue
                    active_stages.pop(video_id, None)
                    enqueue_next_stage(video_id)
                    submit_gpu_jobs()
                    submit_post_jobs()
                elif future in post_futures:
                    video_id = post_futures.pop(future)
                    future.result()
                    active_stages.pop(video_id, None)
                    enqueue_next_stage(video_id)
                    submit_gpu_jobs()
                    submit_post_jobs()
                elif future in output_futures:
                    video_id = output_futures.pop(future)
                    future.result()
                    active_stages.pop(video_id, None)
                    mark_completed(video_id, f"Exported {video_id} to output.")
                    submit_downloads()
    except BaseException:
        failed = True
        print("Workflow interrupted or failed; stopping scheduling and rolling back unfinished videos.", file=sys.stderr)
        raise
    finally:
        download_pool.shutdown(wait=True, cancel_futures=True)
        gpu_pool.shutdown(wait=True, cancel_futures=True)
        post_pool.shutdown(wait=True, cancel_futures=True)
        output_pool.shutdown(wait=True, cancel_futures=True)
        if failed:
            elapsed_sec = time.monotonic() - workflow_started_at
            print(
                "Workflow summary: "
                f"status=failed target={target_count} completed={completed_count} "
                f"known_videos={len(known_video_ids)} exported={len(completed_video_ids)} "
                f"download_failed={len(failed_download_sources)} elapsed_sec={elapsed_sec:.1f}",
                file=sys.stderr,
                flush=True,
            )
            videos = load_videos(config)
            for source in scheduled_sources:
                video = find_video_by_source(videos, source)
                if video is not None:
                    known_video_ids.add(str(video["video_id"]))
            rollback_known_videos(config, known_video_ids, finished_video_ids, active_stages)

    elapsed_sec = time.monotonic() - workflow_started_at
    status = "finished_with_download_failures" if download_failed else "completed"
    print(
        "Workflow summary: "
        f"status={status} selected_sources={len(sources)} "
        f"resume_process={len(resume_process_ids)} resume_export={len(resume_export_ids)} "
        f"target={target_count} completed={completed_count} "
        f"known_videos={len(known_video_ids)} exported={len(completed_video_ids)} "
        f"download_failed={len(failed_download_sources)} elapsed_sec={elapsed_sec:.1f}",
        flush=True,
    )
    if completed_video_ids:
        print(f"Workflow exported video_ids: {', '.join(sorted(completed_video_ids))}", flush=True)
    if failed_download_sources:
        print(f"Workflow failed download sources: {', '.join(sorted(failed_download_sources))}", file=sys.stderr, flush=True)

    if not download_failed and not args.skip_output:
        cleanup_successful_runtime(config)
        print("Removed successful-run cache, metadata, and output staging directories.", flush=True)

    return 1 if download_failed else 0


if __name__ == "__main__":
    sys.exit(main())
