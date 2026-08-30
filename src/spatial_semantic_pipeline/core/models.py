"""Typed shapes used by the dict-based manifest format."""

from typing import Any, Literal, TypedDict

PipelineStatus = Literal[
    "downloading",
    "downloaded",
    "download_failed",
    "scenes_detected",
    "chunks_split",
    "poses_estimated",
    "trajectory_judged",
    "hints_extracted",
    "semantics_labeled",
    "exported",
    "skipped_no_valid_scenes",
    "skipped_vggt",
]


class VideoRecord(TypedDict, total=False):
    video_id: str
    source_url: str
    channel: str
    status: PipelineStatus
    stage_history: list[dict[str, Any]]
    cache_video_path: str
    cache_info_path: str
    duration_sec: float
    frame_count: int
    fps: float
    width: int
    height: int


class ChunkRecord(TypedDict, total=False):
    chunk_index: int
    pose_file: str
    semantic_file: str
    chunk_video_file: str
    trajectory_judge: str
    trajectory_judge_reasons: list[str]
    trajectory_hint: str
    semantic_text: str
    logical_start_sec: float
    logical_end_sec: float
    status: str


class SceneRecord(TypedDict, total=False):
    video_id: str
    scene_index: int
    source_start_timestamp: float
    source_end_timestamp: float
    source_start_frame: int
    source_end_frame: int
    pose_chunks: list[ChunkRecord]


__all__ = ["ChunkRecord", "PipelineStatus", "SceneRecord", "VideoRecord"]
