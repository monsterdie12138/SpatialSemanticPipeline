"""Canonical stage and status contracts used by the standard workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Status:
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    DOWNLOAD_FAILED = "download_failed"
    SCENES_DETECTED = "scenes_detected"
    CHUNKS_SPLIT = "chunks_split"
    POSES_ESTIMATED = "poses_estimated"
    TRAJECTORY_JUDGED = "trajectory_judged"
    HINTS_EXTRACTED = "hints_extracted"
    SEMANTICS_LABELED = "semantics_labeled"
    EXPORTED = "exported"
    SKIPPED_NO_VALID_SCENES = "skipped_no_valid_scenes"
    SKIPPED_VGGT = "skipped_vggt"


@dataclass(frozen=True)
class StageDefinition:
    key: str
    command_name: str
    default_module: str
    input_statuses: tuple[str, ...]
    output_statuses: tuple[str, ...]


STAGE_DEFINITIONS = (
    StageDefinition(
        "download",
        "01_download_video.py",
        "spatial_semantic_pipeline.stages.download",
        (),
        (Status.DOWNLOADED, Status.DOWNLOAD_FAILED),
    ),
    StageDefinition(
        "detect_scenes",
        "02_detect_scenes.py",
        "spatial_semantic_pipeline.stages.detect_scenes",
        (Status.DOWNLOADED,),
        (Status.SCENES_DETECTED, Status.SKIPPED_NO_VALID_SCENES),
    ),
    StageDefinition(
        "split_chunks",
        "03_split_chunks.py",
        "spatial_semantic_pipeline.stages.split_chunks",
        (Status.SCENES_DETECTED, Status.CHUNKS_SPLIT),
        (Status.CHUNKS_SPLIT,),
    ),
    StageDefinition(
        "estimate_trajectories",
        "04_estimate_trajectories.py",
        "spatial_semantic_pipeline.stages.estimate_trajectories",
        (Status.CHUNKS_SPLIT,),
        (Status.POSES_ESTIMATED,),
    ),
    StageDefinition(
        "judge_trajectory",
        "05_judge.py",
        "spatial_semantic_pipeline.stages.judge_trajectory",
        (Status.POSES_ESTIMATED,),
        (Status.TRAJECTORY_JUDGED,),
    ),
    StageDefinition(
        "extract_hint",
        "06_extract_hint.py",
        "spatial_semantic_pipeline.stages.extract_hint",
        (Status.TRAJECTORY_JUDGED,),
        (Status.HINTS_EXTRACTED,),
    ),
    StageDefinition(
        "label_semantics",
        "07_call_gemini.py",
        "spatial_semantic_pipeline.stages.label_semantics",
        (Status.HINTS_EXTRACTED,),
        (Status.SEMANTICS_LABELED,),
    ),
    StageDefinition(
        "publish",
        "08_stage_output.py",
        "spatial_semantic_pipeline.stages.publish",
        (Status.SEMANTICS_LABELED,),
        (Status.EXPORTED,),
    ),
)

_STAGE_BY_COMMAND = {stage.command_name: stage for stage in STAGE_DEFINITIONS}


def stage_module(config: dict[str, Any], command_name: str) -> str:
    try:
        definition = _STAGE_BY_COMMAND[command_name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline stage command: {command_name}") from exc
    overrides = config.get("stages") or {}
    if not isinstance(overrides, dict):
        raise ValueError("Config field 'stages' must be a mapping")
    module = overrides.get(definition.key) or overrides.get(command_name) or definition.default_module
    if not isinstance(module, str) or not module.strip():
        raise ValueError(f"Configured module for stage {definition.key!r} must be a non-empty string")
    return module.strip()


__all__ = ["STAGE_DEFINITIONS", "StageDefinition", "Status", "stage_module"]
