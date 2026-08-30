"""Stable contracts between pipeline stages and replaceable tool providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SceneDetectionRequest:
    config: dict[str, Any]
    video_path: Path
    height: int | None
    fps: float | None
    threshold: float


@dataclass(frozen=True)
class SceneDetectionResult:
    cut_edges: list[list[int]]
    detector: str


@runtime_checkable
class SceneDetector(Protocol):
    def detect(self, request: SceneDetectionRequest) -> SceneDetectionResult:
        """Return adjacent frame pairs marking scene cuts."""


@dataclass(frozen=True)
class PoseEstimationJob:
    input_video: Path
    output_pose: Path
    sample_fps: float


@dataclass(frozen=True)
class PoseEstimationRequest:
    config: dict[str, Any]
    jobs: tuple[PoseEstimationJob, ...]
    work_dir: Path


@runtime_checkable
class PoseEstimator(Protocol):
    def estimate(self, request: PoseEstimationRequest) -> None:
        """Populate every output pose path declared by the request jobs."""


@dataclass(frozen=True)
class SemanticLabelRequest:
    config: dict[str, Any]
    prompt: str
    clip: Path


@dataclass(frozen=True)
class SemanticLabelResult:
    text: str
    raw_response: Any


@runtime_checkable
class SemanticLabeler(Protocol):
    def label(self, request: SemanticLabelRequest) -> SemanticLabelResult:
        """Return the validated semantic response for one clip."""


__all__ = [
    "PoseEstimationJob",
    "PoseEstimationRequest",
    "PoseEstimator",
    "SceneDetectionRequest",
    "SceneDetectionResult",
    "SceneDetector",
    "SemanticLabelRequest",
    "SemanticLabelResult",
    "SemanticLabeler",
]
