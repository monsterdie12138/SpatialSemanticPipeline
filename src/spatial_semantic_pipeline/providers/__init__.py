"""External model/provider integration boundaries."""
"""Replaceable integrations used by otherwise tool-agnostic stages."""

from .base import PoseEstimator, SceneDetector, SemanticLabeler
from .registry import get_pose_estimator, get_scene_detector, get_semantic_labeler

__all__ = [
    "PoseEstimator",
    "SceneDetector",
    "SemanticLabeler",
    "get_pose_estimator",
    "get_scene_detector",
    "get_semantic_labeler",
]
