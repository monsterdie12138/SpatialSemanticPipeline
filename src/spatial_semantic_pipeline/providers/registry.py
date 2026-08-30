"""Provider selection without coupling stages to concrete tool implementations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from .base import PoseEstimator, SceneDetector, SemanticLabeler

ProviderFactory = Callable[[], Any]

_SCENE_DETECTORS: dict[str, ProviderFactory] = {}
_POSE_ESTIMATORS: dict[str, ProviderFactory] = {}
_SEMANTIC_LABELERS: dict[str, ProviderFactory] = {}


def register_scene_detector(name: str, factory: ProviderFactory) -> None:
    _SCENE_DETECTORS[name] = factory


def register_pose_estimator(name: str, factory: ProviderFactory) -> None:
    _POSE_ESTIMATORS[name] = factory


def register_semantic_labeler(name: str, factory: ProviderFactory) -> None:
    _SEMANTIC_LABELERS[name] = factory


def _provider_name(config: dict[str, Any], key: str, default: str) -> str:
    providers = config.get("providers") or {}
    if not isinstance(providers, dict):
        raise ValueError("Config field 'providers' must be a mapping")
    return str(providers.get(key) or config.get(f"{key}_provider") or default).strip()


def _load_dotted_provider(spec: str) -> Any:
    if ":" not in spec:
        raise KeyError(spec)
    module_name, attribute_name = spec.split(":", 1)
    provider = getattr(importlib.import_module(module_name), attribute_name)
    return provider() if isinstance(provider, type) else provider


def _resolve(registry: dict[str, ProviderFactory], name: str, kind: str) -> Any:
    factory = registry.get(name)
    if factory is not None:
        return factory()
    try:
        return _load_dotted_provider(name)
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        available = ", ".join(sorted(registry)) or "none"
        raise ValueError(f"Unknown {kind} provider {name!r}; available built-ins: {available}") from exc


def _ensure_scene_builtin() -> None:
    from .transnet import TransNetV2SceneDetector

    _SCENE_DETECTORS.setdefault("transnet_v2", TransNetV2SceneDetector)


def _ensure_pose_builtin() -> None:
    from .vggt import VGGTOmegaPoseEstimator

    _POSE_ESTIMATORS.setdefault("vggt_omega", VGGTOmegaPoseEstimator)


def _ensure_semantic_builtin() -> None:
    from .vlm import OpenAICompatibleSemanticLabeler

    _SEMANTIC_LABELERS.setdefault("openai_compatible", OpenAICompatibleSemanticLabeler)


def get_scene_detector(config: dict[str, Any]) -> SceneDetector:
    _ensure_scene_builtin()
    name = _provider_name(config, "scene_detector", "transnet_v2")
    provider = _resolve(_SCENE_DETECTORS, name, "scene detector")
    if not isinstance(provider, SceneDetector):
        raise TypeError(f"Scene detector provider {name!r} must implement detect(request)")
    return cast(SceneDetector, provider)


def get_pose_estimator(config: dict[str, Any]) -> PoseEstimator:
    _ensure_pose_builtin()
    name = _provider_name(config, "pose_estimator", "vggt_omega")
    provider = _resolve(_POSE_ESTIMATORS, name, "pose estimator")
    if not isinstance(provider, PoseEstimator):
        raise TypeError(f"Pose estimator provider {name!r} must implement estimate(request)")
    return cast(PoseEstimator, provider)


def get_semantic_labeler(config: dict[str, Any]) -> SemanticLabeler:
    _ensure_semantic_builtin()
    name = _provider_name(config, "semantic_labeler", "openai_compatible")
    provider = _resolve(_SEMANTIC_LABELERS, name, "semantic labeler")
    if not isinstance(provider, SemanticLabeler):
        raise TypeError(f"Semantic labeler provider {name!r} must implement label(request)")
    return cast(SemanticLabeler, provider)


__all__ = [
    "get_pose_estimator",
    "get_scene_detector",
    "get_semantic_labeler",
    "register_pose_estimator",
    "register_scene_detector",
    "register_semantic_labeler",
]
