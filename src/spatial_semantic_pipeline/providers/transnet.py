"""Built-in TransNetV2 scene detector."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ..context import ROOT, path_from_config, pipeline_env
from .base import SceneDetectionRequest, SceneDetectionResult


def resolve_pytorch_transnet_dir(transnetv2_path: Path) -> Path:
    path = transnetv2_path.expanduser().resolve()
    if path.name == "inference-pytorch":
        return path
    sibling = path.parent / "inference-pytorch"
    if sibling.exists():
        return sibling
    candidate = path / "inference-pytorch"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find TransNetV2 inference-pytorch directory near {transnetv2_path}")


def extract_transnet_frames(
    config: dict, video_path: Path, height: int | None, fps: float | None
) -> np.ndarray:
    filters = []
    if fps is not None:
        fps_text = str(int(fps)) if fps.is_integer() else str(fps)
        filters.append(f"fps=fps={fps_text}:start_time=0")
    if height is not None:
        filters.append(f"scale=-2:{height}")
    filters.append("scale=48:27")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", ",".join(filters), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, env=pipeline_env(config))
    frame_size = 27 * 48 * 3
    if len(result.stdout) % frame_size != 0:
        raise ValueError(f"Unexpected raw frame byte count from ffmpeg for {video_path}")
    return np.frombuffer(result.stdout, np.uint8).reshape([-1, 27, 48, 3])


def predict_frames_pytorch(model: torch.nn.Module, frames: np.ndarray, device: torch.device) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("TransNetV2 extracted no frames")
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
    )
    predictions = []
    ptr = 0
    with torch.no_grad():
        while ptr + 100 <= len(padded_inputs):
            window = padded_inputs[ptr : ptr + 100]
            ptr += 50
            tensor = torch.from_numpy(window[np.newaxis]).to(device=device, dtype=torch.uint8)
            single_frame_pred, _all_frame_pred = model(tensor)
            single_frame_pred = torch.sigmoid(single_frame_pred)[0, 25:75, 0].detach().cpu().numpy()
            predictions.append(single_frame_pred)
            print(
                "\r[TransNetV2/PyTorch] Processing video frames {}/{}".format(
                    min(len(predictions) * 50, len(frames)), len(frames)
                ),
                end="",
            )
    print("")
    return np.concatenate(predictions)[: len(frames)]


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def transnetv2_boundaries(
    config: dict,
    video_path: Path,
    transnetv2_path: Path,
    height: int | None,
    fps: float | None,
    threshold: float,
) -> list[list[int]]:
    transnet_dir = resolve_pytorch_transnet_dir(transnetv2_path)
    weights_path = transnet_dir / "transnetv2-pytorch-weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing TransNetV2 PyTorch weights: {weights_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("TransNetV2 requires GPU, but torch.cuda.is_available() is false in this environment.")
    sys.path.insert(0, str(transnet_dir))
    from transnetv2_pytorch import TransNetV2

    device = torch.device("cuda")
    print(f"[TransNetV2/PyTorch] Using GPU: {torch.cuda.get_device_name(0)}")
    model = TransNetV2()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print(f"[TransNetV2/PyTorch] Extracting frames from {video_path}")
    video_frames = extract_transnet_frames(config, video_path, height, fps)
    single_frame_predictions = predict_frames_pytorch(model, video_frames, device)
    scenes = predictions_to_scenes(single_frame_predictions, threshold=threshold)
    cut_edges: list[list[int]] = []
    for _start_frame, next_frame in scenes[:-1]:
        next_frame = int(next_frame)
        if next_frame <= 0:
            continue
        cut_edges.append([next_frame - 1, next_frame])
    return cut_edges


class TransNetV2SceneDetector:
    def detect(self, request: SceneDetectionRequest) -> SceneDetectionResult:
        tool_path = path_from_config(
            request.config,
            "transnetv2_path",
            ROOT / "tools" / "TransNetV2" / "inference-pytorch",
        )
        return SceneDetectionResult(
            cut_edges=transnetv2_boundaries(
                request.config,
                request.video_path,
                tool_path,
                request.height,
                request.fps,
                request.threshold,
            ),
            detector=str(tool_path),
        )


__all__ = [
    "TransNetV2SceneDetector",
    "extract_transnet_frames",
    "predict_frames_pytorch",
    "predictions_to_scenes",
    "resolve_pytorch_transnet_dir",
    "transnetv2_boundaries",
]
