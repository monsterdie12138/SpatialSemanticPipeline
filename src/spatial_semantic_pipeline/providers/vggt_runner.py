"""Pipeline-owned VGGT-Omega video-to-pose adapter.

The model implementation remains in the separately downloaded VGGT-Omega
repository. This module preserves the pipeline's existing frame sampling,
camera conversion, CSV schema, and one-model-load batch behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    sample_fps: float,
    frame_stride: int | None,
    max_frames: int | None,
) -> tuple[list[Path], list[int], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_stride is None:
        if sample_fps <= 0:
            frame_stride = 1
        else:
            frame_stride = max(int(round((fps if fps > 0 else sample_fps) / sample_fps)), 1)
    else:
        frame_stride = max(int(frame_stride), 1)

    image_paths: list[Path] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            image_path = frame_dir / f"{saved_idx:06}.png"
            cv2.imwrite(str(image_path), frame)
            image_paths.append(image_path)
            frame_indices.append(frame_idx)
            timestamps.append(frame_idx / fps if fps > 0 else float(saved_idx))
            saved_idx += 1
            if max_frames is not None and saved_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not image_paths:
        raise ValueError(f"No frames were extracted from {video_path}")
    return image_paths, frame_indices, timestamps


def load_model(checkpoint_path: Path, enable_alignment: bool) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega(enable_alignment=enable_alignment, enable_depth=False).eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model.to("cuda")


def predict_extrinsics(
    model: VGGTOmega,
    image_paths: list[Path],
    image_resolution: int,
) -> np.ndarray:
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], image_resolution=image_resolution
    ).to("cuda")
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, _ = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
    return extrinsics[0].detach().float().cpu().numpy()


def extrinsics_to_poses(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_w2c = extrinsics[:, :3, :3]
    translation_w2c = extrinsics[:, :3, 3]
    rotation_c2w = np.transpose(rotation_w2c, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", rotation_c2w, translation_w2c)
    quaternions_xyzw = Rotation.from_matrix(rotation_c2w).as_quat()
    return centers, quaternions_xyzw


def write_6dof_csv(
    output_path: Path,
    frame_indices: list[int],
    timestamps: list[float],
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for frame_idx, timestamp, position, quaternion in zip(
            frame_indices, timestamps, positions, quaternions
        ):
            writer.writerow(
                [
                    frame_idx,
                    f"{timestamp:.9f}",
                    f"{position[0]:.12f}",
                    f"{position[1]:.12f}",
                    f"{position[2]:.12f}",
                    f"{quaternion[0]:.12f}",
                    f"{quaternion[1]:.12f}",
                    f"{quaternion[2]:.12f}",
                    f"{quaternion[3]:.12f}",
                ]
            )


def load_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    with jobs_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    jobs = payload.get("jobs", payload if isinstance(payload, list) else None)
    if not isinstance(jobs, list):
        raise ValueError(f"Job file must contain a list or a dict with a jobs list: {jobs_path}")
    return jobs


def run_job(job: dict[str, Any], model: VGGTOmega, image_resolution: int) -> None:
    video = Path(job["video"]).expanduser().resolve()
    output = Path(job["output"]).expanduser()
    sample_fps = float(job.get("sample_fps", 1.0))
    frame_stride = job.get("frame_stride")
    max_frames = job.get("max_frames")
    keep_frames = job.get("keep_frames")

    if keep_frames is not None:
        frame_root = Path(keep_frames).expanduser()
        frame_root.mkdir(parents=True, exist_ok=True)
        image_paths, frame_indices, timestamps = extract_frames(
            video,
            frame_root,
            sample_fps,
            int(frame_stride) if frame_stride is not None else None,
            int(max_frames) if max_frames is not None else None,
        )
        extrinsics = predict_extrinsics(model, image_paths, image_resolution)
    else:
        with tempfile.TemporaryDirectory(prefix="vggt_omega_frames_") as tmpdir:
            image_paths, frame_indices, timestamps = extract_frames(
                video,
                Path(tmpdir),
                sample_fps,
                int(frame_stride) if frame_stride is not None else None,
                int(max_frames) if max_frames is not None else None,
            )
            extrinsics = predict_extrinsics(model, image_paths, image_resolution)

    positions, quaternions = extrinsics_to_poses(extrinsics)
    write_6dof_csv(output, frame_indices, timestamps, positions, quaternions)
    print(f"Wrote {len(frame_indices)} poses to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega once for a list of chunk videos.")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--enable-alignment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.jobs.expanduser().resolve())
    if not jobs:
        return
    model = load_model(args.checkpoint.expanduser(), args.enable_alignment)
    for job in jobs:
        run_job(job, model, args.image_resolution)


if __name__ == "__main__":
    main()
