"""Built-in VGGT-Omega pose estimator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..context import ROOT, path_from_config, pipeline_env
from .base import PoseEstimationRequest

VGGT_INTERNAL_IMAGE_RESOLUTION = 512
VGGT_ENABLE_ALIGNMENT = False


def run_vggt_batch(
    tool_root: Path,
    checkpoint: Path,
    jobs_file: Path,
    image_resolution: int = VGGT_INTERNAL_IMAGE_RESOLUTION,
    enable_alignment: bool = VGGT_ENABLE_ALIGNMENT,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "spatial_semantic_pipeline.providers.vggt_runner",
        "--jobs",
        str(jobs_file),
        "--checkpoint",
        str(checkpoint),
        "--image-resolution",
        str(image_resolution),
    ]
    if enable_alignment:
        cmd.append("--enable-alignment")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(tool_root)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    env = pipeline_env(
        None,
        {
            "PYTHONPATH": pythonpath,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    subprocess.run(cmd, cwd=str(tool_root), check=True, env=env)


class VGGTOmegaPoseEstimator:
    def estimate(self, request: PoseEstimationRequest) -> None:
        provider_config = request.config.get("vggt") or {}
        if not isinstance(provider_config, dict):
            raise ValueError("Config field 'vggt' must be a mapping")
        tool_root = path_from_config(request.config, "vggt_tool_root", ROOT / "tools" / "VGGT-Omega")
        checkpoint = path_from_config(
            request.config,
            "vggt_checkpoint",
            tool_root / "checkpoints" / "VGGT-Omega-1B-512" / "model.pt",
        )
        jobs_file = request.work_dir / "vggt_jobs.json"
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        with jobs_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "jobs": [
                        {
                            "video": str(job.input_video),
                            "output": str(job.output_pose),
                            "sample_fps": job.sample_fps,
                        }
                        for job in request.jobs
                    ]
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        image_resolution = int(provider_config.get("image_resolution", VGGT_INTERNAL_IMAGE_RESOLUTION))
        alignment_value = provider_config.get("enable_alignment", VGGT_ENABLE_ALIGNMENT)
        enable_alignment = alignment_value if isinstance(alignment_value, bool) else str(alignment_value).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        run_vggt_batch(
            tool_root,
            checkpoint,
            jobs_file,
            image_resolution=image_resolution,
            enable_alignment=enable_alignment,
        )
        jobs_file.unlink(missing_ok=True)


__all__ = [
    "VGGT_ENABLE_ALIGNMENT",
    "VGGT_INTERNAL_IMAGE_RESOLUTION",
    "VGGTOmegaPoseEstimator",
    "run_vggt_batch",
]
