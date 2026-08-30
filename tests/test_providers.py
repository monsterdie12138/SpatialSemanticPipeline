from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.providers.base import (  # noqa: E402
    PoseEstimationJob,
    PoseEstimationRequest,
    SceneDetectionRequest,
    SceneDetectionResult,
    SemanticLabelRequest,
)
from spatial_semantic_pipeline.providers.registry import (  # noqa: E402
    get_pose_estimator,
    get_scene_detector,
    get_semantic_labeler,
    register_scene_detector,
)
from spatial_semantic_pipeline.providers.transnet import predictions_to_scenes  # noqa: E402
from spatial_semantic_pipeline.providers.vggt import VGGTOmegaPoseEstimator  # noqa: E402
from spatial_semantic_pipeline.providers.vlm import OpenAICompatibleSemanticLabeler  # noqa: E402


class FakeSceneDetector:
    def detect(self, request: SceneDetectionRequest) -> SceneDetectionResult:
        return SceneDetectionResult(cut_edges=[[4, 5]], detector="fake")


class ProviderTest(unittest.TestCase):
    def test_stages_do_not_import_concrete_model_runners(self) -> None:
        package = Path(__file__).resolve().parents[1] / "src" / "spatial_semantic_pipeline"
        stage_sources = "\n".join(
            (package / "stages" / name).read_text(encoding="utf-8")
            for name in ("detect_scenes.py", "estimate_trajectories.py", "label_semantics.py")
        )
        self.assertNotIn("transnetv2_pytorch", stage_sources)
        self.assertNotIn("video_to_6dof_batch.py", stage_sources)
        self.assertNotIn("retry_call_vlm_api", stage_sources)

    def test_default_provider_selection_preserves_existing_tools(self) -> None:
        self.assertEqual(type(get_scene_detector({})).__name__, "TransNetV2SceneDetector")
        self.assertEqual(type(get_pose_estimator({})).__name__, "VGGTOmegaPoseEstimator")
        self.assertEqual(type(get_semantic_labeler({})).__name__, "OpenAICompatibleSemanticLabeler")

    def test_registered_provider_can_replace_builtin_without_stage_changes(self) -> None:
        register_scene_detector("test_fake", FakeSceneDetector)
        provider = get_scene_detector({"providers": {"scene_detector": "test_fake"}})
        result = provider.detect(
            SceneDetectionRequest(
                config={}, video_path=Path("video.mp4"), height=480, fps=None, threshold=0.5
            )
        )
        self.assertEqual(result, SceneDetectionResult(cut_edges=[[4, 5]], detector="fake"))

    def test_dotted_provider_can_be_loaded_from_config(self) -> None:
        provider = get_scene_detector(
            {"providers": {"scene_detector": "test_providers:FakeSceneDetector"}}
        )
        self.assertIsInstance(provider, FakeSceneDetector)

    def test_transnet_scene_conversion_is_unchanged(self) -> None:
        predictions = np.array([0.0, 0.8, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)
        self.assertEqual(predictions_to_scenes(predictions).tolist(), [[0, 1], [2, 4], [5, 5]])

    def test_vggt_provider_preserves_jobs_file_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            captured = {}

            def fake_run(
                tool_root: Path,
                checkpoint: Path,
                jobs_file: Path,
                image_resolution: int,
                enable_alignment: bool,
            ) -> None:
                captured["tool_root"] = tool_root
                captured["checkpoint"] = checkpoint
                captured["payload"] = json.loads(jobs_file.read_text(encoding="utf-8"))
                captured["image_resolution"] = image_resolution
                captured["enable_alignment"] = enable_alignment

            request = PoseEstimationRequest(
                config={
                    "vggt_tool_root": str(work_dir / "tool"),
                    "vggt_checkpoint": str(work_dir / "model.pt"),
                },
                jobs=(
                    PoseEstimationJob(
                        input_video=work_dir / "chunk.mp4",
                        output_pose=work_dir / "poses.csv",
                        sample_fps=1.0,
                    ),
                ),
                work_dir=work_dir,
            )
            with patch("spatial_semantic_pipeline.providers.vggt.run_vggt_batch", side_effect=fake_run):
                VGGTOmegaPoseEstimator().estimate(request)

            self.assertEqual(
                captured["payload"],
                {
                    "jobs": [
                        {
                            "video": str(work_dir / "chunk.mp4"),
                            "output": str(work_dir / "poses.csv"),
                            "sample_fps": 1.0,
                        }
                    ]
                },
            )
            self.assertFalse((work_dir / "vggt_jobs.json").exists())
            self.assertEqual(captured["image_resolution"], 512)
            self.assertFalse(captured["enable_alignment"])

    def test_vlm_provider_preserves_retry_call(self) -> None:
        with patch(
            "spatial_semantic_pipeline.providers.vlm.retry_call_vlm_api",
            return_value=("semantic", {"id": "response"}),
        ) as call:
            result = OpenAICompatibleSemanticLabeler().label(
                SemanticLabelRequest(config={"model": "same"}, prompt="prompt", clip=Path("clip.mp4"))
            )
        self.assertEqual(result.text, "semantic")
        self.assertEqual(result.raw_response, {"id": "response"})
        call.assert_called_once_with({"model": "same"}, "prompt", Path("clip.mp4"))


if __name__ == "__main__":
    unittest.main()
