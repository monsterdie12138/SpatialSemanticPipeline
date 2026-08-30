from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.core.stages import Status, stage_module  # noqa: E402


class StageContractTest(unittest.TestCase):
    def test_default_stage_module_preserves_existing_workflow(self) -> None:
        self.assertEqual(
            stage_module({}, "04_estimate_trajectories.py"),
            "spatial_semantic_pipeline.stages.estimate_trajectories",
        )
        self.assertEqual(Status.POSES_ESTIMATED, "poses_estimated")

    def test_stage_module_can_be_replaced_by_config(self) -> None:
        config = {"stages": {"estimate_trajectories": "example.custom_pose_stage"}}
        self.assertEqual(stage_module(config, "04_estimate_trajectories.py"), "example.custom_pose_stage")

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            stage_module({}, "unknown.py")


if __name__ == "__main__":
    unittest.main()
