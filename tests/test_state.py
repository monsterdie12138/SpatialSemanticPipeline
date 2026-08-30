from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.core.state import load_videos, save_videos


class StateTest(unittest.TestCase):
    def test_video_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"metadata_root": directory}}
            save_videos(config, {"schema_version": 1, "videos": [{"video_id": "v1"}]})
            state = load_videos(config)
            self.assertEqual(state["videos"][0]["video_id"], "v1")
            self.assertIn("updated_at", state)


if __name__ == "__main__":
    unittest.main()

