from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.stages.judge_trajectory import norm3, percentile


class JudgeTest(unittest.TestCase):
    def test_existing_numeric_helpers(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)
        self.assertEqual(norm3((3.0, 4.0, 0.0)), 5.0)


if __name__ == "__main__":
    unittest.main()

