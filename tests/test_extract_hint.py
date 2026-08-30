from __future__ import annotations

import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.stages.extract_hint import (
    HINT_DEFAULTS,
    apply_hint_settings,
    angle_delta,
    hint_settings,
    quat_to_forward_heading,
)


class ExtractHintTest(unittest.TestCase):
    def test_existing_heading_helpers(self) -> None:
        self.assertAlmostEqual(quat_to_forward_heading(0.0, 0.0, 0.0, 1.0), math.pi / 2)
        self.assertAlmostEqual(angle_delta(-math.pi + 0.1, math.pi - 0.1), 0.2)

    def test_hint_defaults_preserve_original_pipeline_values(self) -> None:
        self.assertEqual(hint_settings({}), HINT_DEFAULTS)

    def test_hint_config_overrides_only_named_values(self) -> None:
        settings = hint_settings({"hint": {"min_turn_deg": 65, "left_sign": "negative"}})
        self.assertEqual(settings["min_turn_deg"], 65.0)
        self.assertEqual(settings["left_sign"], "negative")
        self.assertEqual(settings["stop_speed_ratio"], 0.15)

    def test_invalid_left_sign_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hint_settings({"hint": {"left_sign": "clockwise"}})

    def test_cli_value_takes_priority_over_hint_config(self) -> None:
        args = Namespace(**{key: None for key in HINT_DEFAULTS})
        args.min_turn_deg = 75.0
        apply_hint_settings(args, {"hint": {"min_turn_deg": 65, "stop_speed_ratio": 0.2}})
        self.assertEqual(args.min_turn_deg, 75.0)
        self.assertEqual(args.stop_speed_ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
