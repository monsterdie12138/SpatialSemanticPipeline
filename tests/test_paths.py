from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_semantic_pipeline.paths import output_root, path_from_config


class PathsTest(unittest.TestCase):
    def test_relative_paths_resolve_from_project_root(self) -> None:
        config = {"paths": {"output_root": "output-test"}}
        self.assertEqual(output_root(config), (PROJECT_ROOT / "output-test").resolve())
        self.assertEqual(
            path_from_config(config, "missing", PROJECT_ROOT / "fallback"),
            (PROJECT_ROOT / "fallback").resolve(),
        )


if __name__ == "__main__":
    unittest.main()

