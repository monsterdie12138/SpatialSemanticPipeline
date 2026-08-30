from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.config import load_config, load_yaml


class ConfigTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config()
        self.assertEqual(config["paths"]["output_root"], "output")
        self.assertEqual(config["prompt_file"], "prompts/semantic_with_hint.txt")
        self.assertEqual(config["judge"]["min_poses"], 5)
        self.assertEqual(config["hint"]["min_used_poses"], 60)
        self.assertEqual(config["vggt"]["image_resolution"], 512)

    def test_yaml_loader_preserves_nested_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("paths:\n  cache_root: cache\nworkers: 2\n", encoding="utf-8")
            self.assertEqual(load_yaml(path), {"paths": {"cache_root": "cache"}, "workers": 2})


if __name__ == "__main__":
    unittest.main()
