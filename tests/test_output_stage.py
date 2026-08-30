from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spatial_semantic_pipeline.context import (  # noqa: E402
    cleanup_successful_runtime,
    output_root,
    path_from_config,
    write_json,
)
from spatial_semantic_pipeline.stages import publish as output_stage  # noqa: E402


class OutputStageTest(unittest.TestCase):
    def test_relative_paths_are_project_relative(self) -> None:
        config = {"paths": {"output_root": "build/final"}}
        self.assertEqual(output_root(config), (PROJECT_ROOT / "build/final").resolve())
        self.assertEqual(
            path_from_config(config, "missing", PROJECT_ROOT / "fallback"),
            (PROJECT_ROOT / "fallback").resolve(),
        )

    def test_standard_export_writes_local_package_then_cleans_runtime(self) -> None:
        stage = output_stage
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cache = root / "cache"
            metadata = root / "metadata"
            output = root / "output"
            channel = "example_channel"
            video_id = "video_123"

            source_video = cache / "downloads" / video_id / "original.mp4"
            info_file = cache / "downloads" / video_id / "info.json"
            chunk_file = cache / "chunks" / channel / video_id / "scene_0000" / "chunk_0000.mp4"
            pose_file = cache / "poses" / video_id / "pose_chunk_0000.csv"
            semantic_file = cache / "semantics" / channel / video_id / "scene_0000" / "chunk_0000.txt"
            for path, content in (
                (source_video, b"video"),
                (info_file, b"{}"),
                (chunk_file, b"chunk"),
                (pose_file, b"frame,x,y,z\n"),
                (semantic_file, b"semantic"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            config = {
                "paths": {
                    "cache_root": str(cache),
                    "metadata_root": str(metadata),
                    "output_root": str(output),
                },
                "videos_root": str(cache / "chunks"),
                "hints_root": str(cache / "hints"),
                "semantics_root": str(cache / "semantics"),
                "chunk_duration_sec": 60,
            }
            write_json(
                metadata / "pipeline_state.json",
                {
                    "schema_version": 1,
                    "videos": [
                        {
                            "video_id": video_id,
                            "channel": channel,
                            "hf_channel": channel,
                            "cache_video_path": str(source_video),
                            "cache_info_path": str(info_file),
                            "width": 640,
                            "height": 480,
                            "fps": 1,
                            "frame_count": 60,
                            "duration_sec": 60,
                            "status": "semantics_labeled",
                        }
                    ],
                },
            )
            write_json(
                metadata / "pipeline_scenes.json",
                {
                    "scenes": [
                        {
                            "video_id": video_id,
                            "scene_index": 0,
                            "source_start_sec": 0,
                            "source_end_sec": 60,
                            "pose_chunks": [
                                {
                                    "chunk_index": 0,
                                    "logical_start_sec": 0,
                                    "logical_end_sec": 60,
                                    "pose_file": str(pose_file),
                                    "semantic_file": str(semantic_file),
                                    "chunk_video_file": str(chunk_file),
                                }
                            ],
                        }
                    ]
                },
            )

            counts = stage.export_standard(config, video_id)

            self.assertEqual(counts, {"poses": 1, "semantics": 1, "chunks": 1})
            self.assertTrue((output / "info" / channel / f"{video_id}.json").is_file())
            self.assertTrue((output / "scenes" / channel / video_id / "scene_0000" / pose_file.name).is_file())
            self.assertTrue(
                (output / "semantics" / channel / video_id / "scene_0000" / semantic_file.name).is_file()
            )
            output_chunk = output / "chunks" / channel / video_id / "scene_0000" / chunk_file.name
            self.assertEqual(output_chunk.read_bytes(), b"chunk")
            self.assertEqual(json.loads((metadata / "pipeline_state.json").read_text())["videos"], [])
            self.assertEqual(json.loads((metadata / "pipeline_scenes.json").read_text())["scenes"], [])
            self.assertFalse((output / ".staging").exists())

            cleanup_successful_runtime(config)

            self.assertFalse(cache.exists())
            self.assertFalse(metadata.exists())
            self.assertFalse((output / ".staging").exists())
            self.assertEqual(output_chunk.read_bytes(), b"chunk")


if __name__ == "__main__":
    unittest.main()
