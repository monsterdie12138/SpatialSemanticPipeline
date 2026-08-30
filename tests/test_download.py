from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_semantic_pipeline.stages.download import build_yt_dlp_cmd, run_yt_dlp_once


class DownloadTest(unittest.TestCase):
    def test_command_has_no_node_or_proxy_options(self) -> None:
        command = build_yt_dlp_cmd(
            "https://example.invalid/video",
            Path("video.%(ext)s"),
            480,
            30.0,
            ["node"],
        )
        self.assertNotIn("--proxy", command)
        self.assertNotIn("--socket-timeout", command)
        self.assertEqual(command[-1], "https://example.invalid/video")

    def test_optional_download_outputs_and_format_are_configurable(self) -> None:
        command = build_yt_dlp_cmd(
            "https://example.invalid/video",
            Path("video.%(ext)s"),
            480,
            30.0,
            format_selector="bv[height<=360]",
            write_description=False,
            write_thumbnail=False,
        )
        self.assertNotIn("--write-description", command)
        self.assertNotIn("--write-thumbnail", command)
        self.assertEqual(command[command.index("-f") + 1], "bv[height<=360]")

    @mock.patch("spatial_semantic_pipeline.stages.download.pipeline_env", return_value={})
    @mock.patch("spatial_semantic_pipeline.stages.download.subprocess.run")
    def test_first_download_failure_is_propagated(self, run: mock.Mock, _pipeline_env: mock.Mock) -> None:
        failure = subprocess.CalledProcessError(1, ["yt-dlp"])
        run.side_effect = failure
        with self.assertRaises(subprocess.CalledProcessError):
            run_yt_dlp_once(
                "https://example.invalid/video",
                Path("video.%(ext)s"),
                480,
                30.0,
                {},
            )
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
