"""Command-line entry point for the standard pipeline workflow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatial-semantic-pipeline")
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    subparsers.add_parser("run", help="Run the standard 01-08 workflow.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, workflow_args = build_parser().parse_known_args(argv)
    from .workflows import standard as workflow
    previous_argv = sys.argv
    try:
        sys.argv = [f"spatial-semantic-pipeline {args.workflow}", *workflow_args]
        return int(workflow.main())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
