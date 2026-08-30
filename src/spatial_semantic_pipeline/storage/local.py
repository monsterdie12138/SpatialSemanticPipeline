"""Local output storage public API."""

from ..context import output_root
from ..stages.publish import merge_staging_tree, stage_copy

__all__ = ["merge_staging_tree", "output_root", "stage_copy"]

