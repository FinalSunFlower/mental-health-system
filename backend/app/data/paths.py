"""Stable paths for optional, locally acquired benchmark inputs."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"

__all__ = ["REPOSITORY_ROOT", "RAW_DIR"]
