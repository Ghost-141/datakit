"""Shared utilities for datakit core."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its ``Path``.

    Args:
        path: Directory path to create.

    Returns:
        The created or existing directory as a ``Path`` object.

    Example:
        ```python
        from datakit.core.utils import ensure_dir
        out_dir = ensure_dir("new_dataset/train/images")
        ```
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["ensure_dir"]
