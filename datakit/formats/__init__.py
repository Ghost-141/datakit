"""Format handler registry and exports."""

from __future__ import annotations

from ..core.dataset import DatasetFormatHandler
from .yolo import YoloFormatHandler


def get_format_handler(format_name: str) -> DatasetFormatHandler:
    """Return a format handler instance by name.

    Args:
        format_name: Human-friendly format identifier (e.g., "yolo").

    Raises:
        ValueError: If the format name is not supported.
    """
    handlers = {
        "yolo": YoloFormatHandler,
    }

    normalized = format_name.strip().lower()
    if normalized not in handlers:
        supported = ", ".join(sorted(handlers.keys()))
        raise ValueError(f"Unsupported format '{format_name}'. Supported formats: {supported}")

    return handlers[normalized]()


__all__ = [
    "DatasetFormatHandler",
    "YoloFormatHandler",
    "get_format_handler",
]
