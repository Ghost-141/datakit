"""Operations for visualizing labeled YOLO samples."""

import os
from pathlib import Path

from ..formats.yolo import YoloFormatHandler


def _normalize_dir_input(path_value: str) -> str:
    """Normalize user-provided directory paths for common Windows CLI mistakes.

    Args:
        path_value: Raw path value from user or CLI input.

    Returns:
        A normalized path string that resolves common leading-slash mistakes.

    Example:
        ```python
        from datakit.operations.visualize import _normalize_dir_input
        fixed = _normalize_dir_input("/Users/you/dataset/train/images")
        ```
    """
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path)

    # On Windows, a leading "/" can accidentally point to drive root.
    if os.name == "nt" and path_value and path_value[0] in {"/", "\\"}:
        stripped = path_value.lstrip("/\\")
        if stripped:
            candidate = Path(stripped).expanduser()
            if candidate.exists():
                return str(candidate)

    return path_value


class YoloVisualizer:
    """Facade for visualizing YOLO datasets with labels."""

    def __init__(self):
        """Initialize a visualization facade backed by ``YoloFormatHandler``.

        Example:
            ```python
            from datakit.operations.visualize import YoloVisualizer
            visualizer = YoloVisualizer()
            ```
        """
        self._handler = YoloFormatHandler()

    def plot_random_samples(
        self,
        images_dir: str,
        labels_dir: str,
        names: list[str] | None = None,
        n: int = 10,
        seed: int = 2,
        cols: int | None = None,
        tile_size: tuple[int, int] = (640, 640),
    ):
        """Plot a random sample grid with bounding box overlays.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing label files.
            names: Optional list of class names.
            n: Number of images to sample.
            seed: Random seed for sampling.
            cols: Column count for the grid (auto if None).
            tile_size: Target tile size (width, height).

        Example:
            ```python
            visualizer = YoloVisualizer()
            visualizer.plot_random_samples(
                images_dir="new_dataset/train/images",
                labels_dir="new_dataset/train/labels",
                names=["car", "person"],
                n=12,
                seed=2,
            )
            ```
        """
        images_dir = _normalize_dir_input(images_dir)
        labels_dir = _normalize_dir_input(labels_dir)

        self._handler.visualize_samples(
            images_dir=images_dir,
            labels_dir=labels_dir,
            names=names,
            n=n,
            seed=seed,
            cols=cols,
            tile_size=tile_size,
        )


def plot_random_samples(
    images_dir: str,
    labels_dir: str,
    names: list[str] | None = None,
    n: int = 10,
    seed: int = 2,
    cols: int | None = None,
    tile_size: tuple[int, int] = (640, 640),
):
    """Visualize random labeled samples.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing label files.
        names: Optional list of class names.
        n: Number of images to sample.
        seed: Random seed for sampling.
        cols: Column count for the grid (auto if None).
        tile_size: Target tile size (width, height).

    Example:
        ```python
        from datakit import plot_random_samples
        plot_random_samples(
            images_dir="new_dataset/train/images",
            labels_dir="new_dataset/train/labels",
            names=["car", "person"],
            n=9,
        )
        ```
    """
    YoloVisualizer().plot_random_samples(
        images_dir=images_dir,
        labels_dir=labels_dir,
        names=names,
        n=n,
        seed=seed,
        cols=cols,
        tile_size=tile_size,
    )
