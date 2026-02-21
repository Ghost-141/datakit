"""Operations for merging multiple YOLO datasets into one."""

from ..formats.yolo import YoloFormatHandler


class YoloDatasetMerger:
    """Facade for merging multiple YOLO datasets into a single output."""

    def __init__(self):
        """Initialize a merger facade backed by ``YoloFormatHandler``.

        Example:
            ```python
            from datakit.operations.merge import YoloDatasetMerger
            merger = YoloDatasetMerger()
            ```
        """
        self._handler = YoloFormatHandler()

    def merge(self, datasets: list[str], out_dir: str):
        """Merge datasets into a single dataset directory.

        Args:
            datasets: List of dataset root directories.
            out_dir: Output directory for the merged dataset.

        Example:
            ```python
            merger = YoloDatasetMerger()
            merger.merge(["Drones", "R2P2.v2-raw-images.yolov11"], "new_dataset")
            ```
        """
        self._handler.merge_datasets(datasets, out_dir)


def merge_datasets(datasets: list[str], out_dir: str):
    """Merge multiple datasets into one output directory.

    Args:
        datasets: List of dataset root directories.
        out_dir: Output directory for the merged dataset.

    Example:
        ```python
        from datakit import merge_datasets
        merge_datasets(["Drones", "R2P2.v2-raw-images.yolov11"], "new_dataset")
        ```
    """
    YoloDatasetMerger().merge(datasets, out_dir)
