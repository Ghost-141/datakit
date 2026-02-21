"""Operations for merging multiple YOLO datasets into one."""

from ..formats.yolo import YoloFormatHandler


class YoloDatasetMerger:
    """Facade for merging multiple YOLO datasets into a single output."""

    def __init__(self):
        """Initialize the YOLO dataset merge handler."""
        self._handler = YoloFormatHandler()

    def merge(self, datasets: list[str], out_dir: str):
        """Merge datasets into a single dataset directory.

        Args:
            datasets: List of dataset root directories.
            out_dir: Output directory for the merged dataset.
        """
        self._handler.merge_datasets(datasets, out_dir)


def merge_datasets(datasets: list[str], out_dir: str):
    """Convenience function to merge datasets into one output directory."""
    YoloDatasetMerger().merge(datasets, out_dir)
