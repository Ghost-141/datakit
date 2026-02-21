"""Operations for merging class labels in YOLO datasets."""

from ..formats.yolo import YoloFormatHandler


class YoloClassMerger:
    """Facade for merging multiple class names into a target class."""

    def __init__(self):
        """Initialize the YOLO class merge handler."""
        self._handler = YoloFormatHandler()

    def merge(
        self,
        dataset_dir: str,
        merge_from_names: list[str],
        merge_into_name: str,
        update_yaml: bool = True,
    ):
        """Merge multiple source classes into a target class.

        Args:
            dataset_dir: Dataset root containing labels and data.yaml.
            merge_from_names: Class names to merge into the target.
            merge_into_name: Target class name to merge into.
            update_yaml: Whether to update data.yaml names in place.
        """
        self._handler.merge_classes(
            dataset_dir=dataset_dir,
            merge_from_names=merge_from_names,
            merge_into_name=merge_into_name,
            update_yaml=update_yaml,
        )


def merge_classes(
    dataset_dir: str,
    merge_from_names: list[str],
    merge_into_name: str,
    update_yaml: bool = True,
):
    """Convenience function to merge multiple class names into a target class."""
    YoloClassMerger().merge(dataset_dir, merge_from_names, merge_into_name, update_yaml)
