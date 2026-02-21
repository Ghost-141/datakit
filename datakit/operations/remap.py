"""Operations for remapping class IDs in YOLO datasets."""

from ..formats.yolo import YoloFormatHandler


class YoloClassRemapper:
    """Facade for remapping class IDs and names in YOLO datasets."""

    def __init__(self):
        """Initialize a remap facade backed by ``YoloFormatHandler``.

        Example:
            ```python
            from datakit.operations.remap import YoloClassRemapper
            remapper = YoloClassRemapper()
            ```
        """
        self._handler = YoloFormatHandler()

    def remap(self, dataset_dir: str, new_names: list[str], id_mapping: dict[int, int]):
        """Remap class IDs and update class names.

        Args:
            dataset_dir: Dataset root containing labels and data.yaml.
            new_names: New class names in final ID order.
            id_mapping: Mapping from old class IDs to new IDs.

        Example:
            ```python
            remapper = YoloClassRemapper()
            remapper.remap("new_dataset", ["bag", "person"], {0: 0, 1: 0, 2: 1})
            ```
        """
        self._handler.remap_dataset(
            dataset_dir=dataset_dir,
            new_names=new_names,
            id_mapping=id_mapping,
        )


def remap_dataset(dataset_dir: str, new_names: list[str], id_mapping: dict[int, int]):
    """Remap class IDs in a dataset.

    Args:
        dataset_dir: Dataset root containing labels and data.yaml.
        new_names: New class names in final ID order.
        id_mapping: Mapping from old class IDs to new IDs.

    Example:
        ```python
        from datakit import remap_dataset
        remap_dataset("new_dataset", ["bag", "person"], {0: 0, 1: 0, 2: 1})
        ```
    """
    YoloClassRemapper().remap(dataset_dir, new_names, id_mapping)
