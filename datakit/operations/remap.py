"""Operations for remapping class IDs in YOLO datasets."""

from ..formats.yolo import YoloFormatHandler


class YoloClassRemapper:
    """Facade for remapping class IDs and names in YOLO datasets."""

    def __init__(self):
        """Initialize the YOLO class remap handler."""
        self._handler = YoloFormatHandler()

    def remap(self, dataset_dir: str, new_names: list[str], id_mapping: dict[int, int]):
        """Remap class IDs and update class names.

        Args:
            dataset_dir: Dataset root containing labels and data.yaml.
            new_names: New class names in final ID order.
            id_mapping: Mapping from old class IDs to new IDs.
        """
        self._handler.remap_dataset(
            dataset_dir=dataset_dir,
            new_names=new_names,
            id_mapping=id_mapping,
        )


def remap_dataset(dataset_dir: str, new_names: list[str], id_mapping: dict[int, int]):
    """Convenience function to remap class IDs in a dataset."""
    YoloClassRemapper().remap(dataset_dir, new_names, id_mapping)
