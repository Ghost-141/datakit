"""Core dataset abstractions."""

from abc import ABC, abstractmethod


class DatasetFormatHandler(ABC):
    """Base interface for dataset-format-specific operations."""

    format_name: str

    @abstractmethod
    def merge_datasets(self, datasets: list[str], out_dir: str):
        """Merge multiple datasets into one dataset."""

    @abstractmethod
    def merge_classes(
        self,
        dataset_dir: str,
        merge_from_names: list[str],
        merge_into_name: str,
        update_yaml: bool = True,
    ):
        """Merge multiple classes into a target class."""

    @abstractmethod
    def remap_dataset(
        self,
        dataset_dir: str,
        new_names: list[str],
        id_mapping: dict[int, int],
    ):
        """Remap class IDs and class names."""

    @abstractmethod
    def visualize_samples(
        self,
        images_dir: str,
        labels_dir: str,
        names: list[str] | None = None,
        n: int = 10,
        seed: int = 2,
        cols: int | None = None,
        tile_size: tuple[int, int] = (640, 640),
    ):
        """Visualize random labeled samples."""


__all__ = ["DatasetFormatHandler"]
