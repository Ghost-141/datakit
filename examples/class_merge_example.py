"""Example: merge multiple class names into a target class."""

from datakit import merge_classes


if __name__ == "__main__":
    dataset_dir = "/path/dataset"
    merge_from = ["Backpack", "Backpacks"]
    merge_to = "bag"
    merge_classes(dataset_dir, merge_from, merge_to, update_yaml=True)
