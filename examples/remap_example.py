"""Example: remap class IDs and names in a dataset."""

from datakit import remap_dataset


if __name__ == "__main__":
    dataset_dir = "/path/dataset"
    new_names = ["bag", "person"]
    id_mapping = {0: 0, 1: 0, 2: 1}
    remap_dataset(dataset_dir, new_names, id_mapping)
