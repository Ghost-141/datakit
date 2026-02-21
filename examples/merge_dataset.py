"""Example: merge multiple datasets into one."""

from datakit import merge_datasets


if __name__ == "__main__":
    datasets = ["/path/dataset_1", "/path/dataset_2"]
    out_dir = "/path/output"
    merge_datasets(datasets, out_dir)
