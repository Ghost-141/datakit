from __future__ import annotations

from pathlib import Path

from datakit import merge_classes

from .helpers import create_image_and_label, load_names, write_data_yaml


def _read_label_ids(label_path: Path) -> list[int]:
    return [int(line.split()[0]) for line in label_path.read_text().splitlines()]


def test_merge_classes_updates_labels_and_yaml(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    write_data_yaml(ds, ["a", "b", "c"], ["train"])
    create_image_and_label(
        ds,
        "train",
        "img1",
        ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "2 0.3 0.3 0.1 0.1"],
    )

    merge_classes(str(ds), ["a"], "b", update_yaml=True)

    label_path = ds / "labels" / "train" / "img1.txt"
    assert _read_label_ids(label_path) == [1, 1, 2]
    assert load_names(ds / "data.yaml") == ["b", "b", "c"]


def test_merge_classes_without_yaml_update(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    write_data_yaml(ds, ["a", "b"], ["train"])
    create_image_and_label(ds, "train", "img1", ["0 0.5 0.5 0.1 0.1"])

    merge_classes(str(ds), ["a"], "b", update_yaml=False)

    label_path = ds / "labels" / "train" / "img1.txt"
    assert _read_label_ids(label_path) == [1]
    assert load_names(ds / "data.yaml") == ["a", "b"]
