from __future__ import annotations

from pathlib import Path

import pytest

from datakit import remap_dataset

from .helpers import create_image_and_label, load_names, write_data_yaml


def test_remap_dataset_updates_labels_and_yaml(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    write_data_yaml(ds, ["a", "b"], ["train"])
    create_image_and_label(
        ds,
        "train",
        "img1",
        ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"],
    )

    remap_dataset(str(ds), ["b", "a"], {0: 1, 1: 0})

    label_path = ds / "labels" / "train" / "img1.txt"
    new_ids = [int(line.split()[0]) for line in label_path.read_text().splitlines()]
    assert new_ids == [1, 0]
    assert load_names(ds / "data.yaml") == ["b", "a"]


def test_remap_dataset_rejects_out_of_range_new_ids_before_writing(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    write_data_yaml(ds, ["a", "b"], ["train"])
    create_image_and_label(ds, "train", "img1", ["0 0.5 0.5 0.1 0.1"])

    label_path = ds / "labels" / "train" / "img1.txt"
    before_label = label_path.read_text(encoding="utf-8")
    before_names = load_names(ds / "data.yaml")

    with pytest.raises(ValueError, match=r"out of range"):
        remap_dataset(str(ds), ["a", "b"], {0: 2})

    assert label_path.read_text(encoding="utf-8") == before_label
    assert load_names(ds / "data.yaml") == before_names


def test_remap_dataset_missing_id_does_not_mutate_any_files(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    write_data_yaml(ds, ["a", "b"], ["train"])
    create_image_and_label(ds, "train", "img1", ["0 0.5 0.5 0.1 0.1"])
    create_image_and_label(ds, "train", "img2", ["1 0.5 0.5 0.1 0.1"])

    label_path_1 = ds / "labels" / "train" / "img1.txt"
    label_path_2 = ds / "labels" / "train" / "img2.txt"
    before_label_1 = label_path_1.read_text(encoding="utf-8")
    before_label_2 = label_path_2.read_text(encoding="utf-8")
    before_names = load_names(ds / "data.yaml")

    with pytest.raises(ValueError, match=r"Class ID 1 not in mapping"):
        remap_dataset(str(ds), ["a", "b"], {0: 0})

    assert label_path_1.read_text(encoding="utf-8") == before_label_1
    assert label_path_2.read_text(encoding="utf-8") == before_label_2
    assert load_names(ds / "data.yaml") == before_names
