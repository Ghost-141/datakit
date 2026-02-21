from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from datakit import merge_datasets

from .helpers import create_image_and_label, load_names, write_data_yaml


def test_merge_datasets_offsets_and_yaml(tmp_path: Path) -> None:
    ds1 = tmp_path / "ds1"
    ds2 = tmp_path / "ds2"
    out_dir = tmp_path / "out"

    write_data_yaml(ds1, ["cat", "dog"], ["train"])
    create_image_and_label(
        ds1,
        "train",
        "img1",
        ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"],
    )

    write_data_yaml(ds2, ["bird"], ["train"])
    create_image_and_label(
        ds2,
        "train",
        "img2",
        ["0 0.1 0.1 0.2 0.2"],
    )

    merge_datasets([str(ds1), str(ds2)], str(out_dir))

    merged_yaml = out_dir / "data.yaml"
    assert merged_yaml.exists()
    assert load_names(merged_yaml) == ["cat", "dog", "bird"]
    with open(merged_yaml, "r", encoding="utf-8") as handle:
        merged_data = yaml.safe_load(handle)
    assert merged_data["train"] == "train/images"
    assert "val" not in merged_data
    assert "test" not in merged_data

    ds1_label = out_dir / "train" / "labels" / "ds1__img1.txt"
    ds2_label = out_dir / "train" / "labels" / "ds2__img2.txt"
    assert ds1_label.exists()
    assert ds2_label.exists()

    ds1_ids = [int(line.split()[0]) for line in ds1_label.read_text().splitlines()]
    ds2_ids = [int(line.split()[0]) for line in ds2_label.read_text().splitlines()]
    assert ds1_ids == [0, 1]
    assert ds2_ids == [2]


def test_merge_datasets_raises_when_no_images_copied(tmp_path: Path) -> None:
    ds1 = tmp_path / "ds1"
    out_dir = tmp_path / "out"

    write_data_yaml(ds1, ["cat"], ["train"])

    with pytest.raises(RuntimeError, match="No images were copied during merge"):
        merge_datasets([str(ds1)], str(out_dir))
