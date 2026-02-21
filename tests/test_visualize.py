from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from datakit import plot_random_samples


def _deps_available() -> bool:
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ["matplotlib", "numpy", "PIL"]
    )


def test_plot_random_samples_handles_dependencies(tmp_path: Path, monkeypatch) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not _deps_available():
        with pytest.raises(ImportError):
            plot_random_samples(str(images_dir), str(labels_dir), n=1)
        return

    os.environ["MPLBACKEND"] = "Agg"

    from PIL import Image

    img_path = images_dir / "img1.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img_path)
    (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    plot_random_samples(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        names=["box"],
        n=1,
        cols=1,
        tile_size=(10, 10),
    )
