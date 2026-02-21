from __future__ import annotations

from pathlib import Path

import yaml


def write_data_yaml(root: Path, names: list[str], splits: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(root),
        "names": names,
        "nc": len(names),
    }
    for split in splits:
        data[split] = f"images/{split}"

    with open(root / "data.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def create_image_and_label(
    root: Path, split: str, stem: str, label_lines: list[str]
) -> None:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / f"{stem}.jpg").write_bytes(b"")
    with open(labels_dir / f"{stem}.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(label_lines) + "\n")


def load_names(yaml_path: Path) -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)
