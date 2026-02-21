"""YOLO-format dataset handler implementation."""

import math
import random
import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from rich.align import Align
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..core.dataset import DatasetFormatHandler


class YoloFormatHandler(DatasetFormatHandler):
    """DatasetFormatHandler implementation for YOLO-format datasets."""

    format_name = "yolo"
    _img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _img_splits = ["train", "val", "valid", "test"]

    def merge_datasets(self, datasets: list[str], out_dir: str):
        """Merge multiple YOLO datasets into a single output directory.

        Args:
            datasets: List of dataset root directories containing data.yaml.
            out_dir: Output directory for the merged dataset.

        Example:
            ```python
            handler = YoloFormatHandler()
            handler.merge_datasets(["Drones", "R2P2.v2-raw-images.yolov11"], "new_dataset")
            ```
        """
        out_root = Path(out_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        splits = ["train", "val", "test"]

        merged_names = []
        offset = 0
        splits_present = set()
        yamls = []
        total_copied_images = 0

        for ds in datasets:
            ds_root = Path(ds).resolve()
            yaml_path = ds_root / "data.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Missing data.yaml in {ds_root}")

            y = self._load_yaml(yaml_path)
            yamls.append((ds_root, y))

            for split in splits:
                if y.get(split):
                    splits_present.add(split)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            for index, (ds_root, y) in enumerate(yamls):
                names = self._get_names(y)
                dataset_tag = f"ds{index + 1}"

                split_image_paths = {}
                total_images = 0
                for split in splits:
                    if split in splits_present and y.get(split):
                        img_dir = self._split_path(y, split, ds_root)
                        if img_dir is None:
                            continue
                        image_paths = self._collect_split_images(img_dir)
                        split_image_paths[split] = image_paths
                        total_images += len(image_paths)

                task_total = total_images if total_images > 0 else 1
                task_id = progress.add_task(
                    f"[cyan]{dataset_tag}[/cyan] {ds_root.name}",
                    total=task_total,
                )

                if total_images == 0:
                    progress.advance(task_id, task_total)

                for split in splits:
                    if split in split_image_paths:
                        copied_images = self._copy_split(
                            ds_root,
                            y,
                            split,
                            out_root,
                            offset,
                            dataset_tag,
                            image_paths=split_image_paths[split],
                            progress=progress,
                            task_id=task_id,
                        )
                        total_copied_images += copied_images

                merged_names.extend(names)
                offset += len(names)

        if total_copied_images == 0:
            raise RuntimeError(
                "No images were copied during merge. "
                "Check each dataset's data.yaml split paths (train/val/test) and "
                "make sure they point to existing image directories."
            )

        merged_yaml = {
            "path": str(out_root),
            "train": "train/images" if (out_root / "train/images").exists() else None,
            "val": "val/images" if (out_root / "val/images").exists() else None,
            "test": "test/images" if (out_root / "test/images").exists() else None,
            "names": {i: n for i, n in enumerate(merged_names)},
            "nc": len(merged_names),
        }
        merged_yaml = {k: v for k, v in merged_yaml.items() if v is not None}

        with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_yaml, f, sort_keys=False, allow_unicode=True)

        console = Console()

        table = Table(title="Final Class Mapping")
        table.add_column("ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Class Name", style="green")

        for class_id, class_name in enumerate(merged_names):
            table.add_row(str(class_id), str(class_name))

        console.line()
        console.print(Align.center(table))

        summary = Table(title="Summary", show_header=False)
        summary.add_column("Field", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Number of dataset:", str(len(datasets)))
        summary.add_row("Mergred dataset to:", str(out_root))
        console.print(Align.center(summary))

    def merge_classes(
        self,
        dataset_dir: str,
        merge_from_names: list[str],
        merge_into_name: str,
        update_yaml: bool = True,
    ):
        """Merge multiple source class names into a target class.

        Args:
            dataset_dir: Dataset root containing labels and data.yaml.
            merge_from_names: Class names to merge into the target.
            merge_into_name: Target class name to merge into.
            update_yaml: Whether to update data.yaml names in place.

        Example:
            ```python
            handler = YoloFormatHandler()
            handler.merge_classes(
                dataset_dir="new_dataset",
                merge_from_names=["Backpack", "Backpacks"],
                merge_into_name="bag",
            )
            ```
        """
        dataset_dir = Path(dataset_dir).resolve()
        data_yaml = dataset_dir / "data.yaml"

        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

        y = self._load_yaml(data_yaml)
        names = self._get_names(y)
        name_to_id = {name: index for index, name in enumerate(names)}

        missing = [
            n for n in merge_from_names + [merge_into_name] if n not in name_to_id
        ]
        if missing:
            raise ValueError(
                f"These class names were not found in data.yaml names: {missing}"
            )

        target_id = name_to_id[merge_into_name]
        source_ids = {name_to_id[n] for n in merge_from_names}

        label_dirs = self._find_label_dirs(dataset_dir)
        if not label_dirs:
            raise FileNotFoundError(
                "Could not find any labels directories in common YOLO layouts."
            )

        label_files = []
        for label_dir in label_dirs:
            label_files.extend(list(label_dir.rglob("*.txt")))

        if not label_files:
            raise FileNotFoundError(f"No label .txt files found under: {label_dirs}")

        changed_files = 0
        changed_boxes = 0

        for label_file in label_files:
            lines_out = []
            changed_this_file = False

            with open(label_file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        lines_out.append(line)
                        continue

                    cls = int(float(parts[0]))
                    if cls in source_ids:
                        parts[0] = str(target_id)
                        changed_this_file = True
                        changed_boxes += 1

                    lines_out.append(" ".join(parts))

            if changed_this_file:
                changed_files += 1
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines_out) + "\n")

        if update_yaml:
            updated_names = names[:]
            for src_name in merge_from_names:
                updated_names[name_to_id[src_name]] = merge_into_name

            y["names"] = {i: n for i, n in enumerate(updated_names)}
            y["nc"] = len(updated_names)
            self._save_yaml(data_yaml, y)

        print(f"Updated {changed_boxes} boxes across {changed_files} files.")
        if update_yaml:
            print("Updated data.yaml (safe rename; no class ID shifting).")

    def remap_dataset(
        self,
        dataset_dir: str,
        new_names: list[str],
        id_mapping: dict[int, int],
    ):
        """Remap class IDs and update class names.

        Args:
            dataset_dir: Dataset root containing labels and data.yaml.
            new_names: New class names in final ID order.
            id_mapping: Mapping from old class IDs to new IDs.

        Example:
            ```python
            handler = YoloFormatHandler()
            handler.remap_dataset(
                dataset_dir="new_dataset",
                new_names=["bag", "person"],
                id_mapping={0: 0, 1: 0, 2: 1},
            )
            ```
        """
        dataset_dir = Path(dataset_dir).resolve()
        data_yaml = dataset_dir / "data.yaml"

        if not data_yaml.exists():
            raise FileNotFoundError("data.yaml not found")

        label_files = self._find_label_files(dataset_dir)
        if not label_files:
            raise RuntimeError("No label files found")

        max_new_id = len(new_names) - 1
        invalid_new_ids = sorted(
            {
                new_id
                for new_id in id_mapping.values()
                if new_id < 0 or new_id > max_new_id
            }
        )
        if invalid_new_ids:
            raise ValueError(
                f"Mapped new IDs out of range [0, {max_new_id}]: {invalid_new_ids}"
            )

        rewritten_files = []
        total_boxes = 0

        # Pre-scan all label files and validate mappings before mutating any file.
        for label_file in label_files:
            lines_out = []
            changed = False

            with open(label_file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        lines_out.append(line)
                        continue

                    old_id = int(float(parts[0]))
                    if old_id not in id_mapping:
                        raise ValueError(f"Class ID {old_id} not in mapping")

                    new_id = id_mapping[old_id]
                    parts[0] = str(new_id)
                    lines_out.append(" ".join(parts))

                    total_boxes += 1
                    changed = True

            rewritten_files.append((label_file, lines_out, changed))

        for label_file, lines_out, changed in rewritten_files:
            if changed:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines_out) + "\n")

        y = self._load_yaml(data_yaml)
        y["names"] = {i: name for i, name in enumerate(new_names)}
        y["nc"] = len(new_names)
        self._save_yaml(data_yaml, y)

        print(f"Updated {total_boxes} boxes.")
        print("Updated data.yaml")

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
        """Visualize random labeled samples in a grid.

        Args:
            images_dir: Directory containing images.
            labels_dir: Directory containing label files.
            names: Optional class name list indexed by class ID.
            n: Number of images to sample.
            seed: Random seed for sampling.
            cols: Column count for the grid (auto if None).
            tile_size: Target tile size (width, height).

        Example:
            ```python
            handler = YoloFormatHandler()
            handler.visualize_samples(
                images_dir="new_dataset/train/images",
                labels_dir="new_dataset/train/labels",
                names=["car", "person"],
                n=12,
                seed=2,
            )
            ```
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        if n <= 0:
            raise ValueError("n must be a positive integer")
        if not images_path.exists():
            raise FileNotFoundError(f"Images dir not found: {images_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels dir not found: {labels_path}")

        all_images = [
            p
            for p in images_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self._img_exts
        ]
        if not all_images:
            raise RuntimeError(f"No images found under: {images_path}")

        random.seed(seed)
        samples = random.sample(all_images, k=min(n, len(all_images)))

        if cols is None:
            cols = max(1, min(len(samples), math.ceil(math.sqrt(n))))
        elif cols <= 0:
            raise ValueError("cols must be a positive integer")

        tile_w, tile_h = tile_size
        if tile_w <= 0 or tile_h <= 0:
            raise ValueError("tile_size must contain positive integers")

        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        fig.subplots_adjust(
            left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02
        )
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i, img_path in enumerate(samples):
            r, c = divmod(i, cols)
            ax = axes[r][c]

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            letterboxed, scale, pad_x, pad_y = self._letterbox_image(
                img, tile_w, tile_h
            )

            ax.imshow(np.asarray(letterboxed))
            ax.axis("off")
            ax.set_box_aspect(tile_h / tile_w)

            label_path = self._find_label_path(img_path, images_path, labels_path)
            boxes = self._read_yolo_labels(label_path)

            for cls_id, xc, yc, w, h in boxes:
                x1, y1, x2, y2 = self._yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
                x1 = x1 * scale + pad_x
                y1 = y1 * scale + pad_y
                x2 = x2 * scale + pad_x
                y2 = y2 * scale + pad_y

                rect = patches.Rectangle(
                    (x1, y1),
                    (x2 - x1),
                    (y2 - y1),
                    fill=False,
                    linewidth=2,
                )
                ax.add_patch(rect)

                label = str(cls_id)
                if names and 0 <= cls_id < len(names):
                    label = f"{cls_id}:{names[cls_id]}"

                ax.text(
                    x1,
                    max(0, y1 - 3),
                    label,
                    fontsize=12,
                    bbox=dict(alpha=0.4, pad=1),
                )

            if not boxes:
                ax.text(
                    10,
                    20,
                    "NO LABEL FILE / EMPTY",
                    fontsize=12,
                    bbox=dict(alpha=0.4, pad=2),
                )

        total_axes = rows * cols
        for j in range(len(samples), total_axes):
            r, c = divmod(j, cols)
            axes[r][c].axis("off")

        plt.show()

    def _load_yaml(self, yaml_path: Path):
        """Load YAML content from disk.

        Args:
            yaml_path: Path to a YAML file.

        Returns:
            Parsed YAML as a Python object.

        Example:
            ```python
            data = handler._load_yaml(Path("Drones/data.yaml"))
            ```
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_yaml(self, yaml_path: Path, content: dict):
        """Write YAML content to disk.

        Args:
            yaml_path: Destination YAML file path.
            content: Mapping to serialize.

        Example:
            ```python
            handler._save_yaml(Path("new_dataset/data.yaml"), {"nc": 2, "names": {0: "car", 1: "person"}})
            ```
        """
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)

    def _get_names(self, y: dict):
        """Normalize the YAML ``names`` field to a list of class names.

        Args:
            y: Parsed ``data.yaml`` content.

        Returns:
            A class-name list in index order.

        Example:
            ```python
            names = handler._get_names({"names": {0: "car", 1: "person"}})
            # names == ["car", "person"]
            ```
        """
        names = y.get("names")
        if isinstance(names, dict):
            return [names[i] for i in range(len(names))]
        if isinstance(names, list):
            return names
        raise ValueError("Unsupported YAML 'names' format. Use list or dict.")

    def _split_path(self, y: dict, split_key: str, dataset_root: Path):
        """Resolve a split path from YAML relative to the dataset root.

        Args:
            y: Parsed ``data.yaml`` content.
            split_key: Split key such as ``train``, ``val``, or ``test``.
            dataset_root: Root folder of the source dataset.

        Returns:
            Absolute split path, or ``None`` if the split key is missing.

        Example:
            ```python
            train_dir = handler._split_path({"train": "train/images"}, "train", Path("Drones"))
            ```
        """
        p = y.get(split_key)
        if not p:
            return None

        p = Path(p)
        if not p.is_absolute():
            p = (dataset_root / p).resolve()
        return p

    def _rewrite_label_file(self, src_label: Path, dst_label: Path, offset: int):
        """Copy a label file while offsetting class IDs.

        Args:
            src_label: Source YOLO label file.
            dst_label: Destination label file path.
            offset: Class-ID offset to add to each row.

        Example:
            ```python
            handler._rewrite_label_file(
                Path("Drones/train/labels/img1.txt"),
                Path("new_dataset/train/labels/ds1__img1.txt"),
                offset=3,
            )
            ```
        """
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        if not src_label.exists():
            return

        out_lines = []
        with open(src_label, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                cls = int(float(parts[0]))
                parts[0] = str(cls + offset)
                out_lines.append(" ".join(parts))

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    def _copy_split(
        self,
        dataset_root: Path,
        y: dict,
        split_key: str,
        out_root: Path,
        offset: int,
        dataset_tag: str,
        image_paths: list[Path] | None = None,
        progress=None,
        task_id: int | None = None,
    ) -> int:
        """Copy images/labels for a split, applying class ID offsets.

        Args:
            dataset_root: Source dataset root.
            y: Parsed source ``data.yaml`` content.
            split_key: Split key such as ``train``.
            out_root: Merged output dataset root.
            offset: Class-ID offset for this dataset.
            dataset_tag: Prefix used for destination file names.
            image_paths: Optional pre-collected image paths.
            progress: Optional rich progress object.
            task_id: Optional task id for progress updates.

        Returns:
            Number of images copied for this split.

        Example:
            ```python
            copied = handler._copy_split(
                dataset_root=Path("Drones"),
                y={"train": "train/images"},
                split_key="train",
                out_root=Path("new_dataset"),
                offset=0,
                dataset_tag="ds1",
            )
            ```
        """
        img_dir = self._split_path(y, split_key, dataset_root)
        if img_dir is None:
            return 0

        parts = list(img_dir.parts)
        if "images" not in parts:
            raise ValueError(f"Can't infer labels dir (no 'images' in path): {img_dir}")

        idx = parts.index("images")
        lbl_dir = Path(*parts[:idx], "labels", *parts[idx + 1 :])

        out_img_dir = out_root / split_key / "images"
        out_lbl_dir = out_root / split_key / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if image_paths is None:
            image_paths = self._collect_split_images(img_dir)

        copied_images = 0
        for img_path in image_paths:
            new_name = f"{dataset_tag}__{img_path.name}"
            dst_img = out_img_dir / new_name
            shutil.copy2(img_path, dst_img)
            copied_images += 1

            src_label = lbl_dir / (img_path.stem + ".txt")
            dst_label = out_lbl_dir / (Path(new_name).stem + ".txt")
            self._rewrite_label_file(src_label, dst_label, offset)

            if progress is not None and task_id is not None:
                progress.advance(task_id)

        return copied_images

    def _collect_split_images(self, img_dir: Path):
        """Collect image files for a split directory.

        Args:
            img_dir: Split image directory.

        Returns:
            List of image file paths with supported extensions.

        Example:
            ```python
            images = handler._collect_split_images(Path("Drones/train/images"))
            ```
        """
        return [
            p
            for p in img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self._img_exts
        ]

    def _find_label_dirs(self, dataset_dir: Path):
        """Find labels directories in common YOLO layouts.

        Args:
            dataset_dir: Dataset root path.

        Returns:
            List of detected label directories.

        Example:
            ```python
            label_dirs = handler._find_label_dirs(Path("new_dataset"))
            ```
        """
        label_dirs = []

        for split in self._img_splits:
            candidate = dataset_dir / "labels" / split
            if candidate.exists():
                label_dirs.append(candidate)

        for split in self._img_splits:
            candidate = dataset_dir / split / "labels"
            if candidate.exists():
                label_dirs.append(candidate)

        return label_dirs

    def _find_label_files(self, dataset_dir: Path):
        """Find label files in common YOLO layouts.

        Args:
            dataset_dir: Dataset root path.

        Returns:
            List of label ``.txt`` files.

        Example:
            ```python
            label_files = handler._find_label_files(Path("new_dataset"))
            ```
        """
        files = list(dataset_dir.rglob("labels/**/*.txt"))
        if files:
            return files

        alt_files = []
        for split in self._img_splits:
            alt_files.extend(list((dataset_dir / split / "labels").rglob("*.txt")))
        return alt_files

    def _find_label_path(
        self, img_path: Path, images_dir: Path, labels_dir: Path
    ) -> Path:
        """Compute the label path corresponding to an image path.

        Args:
            img_path: Image file path.
            images_dir: Root image directory.
            labels_dir: Root labels directory.

        Returns:
            The corresponding label path with ``.txt`` suffix.

        Example:
            ```python
            label_path = handler._find_label_path(
                Path("new_dataset/train/images/a.jpg"),
                Path("new_dataset/train/images"),
                Path("new_dataset/train/labels"),
            )
            ```
        """
        rel = img_path.relative_to(images_dir)
        return (labels_dir / rel).with_suffix(".txt")

    def _read_yolo_labels(self, label_path: Path):
        """Read YOLO labels into ``(class_id, xc, yc, w, h)`` tuples.

        Args:
            label_path: Path to a YOLO label file.

        Returns:
            List of parsed label tuples.

        Example:
            ```python
            boxes = handler._read_yolo_labels(Path("new_dataset/train/labels/a.txt"))
            ```
        """
        if not label_path.exists():
            return []

        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls_id, xc, yc, w, h))

        return boxes

    def _yolo_to_xyxy(self, xc, yc, w, h, img_w, img_h):
        """Convert YOLO normalized box values to clamped pixel ``xyxy``.

        Args:
            xc: Normalized x-center.
            yc: Normalized y-center.
            w: Normalized width.
            h: Normalized height.
            img_w: Source image width.
            img_h: Source image height.

        Returns:
            Tuple ``(x1, y1, x2, y2)`` in image pixel coordinates.

        Example:
            ```python
            x1, y1, x2, y2 = handler._yolo_to_xyxy(0.5, 0.5, 0.2, 0.2, 1280, 720)
            ```
        """
        x_center = xc * img_w
        y_center = yc * img_h
        bw = w * img_w
        bh = h * img_h

        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        return x1, y1, x2, y2

    def _letterbox_image(self, img, target_w: int, target_h: int):
        """Resize an image with letterboxing to the target size.

        Args:
            img: ``PIL.Image`` instance.
            target_w: Output canvas width.
            target_h: Output canvas height.

        Returns:
            Tuple ``(canvas, scale, pad_x, pad_y)``.

        Example:
            ```python
            canvas, scale, pad_x, pad_y = handler._letterbox_image(img, 640, 640)
            ```
        """
        img_w, img_h = img.size
        scale = min(target_w / img_w, target_h / img_h)
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))

        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), color=(20, 20, 20))

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, pad_x, pad_y
