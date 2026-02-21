# datakit

[PyPI: cv-datakit](https://pypi.org/project/cv-datakit/)

Python package for YOLO-format dataset operations:
- merge multiple datasets into one
- merge multiple class names into a target class
- remap class IDs
- visualize labeled samples

## Install

```bash
pip install cv-datakit
```


## CLI Usage

### 1) Merge datasets

```bash
datakit merge /path/ds1 /path/ds2 --out /path/out
```

### 2) Merge classes

```bash
datakit merge-classes /path/dataset --from Backpack Backpacks --to bag
```

### 3) Remap classes

```bash
datakit remap /path/dataset --names bag person --map 0:0 1:0 2:1
```

Remap safety behavior:
- validates that all mapped target IDs are within `0..len(new_names)-1`
- pre-scans all label files to ensure every class ID has a mapping before writing
- only writes labels and `data.yaml` after validation succeeds
- note: writes are not yet atomic across process interruption (power loss/kill)

### 4) Visualize samples

```bash
datakit visualize --images-dir /path/dataset/val/images --labels-dir /path/dataset/val/labels --n 12 --seed 1
```

### Format selection

```bash
datakit --format yolo merge /path/ds1 /path/ds2 --out /path/out
```

## Python API

```python
from datakit import merge_datasets, merge_classes, remap_dataset, plot_random_samples

merge_datasets(["/path/ds1", "/path/ds2"], "/path/out")
merge_classes("/path/dataset", ["Backpack", "Backpacks"], "bag")
remap_dataset("/path/dataset", ["bag", "person"], {0: 0, 1: 0, 2: 1})
plot_random_samples("/path/dataset/val/images", "/path/dataset/val/labels", n=12, seed=1)
```

## Extend to new formats

1. Implement `DatasetFormatHandler` in a new module, for example `datakit/formats/coco.py`.
2. Register the handler in `datakit/formats/__init__.py`.
3. Use the same CLI commands with `--format coco`.
