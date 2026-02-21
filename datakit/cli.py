"""Command-line interface for datakit."""

import argparse

from .formats import get_format_handler


def _parse_mapping(mapping_values: list[str]):
    """Parse OLD:NEW mappings from CLI into an integer dict."""
    mapping = {}
    for item in mapping_values:
        if ":" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Use OLD:NEW format, e.g. 3:1")
        old, new = item.split(":", 1)
        mapping[int(old)] = int(new)
    return mapping


def main():
    """Entry point for the datakit CLI."""
    parser = argparse.ArgumentParser(prog="datakit")
    parser.add_argument(
        "--format",
        default="yolo",
        help="Dataset format handler to use (default: yolo)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    merge_parser.add_argument("datasets", nargs="+", help="Input dataset directories")
    merge_parser.add_argument("--out", required=True, help="Output merged dataset directory")

    merge_cls_parser = subparsers.add_parser(
        "merge-classes", help="Merge multiple class names into a target class"
    )
    merge_cls_parser.add_argument("dataset", help="Dataset directory")
    merge_cls_parser.add_argument(
        "--from",
        dest="merge_from",
        nargs="+",
        required=True,
        help="Source class names to merge",
    )
    merge_cls_parser.add_argument("--to", required=True, help="Target class name")
    merge_cls_parser.add_argument(
        "--no-update-yaml",
        action="store_true",
        help="Do not update class names in dataset metadata",
    )

    remap_parser = subparsers.add_parser("remap", help="Remap class IDs")
    remap_parser.add_argument("dataset", help="Dataset directory")
    remap_parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="New class names in final ID order",
    )
    remap_parser.add_argument(
        "--map",
        nargs="+",
        required=True,
        metavar="OLD:NEW",
        help="ID mappings, e.g. --map 0:0 1:0 2:1",
    )

    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize random samples with labels"
    )
    visualize_parser.add_argument("--images-dir", required=True, help="Images directory")
    visualize_parser.add_argument("--labels-dir", required=True, help="Labels directory")
    visualize_parser.add_argument(
        "--names",
        nargs="+",
        help="Optional class names list where index is class ID",
    )
    visualize_parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of random images to visualize",
    )
    visualize_parser.add_argument("--seed", type=int, default=2, help="Random seed")
    visualize_parser.add_argument(
        "--cols",
        type=int,
        help="Columns in visualization grid (auto if omitted)",
    )
    visualize_parser.add_argument(
        "--tile-width",
        type=int,
        default=640,
        help="Letterboxed tile width",
    )
    visualize_parser.add_argument(
        "--tile-height",
        type=int,
        default=640,
        help="Letterboxed tile height",
    )

    args = parser.parse_args()
    handler = get_format_handler(args.format)

    if args.command == "merge":
        handler.merge_datasets(args.datasets, args.out)
    elif args.command == "merge-classes":
        handler.merge_classes(
            dataset_dir=args.dataset,
            merge_from_names=args.merge_from,
            merge_into_name=args.to,
            update_yaml=not args.no_update_yaml,
        )
    elif args.command == "remap":
        handler.remap_dataset(
            dataset_dir=args.dataset,
            new_names=args.names,
            id_mapping=_parse_mapping(args.map),
        )
    elif args.command == "visualize":
        handler.visualize_samples(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            names=args.names,
            n=args.n,
            seed=args.seed,
            cols=args.cols,
            tile_size=(args.tile_width, args.tile_height),
        )


if __name__ == "__main__":
    main()
