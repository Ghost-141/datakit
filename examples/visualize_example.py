"""Example: visualize labeled samples from a dataset."""

from datakit import plot_random_samples


if __name__ == "__main__":
    images_dir = "/path/dataset/val/images"
    labels_dir = "/path/dataset/val/labels"
    names = ["bag", "person"]
    plot_random_samples(
        images_dir=images_dir,
        labels_dir=labels_dir,
        names=names,
        n=12,
        seed=1,
        cols=4,
        tile_size=(640, 640),
    )
