from datakit import merge_datasets, merge_classes, remap_dataset, plot_random_samples

merge_datasets(["Drones/", "R2P2.v2-raw-images/"], "new_dataset")
# merge_classes("/path/dataset", ["Backpack", "Backpacks"], "bag")
# remap_dataset("/path/dataset", ["bag", "person"], {0: 0, 1: 0, 2: 1})
# plot_random_samples(
#     "new_dataset/train/images/",
#     "/new_dataset/train/labels/",
#     names=[
#         "auto",
#         "czlowiek",
#         "dron",
#         "Oil",
#         "jam",
#         "pasta",
#         "rice",
#         "soda",
#         "tomato sauce",
#     ],
#     n=12,
#     seed=2,
# )
