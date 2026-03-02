import torch

from .dataset import UnpairedDataset
from .transforms import get_test_transforms, get_train_transforms


def get_dataloader(
    root_dir, mode="train", batch_size=1, img_size=256, num_workers=4, shuffle=None
):

    if shuffle is None:
        shuffle = mode == "train"

    if mode == "train":
        transform = get_train_transforms(img_size)
    else:
        transform = get_test_transforms(img_size)

    dataset = UnpairedDataset(
        root_dir=root_dir, mode=mode, transform_A=transform, transform_B=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )

    return dataloader


def get_dataloaders(root_dir, batch_size=1, img_size=256, num_workers=4):
    train_loader = get_dataloader(root_dir, "train", batch_size, img_size, num_workers)
    test_loader = get_dataloader(root_dir, "test", 1, img_size, num_workers)
    return train_loader, test_loader
