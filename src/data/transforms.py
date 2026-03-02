from torchvision import transforms


def get_train_transforms(img_size=256):
    return transforms.Compose(
        [
            # resize to 286x286 to give room to randomly crop
            transforms.Resize(
                (int(img_size * 1.12), int(img_size * 1.12)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # Generator final layer uses tanh activation
            # tanh outputs values in [-1, 1]
            # input and output ranges must match for training stability
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_test_transforms(img_size=256):
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def denormalize(tensor):
    return tensor * 0.5 + 0.5
