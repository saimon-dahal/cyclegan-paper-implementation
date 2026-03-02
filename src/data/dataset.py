import os
import random

from PIL import Image
from torch.utils.data import Dataset


class UnpairedDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform_A=None, transform_B=None):
        self.dir_A = os.path.join(root_dir, f"{mode}A")
        self.dir_B = os.path.join(root_dir, f"{mode}B")
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.mode = mode

        self.A_images = sorted(
            [
                os.path.join(self.dir_A, x)
                for x in os.listdir(self.dir_A)
                if self._is_image(x)
            ]
        )
        self.B_images = sorted(
            [
                os.path.join(self.dir_B, x)
                for x in os.listdir(self.dir_B)
                if self._is_image(x)
            ]
        )

        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)

    @staticmethod
    def _is_image(filename):
        return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))

    def __len__(self):
        # if mismatch on both datasets, this max will ensure
        # both domains get equal exposure during training
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        A_path = self.A_images[index % self.A_size]

        if self.mode == "train":
            # random pairing to prevent memorization of specific A-B pairs
            B_path = self.B_images[random.randint(0, self.B_size - 1)]
        else:
            # deterministic pairing
            B_path = self.B_images[index % self.B_size]

        # ensure 3 channels only
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        if self.transform_A:
            A_img = self.transform_A(A_img)
        if self.transform_B:
            B_img = self.transform_B(B_img)

        return {"A": A_img, "B": B_img}
