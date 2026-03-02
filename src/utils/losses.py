import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    GAN loss (LSGAN)
    Uses MSE instead of BCE cause BCE saturates when very wrong
    MSE never saturates, always has gradient
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


def cycle_consistency_loss(real_images, reconstructed_images):
    """L1 loss for cycle consistency"""
    return torch.mean(torch.abs(real_images - reconstructed_images))


def identity_loss(real_images, same_images):
    """L1 loss for identity mapping"""
    return torch.mean(torch.abs(real_images - same_images))
