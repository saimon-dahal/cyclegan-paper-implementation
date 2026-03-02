import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN discriminator (70x70 receptive field)
    Input: Image ( 3, 256, 156)
    Output: Image ( 1, 30, 30)

    Each position in 30x30 map: "Is this patch of image real or fake?"
    Due to the local decisions, the Generator gets more precise feedback
    """

    def __init__(self, input_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)
