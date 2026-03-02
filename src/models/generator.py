import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super().__init__()

        # Initial convolution block
        model = [
            # reflection padding creates smooth natural continuation than zero padding
            nn.ReflectionPad2d(3),
            # 7X7 kernel so that it can see boarder context and good for initial feature extraction
            nn.Conv2d(input_channels, 64, 7),
            # calculate mean/std for each image separately [Better for style transfer]
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),  # inplace=True saves memory
        ]
        # input (3, 256, 256) -> output (64, 256, 256)
        # extracted 64 feature maps from RGB input

        # Downsampling
        # input (64, 256, 256)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Output
        # Downsample 1: (128, 128, 128)
        # Downsample 2: (256, 64, 64)
        # compressed spatial information
        # increased channel depth (more abstract features) + easier to process cause fewer pixels

        # Residual blocks
        # more blocks = more capacity to learn.
        # 9 is standard for 256 * 256 images
        # for 128x128, use 6 blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        # Input: (256, 64, 64)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        # Output: (64, 256, 256)
        # Back to original spatial size!

        # Output layer
        # Input: (64, 256, 256)
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_channels, 7), nn.Tanh()]
        # Output: (3, 256, 256)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
