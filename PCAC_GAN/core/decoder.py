import MinkowskiEngine as ME
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(channels)
        )
        self.relu = ME.MinkowskiReLU()
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return self.relu(out)

class MultiScaleDecoder(ME.MinkowskiNetwork):
    def __init__(self, in_channels=256):
        super().__init__(D=3)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(128) for _ in range(3)]
        )
        self.upsample = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dimension=3
        )
    
    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        return self.upsample(x)
