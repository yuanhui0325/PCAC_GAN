import MinkowskiEngine as ME
import torch.nn as nn

class PointCloudDiscriminator(ME.MinkowskiNetwork):
    def __init__(self):
        super().__init__(D=3)
        self.layers = nn.ModuleList([
            ME.MinkowskiConvolution(3, 64, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(128, 256, kernel_size=3, dimension=3),
            ME.MinkowskiGlobalPooling()
        ])
        self.linear = nn.Linear(256, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.linear(x.F)
