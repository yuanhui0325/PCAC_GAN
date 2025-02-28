import torch.nn as nn
import MinkowskiEngine as ME

class AVRPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(  
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU()
        )
    
    def forward(self, x):
        return self.net(x)