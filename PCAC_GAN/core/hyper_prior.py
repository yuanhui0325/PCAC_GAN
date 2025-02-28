import MinkowskiEngine as ME

class HyperPriorCoder(ME.MinkowskiNetwork):
    def __init__(self, in_channels=256):
        super().__init__(D=3)
        self.encoder = ME.Sequential(
            ME.MinkowskiConvolution(in_channels, 256, kernel_size=3),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU()
        )
        self.mu_head = ME.MinkowskiConvolution(256, 256, kernel_size=3)
        self.sigma_head = ME.MinkowskiConvolution(256, 256, kernel_size=3)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)
        return mu, torch.exp(log_sigma)  # 保证sigma为正
