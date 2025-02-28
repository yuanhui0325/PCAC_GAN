import torch
import torch.nn as nn

class RateDistortionLoss(nn.Module):
    def __init__(self, lambda_rd=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_rd = lambda_rd
        
    def forward(self, pred, target, bitstream, num_points):
        recon_loss = self.mse(pred['reconstruction'].F, target.F)
        
        bpp = len(bitstream) * 8 / num_points

        latent_loss = 0.5 * torch.mean(
            torch.log(pred['latent_sigma'].F**2) + 
            (pred['latent_mu'].F**2) / pred['latent_sigma'].F**2
        )
        
        return recon_loss + self.lambda_rd * (bpp + latent_loss)
