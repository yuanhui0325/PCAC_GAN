import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def psnr(recon_tensor, orig_tensor):

    mse = torch.mean((recon_tensor.F - orig_tensor.F) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def ssim(recon_tensor, orig_tensor):

    recon = recon_tensor.dense()[0].cpu().numpy().transpose(1,2,3,0)
    orig = orig_tensor.dense()[0].cpu().numpy().transpose(1,2,3,0)
    
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min())
    orig_norm = (orig - orig.min()) / (orig.max() - orig.min())
    
    return ssim(orig_norm, recon_norm, multichannel=True)

def calculate_bpp(bitstream, num_points):

    total_bits = len(bitstream) * 8  
    return total_bits / num_points
