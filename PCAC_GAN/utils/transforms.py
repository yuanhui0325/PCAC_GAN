
import torch

def rgb_to_yuv(rgb_tensor):
    rgb_normalized = rgb_tensor / 255.0
    """
    将RGB张量转换为YUV颜色空间
    输入范围假设为[0, 1]
    """
    transform_mat = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], device=rgb_tensor.device)
    

    yuv = torch.matmul(rgb_tensor, transform_mat.T)
    
    yuv[..., 1:] += 0.5  
    return torch.clamp(yuv, 0.0, 1.0)

def yuv_to_rgb(yuv_tensor):
    """
    将YUV张量转换回RGB颜色空间
    """
    # 逆变换矩阵
    inv_mat = torch.tensor([
        [1.0, 0.0, 1.13983],
        [1.0, -0.39465, -0.58060],
        [1.0, 2.03211, 0.0]
    ], device=yuv_tensor.device)
    

    yuv = yuv_tensor.clone()
    yuv[..., 1:] -= 0.5  # 去除U/V分量偏移
    
    return torch.clamp(torch.matmul(yuv, inv_mat.T), 0.0, 1.0)
