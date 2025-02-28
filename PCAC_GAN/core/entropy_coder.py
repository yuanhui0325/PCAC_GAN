# core/entropy_coder.py
import torch
import torchac
import torch.nn as nn
import math  

class ArithmeticCoder:
    
    def __init__(self, precision=16):
        """
        参数:
        precision (int): CDF精度位数，默认16位
        """
        self.precision = precision
        
    def _pmf_to_cdf(self, pmf):
        """
        将PMF转换为累积分布函数(CDF)
        
        参数:
        pmf (torch.Tensor): 形状为(B, L)的概率质量函数
        
        返回:
        torch.Tensor: 形状为(B, L+1)的累积分布函数
        """
        assert torch.all(pmf >= 0), "PMF包含负值"
        assert torch.allclose(pmf.sum(dim=-1), torch.ones_like(pmf.sum(dim=-1))), "PMF未归一化"
        
        cdf = torch.cumsum(pmf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        cdf = (cdf * (2 ** self.precision)).int()
        
        cdf[..., -1] = 2 ** self.precision  # 强制最后一位为2^precision
        return cdf
    
    def encode(self, symbols, pmf):
        """
        执行算术编码
        
        参数:
        symbols (torch.Tensor): 要编码的符号，形状为(B,)
        pmf (torch.Tensor): 每个符号的概率分布，形状为(B, L)
        
        返回:
        bytes: 编码后的字节流
        """
        assert symbols.min() >= 0, "符号值必须非负"
        assert symbols.max() < pmf.shape[-1], "符号值超出PMF范围"
        
        cdf = self._pmf_to_cdf(pmf)
        symbols = symbols.to(torch.int16)
        return torchac.encode_float_cdf(cdf.cpu(), symbols.cpu())
    
    def decode(self, byte_stream, pmf, output_shape):
        """
        执行算术解码
        
        参数:
        byte_stream (bytes): 编码后的字节流
        pmf (torch.Tensor): 每个符号的概率分布，形状为(B, L)
        output_shape (tuple): 输出符号的形状
        
        返回:
        torch.Tensor: 解码后的符号
        """
        cdf = self._pmf_to_cdf(pmf)
        return torchac.decode_float_cdf(cdf.cpu(), byte_stream, output_shape).to(pmf.device)

class EntropyBottleneck(nn.Module):
    """熵瓶颈模块（包含量化噪声建模）"""
    
    def __init__(self, channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU()
        )
        self.mu_layer = nn.Conv1d(channels, channels, 3, padding=1)
        self.sigma_layer = nn.Conv1d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        """
        前向传播流程：
        1. 生成潜在表示的统计参数
        2. 添加量化噪声（训练时）
        3. 计算率损失
        
        参数:
        x (torch.Tensor): 输入特征张量
        
        返回:
        (torch.Tensor, torch.Tensor): (量化后的潜在变量, 率损失)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_sigma = self.sigma_layer(h)
        sigma = torch.exp(log_sigma)
        
        if self.training:
            noise = torch.rand_like(x) - 0.5
            quantized = x + noise
        else:
            quantized = torch.round(x)
            
        rate = self._calculate_rate(x, mu, sigma)
        
        return quantized, rate
    
    def _calculate_rate(self, x, mu, sigma):

        cdf_center = 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))
        cdf_low = cdf_center - 0.5 / 256
        cdf_high = cdf_center + 0.5 / 256
        rate = -torch.log2(cdf_high - cdf_low).mean()
        return rate
