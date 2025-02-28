import os
os.environ["ME_MEMORY_ALIGN"] = "16"  
os.environ["CUDA_CACHE_PATH"] = "/tmp/cuda_cache" 
import yaml
import argparse
import torch
torch.cuda.init()
_ = torch.zeros(1).cuda()  # 预初始化上下文
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from models.generator import PCACGenerator
from models.discriminator import PointCloudDiscriminator
from data.ivfb_loader import IVFBDataset
from losses.adversarial import WGAN_GP_Loss
from losses.rate_distortion import RateDistortionLoss
from utils.config import load_config

def minkowski_collate_fn(batch):
    """
    正确处理批次数据，返回字典格式
    """

    coordinates = [item['coordinates'].contiguous() for item in batch]
    features = [item['features'].contiguous() for item in batch]
    
    coords_batch, feats_batch = ME.utils.sparse_collate(
        coordinates, 
        features,
        dtype=torch.float32
    )
    
    return {
        'coordinates': coords_batch.to(torch.int32).contiguous(),
        'features': feats_batch.contiguous()
    }

def main(config_path):
    print("[1/6] 加载配置...")
    config = load_config(config_path)
    
    print("[2/6] 初始化模型...")
    generator = PCACGenerator(config).cuda()
    discriminator = PointCloudDiscriminator().cuda()
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=config['training']['lr_g'])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config['training']['lr_d'])
    
    print("[3/6] 加载数据集...")
    dataset = IVFBDataset(
        root_dir=config['dataset']['root_dir'],  
        voxel_size=config['dataset']['voxel_size']
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=minkowski_collate_fn  
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    adv_loss = WGAN_GP_Loss(lambda_gp=config['training']['lambda_gp'])
    rd_loss = RateDistortionLoss(lambda_rd=config['training']['lambda_rd'])
    
    print("[4/6] 开始训练...")
    for epoch in range(config['training']['epochs']):
        for batch_idx, batch in enumerate(train_loader):
            # 调试输出
            print("批次数据类型:", type(batch))  # 应输出dict
            print("坐标张量形状:", batch['coordinates'].shape)  # 例如：[N, 3]
            print("特征张量形状:", batch['features'].shape)    # 例如：[N, C]
            # 数据准备
            real_data = {
                'coordinates': batch['coordinates'].cuda(),
                'features': batch['features'].cuda()
            }
            
            # ================== 训练判别器 ==================
            with torch.cuda.amp.autocast():
                fake_outputs = generator(real_data)
                fake_data = fake_outputs['reconstruction'].detach()
                
                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data)
                gp_loss = adv_loss.compute_gradient_penalty(discriminator, real_data, fake_data)
                d_loss = d_fake.mean() - d_real.mean() + config['training']['lambda_gp'] * gp_loss
                
            opt_d.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            
            # ================== 训练生成器 ==================
            with torch.cuda.amp.autocast():
                fake_outputs = generator(real_data)
                
                g_adv = -torch.mean(discriminator(fake_outputs['reconstruction']))
                
                g_rd = rd_loss(
                    fake_outputs, 
                    real_data,
                    fake_outputs['bitstream'],
                    real_data.F.shape[0]
                )
                
                g_total = g_adv + g_rd
                
            opt_g.zero_grad()
            scaler.scale(g_total).backward()
            scaler.step(opt_g)
            scaler.update()
            
            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1}/{config['training']['epochs']} "
                      f"Batch: {batch_idx}/{len(train_loader)} "
                      f"D_loss: {d_loss.item():.4f} "
                      f"G_loss: {g_total.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    
    import os
    os.environ["OMP_NUM_THREADS"] = "16"  
    
    main(args.config)
