import os
import torch
import open3d as o3d
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from utils.transforms import rgb_to_yuv

class IVFBDataset(Dataset):
    def __init__(self, root_dir, voxel_size=0.01, color_norm=True):
        """
        Args:
            root_dir (str): PLY文件存储目录
            voxel_size (float): 体素化网格大小（单位：米）
            color_norm (bool): 是否将颜色归一化到[0,1]
        """
        import glob
        self.file_list = sorted(glob.glob(f"{root_dir}/*.ply"))
        self.voxel_size = voxel_size
        self.color_norm = color_norm
        
        self.valid_files = []
        for f in self.file_list:
            try:
                pcd = o3d.io.read_point_cloud(f)
                if len(pcd.points) > 0:
                    self.valid_files.append(f)
            except:
                print(f"警告：无法加载文件 {f}，已跳过")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.valid_files[idx])
        
        down_pcd = pcd.voxel_down_sample(self.voxel_size)
        points = np.asarray(down_pcd.points)
        colors = np.asarray(down_pcd.colors)

        points = np.asarray(down_pcd.points)
        #points = (points / self.voxel_size).astype(np.int64) 
        #points = points.astype(np.int32)  
        quantized_points = (points / self.voxel_size).astype(np.int32)

        aligned_coords = np.zeros((points.shape[0], 4), dtype=np.int32)
        aligned_coords[:, :3] = quantized_points  
        coords = torch.from_numpy(aligned_coords[:, :3]).contiguous()

        aligned_features = np.zeros((colors.shape[0], 4), dtype=np.float32)
        aligned_features[:, :3] = colors
        features = torch.from_numpy(aligned_features).contiguous()

        assert (coords.data_ptr() % 16) == 0, f"坐标未对齐: {coords.data_ptr()}"
        assert (features.data_ptr() % 16) == 0, f"特征未对齐: {features.data_ptr()}"
        
        coords = torch.tensor(
            np.floor(points / self.voxel_size),  # 对齐体素网格
            dtype=torch.int32
        ).contiguous()  # 强制内存连续
        
        if self.color_norm and colors.max() > 1:
            colors = colors / 255.0  # 归一化到[0,1]
            
        features = torch.tensor(
            colors,
            dtype=torch.float32
        ).contiguous() 

        #features = torch.zeros((colors.shape[0], 4), dtype=torch.float32)
        #features[:, :3] = torch.from_numpy(colors).contiguous()

       # assert features.dtype == torch.float32, f"特征类型错误: {features.dtype}"
        assert np.all(coords.numpy() == np.floor(points / self.voxel_size)), "坐标计算错误"
        assert (coords.data_ptr() % 16) == 0, f"坐标内存未对齐: {coords.data_ptr()}"
        assert (features.data_ptr() % 16) == 0, f"特征内存未对齐: {features.data_ptr()}"

        def check_alignment(tensor, name):
            addr = tensor.data_ptr()
            if addr % 16 != 0:
                raise ValueError(f"{name} 内存未对齐 (地址: {addr}, 余数: {addr % 16})")

        check_alignment(coords, "坐标张量")
        check_alignment(features, "特征张量")
        return {
            'coordinates': torch.from_numpy(aligned_coords[:, :3]).contiguous(),
            'features': torch.from_numpy(aligned_features[:, :3]).contiguous()
        }
      
    @staticmethod
    def minkowski_collate_fn(batch):
        print(f"[DEBUG] 单样本特征维度: {batch[0]['features'].shape}")  # 应为[N,3]
        coords_batch, feats_batch = ME.utils.sparse_collate(...)
        print(f"[DEBUG] 合并后特征维度: {feats_batch.shape}")  # 应为[N_total,3]
        return {'coordinates': coords_batch, 'features': feats_batch}
        #coordinates = [item['coordinates'].to(torch.int32).contiguous() for item in batch]
        #features = [item['features'].contiguous() for item in batch]
    
       # coords_batch, feats_batch = ME.utils.sparse_collate(
        #    coordinates,
        #    features,
        #    dtype=torch.float32
       # )
    
       # return {
        #    'coordinates': coords_batch.contiguous(),
        #    'features': feats_batch.contiguous()
        #}


if __name__ == "__main__":
    dataset = IVFBDataset("/data/maoxiaolong/dancer_vox11", voxel_size=0.01)
    sample = dataset[0]
    
    print("单个样本验证：")
    print("坐标形状:", sample['coordinates'].shape)  # 应为 [N, 3]
    print("坐标类型:", sample['coordinates'].dtype)  # torch.int32
    print("特征形状:", sample['features'].shape)    # 应为 [N, 3]
    print("特征范围:", sample['features'].min(), sample['features'].max())  # 应在[0,1]之间
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=IVFBDataset.minkowski_collate_fn)
    batch = next(iter(loader))
    
    print("\n批次验证：")
    print("坐标形状:", batch['coordinates'].shape)  # 应为 [N_total, 4]
    print("坐标类型:", batch['coordinates'].dtype)  # torch.int32
    print("特征形状:", batch['features'].shape)    # 应为 [N_total, 3]
