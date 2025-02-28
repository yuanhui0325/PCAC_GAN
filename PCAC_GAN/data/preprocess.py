import numpy as np
from plyfile import PlyData

def ply_to_npz(ply_path, npz_path, voxel_size=0.01):
    """将PLY文件预处理为压缩格式"""
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    coords = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0
    
    quantized_coords = np.floor(coords / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(quantized_coords, axis=0, return_index=True)
    
    np.savez_compressed(
        npz_path,
        coords=quantized_coords[unique_idx],
        colors=colors[unique_idx]
    )
