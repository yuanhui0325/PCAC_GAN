import open3d as o3d
import numpy as np

def save_as_ply(tensor, filename):
    coords = tensor.C.cpu().numpy()[:, 1:]  # 去除batch维度
    colors = (tensor.F.cpu().numpy() * 255).astype(np.uint8)
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors[..., :3]/255.0)
    
    o3d.io.write_point_cloud(filename, pcd)
