import h5py
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms

class PointCloudAttributeDataset(Dataset):
    def __init__(self, 
                 shapenet_h5_path, 
                 coco_annotation_path,
                 coco_image_dir,
                 modelnet40_path=None,
                 quantize_bits=8,
                 img_size=256,
                 rotation_augmentation=True):
        """
        Args:
            shapenet_h5_path: ShapeNet h5文件路径
            coco_annotation_path: COCO标注文件路径
            coco_image_dir: COCO图像目录
            modelnet40_path: ModelNet40数据路径(可选)
            quantize_bits: 坐标量化位数
            img_size: 投影图像尺寸
            rotation_augmentation: 是否启用随机旋转
        """
        with h5py.File(shapenet_h5_path, 'r') as f:
            self.point_clouds = f['points'][:]
            self.normals = f['normals'][:] if 'normals' in f else None

        self.coco = COCO(coco_annotation_path)
        self.coco_image_dir = coco_image_dir
        self.img_ids = list(self.coco.imgs.keys())
        
        self.rotation_aug = rotation_augmentation
        self.quant_scale = (2**quantize_bits - 1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        self.modelnet_data = None
        if modelnet40_path:
            with h5py.File(modelnet40_path, 'r') as f:
                self.modelnet_data = {
                    'points': f['points'][:],
                    'labels': f['labels'][:]
                }

    def __len__(self):
        return len(self.point_clouds) + (len(self.modelnet_data['points']) 
                if self.modelnet_data else 0)

    def _random_rotation(self, points):
        """随机旋转点云"""
        theta = np.random.uniform(0, 2*np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(points, rot_matrix.T)

    def _quantize_coords(self, coords):
        """坐标量化到8位整数"""
        return (coords * self.quant_scale).astype(np.uint8)

    def _project_color(self, points, normals, img):

        proj_weights = normals[:, 2].clip(0, 1)
        img_coords = (points[:, :2] * [img.size[0], img.size[1]]).astype(int)
        
        img_coords[:, 0] = np.clip(img_coords[:, 0], 0, img.size[0]-1)
        img_coords[:, 1] = np.clip(img_coords[:, 1], 0, img.size[1]-1)
        
        colors = np.array(img)[img_coords[:, 1], img_coords[:, 0]]
        return colors * proj_weights[:, None]

    def __getitem__(self, idx):
        
        if idx < len(self.point_clouds):
            points = self.point_clouds[idx]
            normals = self.normals[idx] if self.normals is not None else None
        else:
            m_idx = idx - len(self.point_clouds)
            points = self.modelnet_data['points'][m_idx]
            normals = None 


        if self.rotation_aug:
            points = self._random_rotation(points)
            if normals is not None:
                normals = self._random_rotation(normals)


        quantized_points = self._quantize_coords(points)


        img_id = np.random.choice(self.img_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.coco_image_dir}/{img_info['file_name']}"
        img = Image.open(img_path).convert('RGB')

        if normals is not None:
            colors = self._project_color(points, normals, img)
        else:
            colors = np.zeros((len(points), 3)) 


        points_tensor = torch.from_numpy(quantized_points).float()
        colors_tensor = torch.from_numpy(colors).float()
        img_tensor = self.img_transform(img)

        return {
            'points': points_tensor,
            'colors': colors_tensor,
            'image': img_tensor
        }

if __name__ == "__main__":
    dataset = PointCloudAttributeDataset(
        shapenet_h5_path="path/to/shapenet.h5",
        coco_annotation_path="path/to/coco/annotations.json",
        coco_image_dir="path/to/coco/images",
        modelnet40_path="path/to/modelnet40.h5"
    )
    
    sample = dataset[0]
    print(f"Point cloud shape: {sample['points'].shape}")
    print(f"Color attributes shape: {sample['colors'].shape}")
    print(f"Projection image shape: {sample['image'].shape}")
