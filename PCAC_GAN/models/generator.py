import torch
import torch.nn as nn
import MinkowskiEngine as ME
from core.entropy_coder import EntropyBottleneck
from core.avrpm import AVRPM
from core.decoder import MultiScaleDecoder

class PCACGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.voxel_size = config['dataset']['voxel_size']
        print(f"[DEBUG] 生成器输入通道数: {config.get('in_channels', 3)}")
        
        # 使用标准nn.Sequential包装ME层
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(
                in_channels=64,
                out_channels=3,
                kernel_size=3,
                stride=1,
                dimension=3
            )
        )


    def forward(self, x):
        # 强制内存重新对齐（关键修复）
        def force_align(tensor, align=16):
            aligned = torch.empty(
                tensor.shape[0], 
                (tensor.shape[1] + (align//4 -1)) // (align//4) * (align//4),
                dtype=tensor.dtype,
                device=tensor.device
            ).contiguous()
            aligned[:, :tensor.shape[1]] = tensor
            return aligned
 
        me_coords = x['coordinates'].int().contiguous()
        me_coords = force_align(me_coords)
    
        feats = x['features'].contiguous()
        feats = force_align(feats)

     # 最终对齐验证
        assert (me_coords.data_ptr() % 16) == 0, f"坐标最终未对齐: {me_coords.data_ptr()}"
        assert (feats.data_ptr() % 16) == 0, f"特征最终未对齐: {feats.data_ptr()}"

        sparse_input = ME.SparseTensor(
            features=feats,
            coordinates=me_coords,
            tensor_stride=1
        )
        return self.net(sparse_input)
    '''def forward(self, x):
        me_coords = x['coordinates'].int()  # [N,4] (batch,x,y,z)
        feats = x['features']              # [N,3]

    # 强制内存对齐到16字节（关键修复）
        aligned_coords = torch.zeros(
            me_coords.shape[0], 
            4, 
            dtype=torch.int32, 
            device=me_coords.device
        ).contiguous()
        aligned_coords[:, :4] = me_coords  # 复制原始四维坐标

    # 特征填充到4的倍数维度
        aligned_feats = torch.zeros(
            feats.shape[0], 
            4, 
            dtype=torch.float32, 
            device=feats.device
        ).contiguous()
        aligned_feats[:, :3] = feats

        sparse_input = ME.SparseTensor(
            features=aligned_feats,
            coordinates=aligned_coords,
            tensor_stride=1
        )
        return self.net(sparse_input)'''