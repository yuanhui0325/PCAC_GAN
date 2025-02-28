import os
import json
import argparse
import torch
from tqdm import tqdm
import MinkowskiEngine as ME
from models.generator import PCACGenerator
from data.ivfb_loader import IVFBDataset
from torch.utils.data import DataLoader

def load_test_config():
    return {
        'dataset': {
            'root_dir': '/data/test_set',
            'voxel_size': 0.01
        },
        'model': {
            'channels': [32, 64, 128],
            'bottleneck_dim': 256
        }
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = load_test_config()
    
    print("=> 初始化模型...")
    model = PCACGenerator(config['model']).to(device)
    
    if os.path.isfile(args.weights):
        print(f"=> 加载模型权重: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise FileNotFoundError(f"模型权重文件 {args.weights} 不存在")

    model.eval() 
    
    print("\n=> 准备测试数据...")
    test_dataset = IVFBDataset(
        root_dir=config['dataset']['root_dir'],
        voxel_size=config['dataset']['voxel_size'],
        is_training=False  
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=minkowski_collate_fn 
    )
    
    print("\n=> 开始评估...")
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device
    )
    
    print("\n=== 测试结果 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存至 {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='点云压缩模型测试')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--output', type=str, default='results.json',
                       help='输出结果文件路径')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='测试批次大小')
    args = parser.parse_args()

    torch.manual_seed(42)
    
    main(args)
#python test.py --weights path/to/model.pth --output test_results.json
