# PCAC_GAN
During the process of machine updates and migration, the original code was unfortunately lost. This version is a re-implemented codebase based on the original design concepts and experimental results. While we have made every effort to maintain consistency with the original code, there may be slight discrepancies in the results due to differences in certain implementation details.



## Code Structure

├── configs/                  # Configuration files

│   └── base.yaml             # Main training configuration

├── core/                     # Core compression components

│   ├── entropy_coder.py      # Arithmetic coding implementation

│   ├── hyper_prior.py        # Hyperprior network

│   └── avrpm.py              # Adaptive voxel residual prediction module

├── data/

│   ├── ivfb_loader.py        # IVFBDataset implementation

│   └── preprocess.py         # PLY to NPZ conversion

├── models/

│   ├── generator.py          # PCACGenerator (main compression network)

│   └── discriminator.py      # PointCloudDiscriminator

├── losses/                   # Loss functions

│   ├── adversarial.py        # GAN implementation

│   └── rate_distortion.py    # Rate-distortion loss

├── utils/

│   ├── metrics.py            # PSNR/SSIM/BPP calculations

│   └── transforms.py         # Color space conversions

├── train.py                  # Main training script

└── evaluate.py               # Model evaluation script

└── dataset_gen.py             # Shapenet+CoCo

## Getting Started

### Installation
```bash
# Base environment
conda create -n pcac python=3.8
conda activate pcac

# Install PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install MinkowskiEngine
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"

# Install other dependencies
pip install open3d plyfile pycocotools tqdm

Training
Prepare dataset in PLY format under /data/yourpath/
Update dataset.root_dir in configs/base.yaml
Start training:

python train.py --config configs/base.yaml \
    --batch_size 32 \
    --lr_g 0.0001 \
    --lr_d 0.0004

Evaluation


models: https://pan.baidu.com/s/1-5TRTShyW5pYBSNiDRGNoA  8a97 


python evaluate.py \
    --weights path/to/checkpoint \
    --output results.json \
    --batch_size 8
