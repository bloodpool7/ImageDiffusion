# Image Diffusion - DDPM from Scratch

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for high-quality image generation. This project demonstrates the complete pipeline from training to sampling, with support for both unconditional and class-conditional generation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

## Features

- **Pure PyTorch Implementation**: Built entirely from scratch using PyTorch, no high-level diffusion libraries
- **Class-Conditional Generation**: Generate images conditioned on specific classes with classifier-free guidance
- **Multiple Datasets**: Pre-configured support for CIFAR-10 and Flowers datasets
- **EMA for Quality**: Exponential Moving Average (EMA) integration for improved sample quality
- **Flexible Architecture**: Configurable U-Net with attention mechanisms and residual blocks
- **Ready-to-Use Scripts**: Simple training and sampling workflows

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ImageDiffusion.git
cd ImageDiffusion
```

2. **Set up the environment**:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training Your Model

The `train.py` script handles the complete training pipeline with sensible defaults:

```bash
python train.py
```

**Configuration Options** (edit in `train.py`):

- `dataset_name`: Choose between `'cifar10'` or `'flowers'`
- `image_size`: Resolution of generated images (e.g., `32`, `64`)
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `sample_every`: Generate sample images every N epochs
- `save_every`: Save checkpoint every N epochs

**Training Features**:
- Automatic checkpoint saving to `./checkpoints/`
- Periodic sample generation to monitor progress
- Resume training from the latest checkpoint
- Mixed precision training support (CUDA/MPS)

**Example Training Configuration**:
```python
# In train.py
dataset_name = 'flowers'  # Use flowers dataset
image_size = 64           # Generate 64x64 images
num_epochs = 2400         # Train for 2400 epochs
sample_every = 300        # Generate samples every 300 epochs
```

### Generating Images

Once you have a trained model, use the `example.ipynb` notebook for easy sampling.

#### Using the Example Notebook

1. **Open the notebook**:
```bash
jupyter notebook example.ipynb
```

2. **Basic Sampling**:
```python
import utils
import torch

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Configuration
num_samples = 16
image_shape = (3, 64, 64)  # Channels, Height, Width

# Load checkpoint
_, _, ema, conditioner, diffusion_model = utils.load_from_checkpoint(
    "./checkpoints/your_checkpoint_name", 
    device=device
)

# Generate images
output = diffusion_model.sample(
    ema.ema_model, 
    conditioner, 
    num_samples=num_samples, 
    device=device, 
    image_shape=image_shape
)

# Display results
_ = utils.save_images(output, show=True)
```

#### Advanced Sampling Options

**Unconditional Generation**:
```python
# Generate random images without class conditioning
output = diffusion_model.sample(
    ema.ema_model,
    conditioner,
    num_samples=16,
    device=device,
    image_shape=(3, 64, 64),
    labels=None,  # No class conditioning
    guidance_scale=0.0
)
```

**Class-Conditional Generation**:
```python
# Generate specific classes (for Flowers dataset: 0=daisy, 1=dandelion, 2=rose, 3=sunflower, 4=tulip)
labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 4], device=device)

output = diffusion_model.sample(
    ema.ema_model,
    conditioner,
    num_samples=16,
    device=device,
    image_shape=(3, 64, 64),
    labels=labels,
    guidance_scale=7.0  # Higher values = stronger class conditioning
)
```

**Guidance Scale Effects**:
- `guidance_scale=0.0`: Fully unconditional (ignores labels)
- `guidance_scale=1.0`: Conditional without guidance
- `guidance_scale=7.0`: Strong guidance (recommended, more realistic to class)
- Higher values increase class fidelity but may reduce diversity

## Project Structure

```
ImageDiffusion/
├── model.py            # UNet architecture and Conditioner
├── diffusion.py        # DDPM forward/reverse diffusion and sampling
├── train.py            # Training script with checkpoint management
├── dataset.py          # Dataset loaders (CIFAR-10, Flowers)
├── utils.py            # Helper functions (save/load checkpoints, visualize)
├── example.ipynb       # Jupyter notebook for sampling demonstrations
├── requirements.txt    # Python dependencies
├── data/               # Dataset directory (auto-downloaded)
└── checkpoints/        # Saved model checkpoints
```

## Key Components

### Model Architecture
- **UNet with Attention**: Multi-scale architecture with self-attention at key resolutions
- **Residual Blocks**: Deep residual connections with group normalization
- **Conditional Injection**: Time and class embeddings via adaptive group normalization

### Diffusion Process
- **Forward Process**: Gradual noise addition over 1000 timesteps
- **Reverse Process**: DDPM denoising with learned noise prediction
- **Sampling Methods**: Classifier-free guidance for controllable generation

### Training Features
- **EMA**: Exponential moving average of model weights for better sample quality
- **Automatic Mixed Precision**: Faster training on CUDA GPUs
- **Resume Capability**: Continue training from any saved checkpoint

## Tips for Best Results

1. **Training Duration**: More epochs = better quality
   - CIFAR-10: ~1000-2000 epochs for good results
   - Flowers (64x64): ~2000-3000 epochs for high quality

2. **Guidance Scale**: Experiment with values between 5.0-10.0
   - Higher = more class-specific but less diverse
   - Lower = more diverse but less accurate to class

3. **Checkpoint Selection**: Use the EMA model (automatically loaded) for sampling
   - EMA weights produce higher quality samples than regular checkpoints

4. **Hardware Recommendations**:
   - GPU highly recommended (CUDA or Apple Silicon MPS)
   - CPU sampling takes ~5-10 minutes per 16 samples

## Datasets

### CIFAR-10
- 10 classes of 32×32 color images
- Automatically downloaded on first run
- Classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

### Flowers
- 5 classes of flower images
- Place your dataset in `./data/flowers/` with class subdirectories
- Classes: daisy, dandelion, rose, sunflower, tulip

## Customization

### Adding Your Own Dataset

1. Create a dataset class in `dataset.py` following the existing patterns
2. Update `utils.py` to include your dataset in `get_train_val_test_loaders()`
3. Adjust `num_classes` in `train.py` when creating the Conditioner

### Modifying Architecture

Edit `train.py` to customize the UNet:
```python
unet = model.UNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,        # Increase for more capacity
    embedding_dim=512,
    channel_mults=(1, 2, 4, 8),  # Add more stages for larger images
    device=device
)
```

## Notes

- Checkpoints are saved in `./checkpoints/{experiment_name}/`
- Generated samples during training appear in `./checkpoints/{experiment_name}/samples/`
- Each checkpoint includes model weights, optimizer state, EMA weights, and training metadata