# GLM-Image Training Repository

A comprehensive training framework for [GLM-Image](https://github.com/zai-org/GLM-Image), supporting reward-based training, multi-GPU setups, and separate training of DiT and autoregressive model components.

## Features

- **Reward-Based Training**: Implements GRPO (Group Relative Policy Optimization) algorithm with pre-created target images for reward computation
- **Multi-GPU Support**: Distributed Data Parallel (DDP) training across multiple GPUs
- **Modular Training**: Train DiT (Diffusion Decoder) and Autoregressive components separately or together
- **Dual Mode Support**: 
  - Text-to-Image (t2i) finetuning
  - Image-to-Image (i2i) finetuning
- **Flexible Configuration**: YAML-based configuration system for easy customization

## Architecture Overview

GLM-Image uses a hybrid autoregressive + diffusion decoder architecture:
- **Autoregressive Generator**: 9B-parameter model for generating compact visual token encodings
- **Diffusion Decoder**: 7B-parameter DiT architecture for latent-space image decoding with Glyph Encoder

## Installation

```bash
# Clone the repository
git clone https://github.com/kat3ri/GLM-training.git
cd GLM-training

# Install PyTorch with CUDA support (adjust URL for your CUDA version)
# Example for CUDA 12.1 (check the official site for the exact command for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies (includes git-based transformers and diffusers)
pip install -r requirements.txt

# Optional: Install additional dependencies (Linux/Mac only)
# Note: bitsandbytes may not work on Windows
pip install -r requirements-optional.txt
```

### Windows Installation Note

On Windows, some optional dependencies (like `bitsandbytes`) may fail to install due to missing system libraries (`aio.lib`, `cufile.lib`). These packages are not required for core functionality. If you encounter installation errors, you can safely skip the optional dependencies.

## Quick Start

### 1. Prepare Your Dataset

Organize your training data in the following structure:

```
data/
├── t2i/
│   ├── prompts.txt          # Text prompts for t2i
│   └── target_images/       # Target images for reward computation
│       ├── 0.png
│       ├── 1.png
│       └── ...
└── i2i/
    ├── source_images/       # Source images for i2i
    │   ├── 0.png
    │   └── ...
    ├── prompts.txt          # Edit prompts
    └── target_images/       # Target images for reward computation
        ├── 0.png
        └── ...
```

### 2. Configure Training

Edit the configuration file for your use case:

```bash
# For t2i training
cp configs/t2i_training.yaml configs/my_t2i_config.yaml

# For i2i training
cp configs/i2i_training.yaml configs/my_i2i_config.yaml
```

### 3. Run Training

**Train Autoregressive Model (t2i mode):**
```bash
python train_ar.py --config configs/my_t2i_config.yaml
```

**Train DiT Decoder (t2i mode):**
```bash
python train_dit.py --config configs/my_t2i_config.yaml
```

**Train Both Components (i2i mode):**
```bash
python train.py --config configs/my_i2i_config.yaml --mode i2i
```

**Multi-GPU Training:**
```bash
torchrun --nproc_per_node=4 train.py --config configs/my_config.yaml
```

## Training Modes

### Reward-Based Training

The reward-based training uses GRPO algorithm with target images:

1. **Autoregressive Rewards**: Focus on aesthetics and semantic alignment
2. **Decoder Rewards**: Target detail fidelity and text accuracy

Rewards are computed by comparing generated images with target images using:
- Image similarity metrics (LPIPS, SSIM)
- Aesthetic quality scores
- Text rendering accuracy (OCR-based)

### Training Components

**Train Only Autoregressive Model:**
```bash
python train_ar.py --config configs/t2i_training.yaml
```

**Train Only DiT Decoder:**
```bash
python train_dit.py --config configs/t2i_training.yaml
```

**Train Both (End-to-End):**
```bash
python train.py --config configs/t2i_training.yaml --train_both
```

## Configuration

Key configuration parameters in YAML files:

```yaml
# Model settings
model:
  name: "zai-org/GLM-Image"
  component: "both"  # Options: "ar", "dit", "both"
  
# Training settings
training:
  mode: "t2i"  # Options: "t2i", "i2i"
  batch_size: 2
  num_epochs: 10
  learning_rate: 1e-5
  use_reward: true
  
# Multi-GPU settings
distributed:
  enabled: true
  backend: "nccl"
  
# Reward settings
reward:
  enabled: true
  target_images_dir: "data/t2i/target_images"
  metrics:
    - lpips
    - aesthetic
    - text_accuracy
  weights:
    lpips: 0.4
    aesthetic: 0.3
    text_accuracy: 0.3
```

## Project Structure

```
GLM-training/
├── configs/              # Configuration files
│   ├── t2i_training.yaml
│   └── i2i_training.yaml
├── glm_training/        # Core training modules
│   ├── trainers/        # Training logic
│   │   ├── base_trainer.py
│   │   ├── ar_trainer.py
│   │   ├── dit_trainer.py
│   │   └── reward_trainer.py
│   ├── data/            # Data loading utilities
│   │   ├── t2i_dataset.py
│   │   └── i2i_dataset.py
│   ├── models/          # Model wrappers
│   │   └── glm_wrapper.py
│   ├── rewards/         # Reward computation
│   │   └── reward_calculator.py
│   └── utils/           # Utility functions
│       ├── distributed.py
│       └── logging.py
├── train.py             # Main training script
├── train_ar.py          # AR-only training script
├── train_dit.py         # DiT-only training script
├── requirements.txt
└── README.md
```

## Advanced Usage

### Custom Reward Functions

You can define custom reward functions by extending the `RewardCalculator` class:

```python
from glm_training.rewards import RewardCalculator

class MyRewardCalculator(RewardCalculator):
    def compute_reward(self, generated_image, target_image, prompt):
        # Your custom reward logic
        return reward_score
```

### Gradient Checkpointing

For memory-efficient training:

```yaml
training:
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

## Hardware Requirements

- **Minimum**: Single GPU with 80GB VRAM (e.g., A100)
- **Recommended**: 4x GPUs with 80GB VRAM each for multi-GPU training
- **Training Mode Requirements**:
  - AR only: ~40GB VRAM
  - DiT only: ~50GB VRAM
  - Both: ~80GB VRAM

## Citation

If you use this training repository, please cite the GLM-Image paper:

```bibtex
@article{glm-image,
  title={GLM-Image: Hybrid Autoregressive and Diffusion Image Generation},
  author={GLM Team},
  year={2024}
}
```

## License

This project is licensed under the same license as GLM-Image. See LICENSE file for details.

## Acknowledgments

- [GLM-Image](https://github.com/zai-org/GLM-Image) by zai-org
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
