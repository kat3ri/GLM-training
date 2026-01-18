# GLM-Training Implementation Summary

## Overview

This repository provides a comprehensive training framework for [GLM-Image](https://github.com/zai-org/GLM-Image), implementing all requested features:

### ✅ Core Requirements Met

1. **Reward-Based Training Mode**
   - Implemented GRPO (Group Relative Policy Optimization) algorithm
   - Multiple reward metrics: LPIPS, aesthetic quality, text accuracy, structure preservation
   - Target image-based reward computation
   - Configurable reward weights per component (AR/DiT)

2. **Multi-GPU Support**
   - DistributedDataParallel (DDP) implementation
   - Support for NCCL and Gloo backends
   - Automatic rank and world size detection
   - Synchronized training across multiple GPUs

3. **Separate Component Training**
   - Train Autoregressive (AR) model independently
   - Train DiT (Diffusion Decoder) independently
   - Train both components together
   - Component-specific learning rates

4. **Dual Training Modes**
   - Text-to-Image (t2i) finetuning
   - Image-to-Image (i2i) finetuning
   - Mode-specific datasets and data loaders
   - Configurable via YAML files

5. **Target Image Rewards**
   - Pre-created target images as training objectives
   - Perceptual similarity metrics (LPIPS)
   - OCR-based text accuracy measurement
   - Structure preservation for i2i

## Project Structure

```
GLM-training/
├── configs/                    # YAML configuration files
│   ├── t2i_training.yaml      # Text-to-image training config
│   └── i2i_training.yaml      # Image-to-image training config
│
├── glm_training/              # Core package
│   ├── data/                  # Dataset implementations
│   │   ├── t2i_dataset.py    # T2I dataset loader
│   │   └── i2i_dataset.py    # I2I dataset loader
│   │
│   ├── models/                # Model wrappers
│   │   └── glm_wrapper.py    # GLM-Image model wrapper
│   │
│   ├── rewards/               # Reward computation
│   │   └── reward_calculator.py  # GRPO reward calculator
│   │
│   ├── trainers/              # Training logic
│   │   ├── base_trainer.py   # Base trainer with DDP support
│   │   ├── reward_trainer.py # Reward-based trainer (GRPO)
│   │   ├── ar_trainer.py     # AR-specific trainer
│   │   └── dit_trainer.py    # DiT-specific trainer
│   │
│   └── utils/                 # Utilities
│       ├── distributed.py    # Multi-GPU utilities
│       └── logging.py        # TensorBoard/W&B logging
│
├── data/                      # Training data structure
│   ├── t2i/                  # T2I data
│   │   ├── prompts.txt
│   │   ├── target_images/
│   │   └── eval_target_images/
│   └── i2i/                  # I2I data
│       ├── source_images/
│       ├── prompts.txt
│       ├── target_images/
│       └── eval_source_images/
│
├── train.py                   # Main training script
├── train_ar.py               # AR-only training script
├── train_dit.py              # DiT-only training script
├── test_installation.py      # Installation verification script
│
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick reference guide
├── CONTRIBUTING.md          # Contribution guidelines
├── examples/USAGE.md        # Detailed usage examples
│
├── requirements.txt          # Python dependencies
└── setup.py                 # Package setup
```

## Key Features

### 1. Reward-Based Training (GRPO)

The framework implements Group Relative Policy Optimization with:
- Multiple samples per prompt for variance reduction
- Relative reward computation within sample groups
- PPO-style clipping for stability
- Component-specific reward weights (AR vs DiT)

**Reward Metrics:**
- **LPIPS**: Perceptual similarity to target images
- **Aesthetic**: Simple aesthetic quality scoring (extensible)
- **Text Accuracy**: OCR-based text rendering verification
- **Structure Preservation**: SSIM-based structure matching (i2i only)

### 2. Multi-GPU Training

Full DDP implementation with:
- Automatic GPU detection and initialization
- Rank-aware logging (only main process logs)
- Synchronized gradient updates
- Efficient data parallelism via DistributedSampler

**Usage:**
```bash
torchrun --nproc_per_node=4 train.py --config configs/t2i_training.yaml
```

### 3. Component-Specific Training

Train AR and DiT components separately or together:

**Autoregressive Only:**
```bash
python train_ar.py --config configs/t2i_training.yaml
```

**DiT Only:**
```bash
python train_dit.py --config configs/t2i_training.yaml
```

**Both Components:**
```bash
python train.py --config configs/t2i_training.yaml --train_both
```

**Benefits:**
- Faster iteration (train smaller component first)
- Memory efficiency (fit larger batches)
- Component-specific learning rates
- Targeted improvements

### 4. Dual Training Modes

**Text-to-Image (T2I):**
- Prompts → Images
- Focus on semantic understanding
- Text rendering accuracy

**Image-to-Image (I2I):**
- Source Images + Edit Prompts → Edited Images
- Structure preservation
- Style transfer capabilities
- Multi-subject consistency

### 5. Memory Optimizations

- Gradient checkpointing support
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Efficient data loading

## Configuration

All training parameters are configurable via YAML files:

```yaml
# Example: configs/t2i_training.yaml

model:
  component: "both"  # "ar", "dit", or "both"

training:
  mode: "t2i"
  batch_size: 2
  use_reward: true
  gradient_checkpointing: true
  mixed_precision: "bf16"

reward:
  enabled: true
  metrics:
    lpips: 0.4
    aesthetic: 0.3
    text_accuracy: 0.3
  grpo:
    num_samples: 4
    kl_coef: 0.05
    clip_range: 0.2

distributed:
  enabled: true
  backend: "nccl"
```

## Usage Examples

### Basic T2I Training
```bash
python train.py --config configs/t2i_training.yaml
```

### I2I Training with Multi-GPU
```bash
torchrun --nproc_per_node=4 train.py --config configs/i2i_training.yaml --mode i2i
```

### Sequential Component Training
```bash
# Step 1: Train AR model
python train_ar.py --config configs/t2i_training.yaml

# Step 2: Train DiT decoder
python train_dit.py --config configs/t2i_training.yaml
```

## Testing Installation

Verify your setup with the included test script:
```bash
python test_installation.py
```

This checks:
- Required package imports
- GLM-training module installation
- Configuration file validity
- Data structure
- GPU availability
- Training script accessibility

## Documentation

- **README.md**: Complete overview and installation
- **QUICKSTART.md**: Quick reference for common tasks
- **examples/USAGE.md**: Detailed usage examples and troubleshooting
- **CONTRIBUTING.md**: Guidelines for contributors
- **data/README.md**: Data preparation instructions

## Hardware Requirements

- **Minimum**: Single GPU with 80GB VRAM (e.g., A100)
- **Recommended**: 4x GPUs with 80GB VRAM for distributed training
- **Component Training**:
  - AR only: ~40GB VRAM
  - DiT only: ~50GB VRAM
  - Both: ~80GB VRAM

## Dependencies

Core dependencies:
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.40.0
- Diffusers ≥ 0.27.0
- Accelerate ≥ 0.20.0

Optional dependencies:
- bitsandbytes ≥ 0.41.0 (Linux/Mac only, for 8-bit/4-bit quantization)

See `requirements.txt` for complete list. Optional dependencies are in `requirements-optional.txt`.

## Design Decisions

1. **YAML Configuration**: Easy to version control and modify
2. **Modular Trainers**: Base trainer with specialized subclasses
3. **Reward Calculator**: Extensible design for custom metrics
4. **Separate Scripts**: train.py, train_ar.py, train_dit.py for clarity
5. **DDP over DataParallel**: Better performance and scalability

## Extensibility

The framework is designed to be extensible:

1. **Custom Reward Metrics**: Extend `RewardCalculator` class
2. **New Training Modes**: Subclass `BaseTrainer` or `RewardTrainer`
3. **Data Augmentation**: Extend dataset classes
4. **Custom Schedulers**: Modify `_build_scheduler` method

## Future Enhancements

Potential additions (see CONTRIBUTING.md):
- LoRA/QLoRA support for efficient fine-tuning
- Additional reward models (CLIP, aesthetic predictors)
- Hyperparameter search utilities
- Better visualization tools
- Integration with more logging platforms
- Quantization support (8-bit, 4-bit) - Note: bitsandbytes is available as an optional dependency for Linux/Mac users

## Testing Status

✅ Python module structure verified
✅ Configuration files validated
✅ Import paths tested
✅ Data structure created
✅ Scripts are executable

⚠️ Note: Actual training requires:
- Installing dependencies (see requirements.txt)
- Downloading GLM-Image model
- Preparing training data with target images

## License

Same as GLM-Image project (see LICENSE file)

## Acknowledgments

- Built for [GLM-Image](https://github.com/zai-org/GLM-Image) by zai-org
- Uses [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Uses [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- GRPO algorithm inspired by reinforcement learning literature

## Contact

For issues, questions, or contributions:
- GitHub Issues: https://github.com/kat3ri/GLM-training/issues
- See CONTRIBUTING.md for contribution guidelines
