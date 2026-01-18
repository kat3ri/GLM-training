# Quick Reference Guide

## Installation

```bash
git clone https://github.com/kat3ri/GLM-training.git
cd GLM-training
pip install -e .
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git

# Optional (Linux/Mac only, not required on Windows):
pip install -r requirements-optional.txt
```

**Windows Users:** Skip `requirements-optional.txt` if you encounter errors with `aio.lib` or `cufile.lib`.

## Basic Commands

### Train Both Components (T2I)
```bash
python train.py --config configs/t2i_training.yaml --train_both
```

### Train Only Autoregressive Model
```bash
python train_ar.py --config configs/t2i_training.yaml
```

### Train Only DiT Decoder
```bash
python train_dit.py --config configs/t2i_training.yaml
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 train.py --config configs/t2i_training.yaml
```

### I2I Training
```bash
python train.py --config configs/i2i_training.yaml --mode i2i
```

## Configuration Quick Reference

### Model Settings
```yaml
model:
  name: "zai-org/GLM-Image"
  component: "both"  # "ar", "dit", or "both"
  torch_dtype: "bfloat16"
```

### Training Settings
```yaml
training:
  mode: "t2i"  # or "i2i"
  batch_size: 2
  num_epochs: 10
  learning_rate: 1e-5
  use_reward: true
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

### Reward Settings
```yaml
reward:
  enabled: true
  metrics:
    lpips: 0.4          # Perceptual similarity
    aesthetic: 0.3      # Aesthetic quality
    text_accuracy: 0.3  # Text rendering accuracy
  grpo:
    num_samples: 4      # Samples per GRPO iteration
    kl_coef: 0.05
    clip_range: 0.2
```

### Data Settings
```yaml
data:
  prompts_file: "data/t2i/prompts.txt"
  target_images_dir: "data/t2i/target_images"
  image_size:
    height: 1024
    width: 1024
```

### Multi-GPU Settings
```yaml
distributed:
  enabled: true
  backend: "nccl"
```

### Logging Settings
```yaml
logging:
  log_dir: "logs"
  log_interval: 10
  save_interval: 500
  use_tensorboard: true
  use_wandb: false
```

## Common Use Cases

### Fine-tune for Specific Art Style
1. Collect 50-100 images in target style
2. Generate descriptive prompts for each
3. Configure with higher aesthetic reward weight
4. Train for 5-10 epochs

### Improve Text Rendering
1. Create dataset with images containing text
2. Include quoted text in prompts
3. Increase text_accuracy reward weight
4. Train DiT component separately for faster iteration

### Domain-Specific I2I
1. Prepare source/target image pairs
2. Write specific edit instructions
3. Use structure_preservation reward for consistency
4. Train with lower learning rate (5e-6)

## Monitoring

### View TensorBoard
```bash
tensorboard --logdir logs/
```

### Check GPU Usage
```bash
nvidia-smi -l 1
```

### Monitor Training Progress
```bash
tail -f logs/train_*.log
```

## Troubleshooting

### Windows Installation Error (aio.lib/cufile.lib)
- **Error**: `LINK : fatal error LNK1181: cannot open input file 'aio.lib'` or `'cufile.lib'`
- **Cause**: The `bitsandbytes` package doesn't support Windows
- **Solution**: Use the main `requirements.txt` only, skip `requirements-optional.txt`
- **Note**: `bitsandbytes` is not required for core functionality

### OOM Error
- Reduce `batch_size` to 1
- Enable `gradient_checkpointing`
- Train components separately
- Reduce image size (e.g., 512x512)

### Slow Training
- Increase `num_workers` in data config
- Use multiple GPUs
- Enable mixed precision (`bf16` or `fp16`)

### Poor Results
- Increase training epochs
- Adjust reward metric weights
- Collect more/better training data
- Try different learning rates

## File Structure

```
GLM-training/
├── configs/              # YAML configuration files
├── glm_training/        # Main package
│   ├── data/           # Dataset classes
│   ├── models/         # Model wrappers
│   ├── rewards/        # Reward calculation
│   ├── trainers/       # Training logic
│   └── utils/          # Utilities
├── data/               # Training data
├── train.py            # Main training script
├── train_ar.py         # AR-only training
└── train_dit.py        # DiT-only training
```

## Important Notes

- Images must be divisible by 32 (e.g., 1024, 512, 2048)
- Requires 80GB+ GPU for full model training
- AR component: ~40GB, DiT component: ~50GB
- Use gradient checkpointing for memory efficiency
- GRPO requires multiple samples per iteration (slower but better quality)

## Next Steps

1. Prepare your dataset (see `data/README.md`)
2. Customize config (see `configs/`)
3. Start with small batch size to test
4. Monitor initial training in TensorBoard
5. Adjust hyperparameters based on results
6. Scale up batch size and epochs

## Support

- Documentation: `README.md`, `examples/USAGE.md`
- Contributing: `CONTRIBUTING.md`
- Issues: GitHub Issues
