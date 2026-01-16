# Example Usage Guide

This guide provides practical examples of using the GLM-Image training framework.

## Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/kat3ri/GLM-training.git
cd GLM-training

# Install dependencies
pip install -e .

# Install GLM-Image dependencies
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
```

## Preparing Your Data

### Text-to-Image (T2I) Training

1. Create a `prompts.txt` file with one prompt per line:
```
A beautiful sunset over mountains
"Hello World" text on a colorful background
A futuristic city at night
```

2. Create corresponding target images:
- Name them `0.png`, `1.png`, `2.png`, etc.
- Place them in the `target_images` directory
- Each image corresponds to the prompt at the same line number

Directory structure:
```
data/t2i/
├── prompts.txt
└── target_images/
    ├── 0.png
    ├── 1.png
    └── 2.png
```

### Image-to-Image (I2I) Training

1. Create source images: `0.png`, `1.png`, etc. in `source_images/`
2. Create a `prompts.txt` file with edit instructions
3. Create target images showing the desired result

Directory structure:
```
data/i2i/
├── source_images/
│   ├── 0.png
│   └── 1.png
├── prompts.txt
└── target_images/
    ├── 0.png
    └── 1.png
```

## Training Examples

### Example 1: Basic T2I Training

```bash
python train.py --config configs/t2i_training.yaml
```

### Example 2: I2I Training with Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py --config configs/i2i_training.yaml
```

### Example 3: Train Only Autoregressive Model

```bash
python train_ar.py --config configs/t2i_training.yaml
```

### Example 4: Train Only DiT Decoder

```bash
python train_dit.py --config configs/t2i_training.yaml
```

### Example 5: Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
model:
  name: "zai-org/GLM-Image"
  component: "both"

training:
  mode: "t2i"
  batch_size: 1  # Reduce for limited GPU memory
  num_epochs: 5
  learning_rate: 5e-6

reward:
  enabled: true
  metrics:
    lpips: 0.5
    aesthetic: 0.5

data:
  prompts_file: "my_data/prompts.txt"
  target_images_dir: "my_data/targets"
```

Run with:
```bash
python train.py --config my_config.yaml
```

## Memory Optimization Tips

If you encounter OOM errors:

1. **Reduce batch size**:
```yaml
training:
  batch_size: 1
```

2. **Enable gradient checkpointing**:
```yaml
training:
  gradient_checkpointing: true
```

3. **Use gradient accumulation**:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 8
```

4. **Train components separately**:
```bash
# Train AR first (uses less memory)
python train_ar.py --config configs/t2i_training.yaml

# Then train DiT
python train_dit.py --config configs/t2i_training.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006

### Weights & Biases

Enable in config:
```yaml
logging:
  use_wandb: true
  wandb_project: "my-glm-training"
```

## Resuming Training

To resume from a checkpoint:

```yaml
checkpoint:
  resume_from: "checkpoints/checkpoint_step_1000.pt"
```

## Evaluation

The framework automatically evaluates during training if enabled:

```yaml
evaluation:
  enabled: true
  eval_prompts_file: "data/t2i/eval_prompts.txt"
  eval_target_images_dir: "data/t2i/eval_target_images"
  num_eval_samples: 10
```

## Advanced: Custom Reward Functions

Create a custom reward calculator:

```python
# my_rewards.py
from glm_training.rewards import RewardCalculator

class MyCustomRewardCalculator(RewardCalculator):
    def _compute_custom_metric(self, images, targets):
        # Your custom logic here
        return scores

# In training script
from my_rewards import MyCustomRewardCalculator
trainer.reward_calculator = MyCustomRewardCalculator(...)
```

## Troubleshooting

### Issue: "CUDA out of memory"
- Reduce batch size
- Enable gradient checkpointing
- Train components separately

### Issue: "No target image found"
- Check that image files are named correctly (0.png, 1.png, etc.)
- Verify image paths in config
- Check file permissions

### Issue: "Distributed training not working"
- Ensure NCCL is properly installed
- Check that all GPUs are visible: `nvidia-smi`
- Verify network configuration for multi-node training

## Next Steps

- Experiment with different reward metrics
- Adjust learning rates for each component
- Try different numbers of GRPO samples
- Fine-tune on domain-specific data
