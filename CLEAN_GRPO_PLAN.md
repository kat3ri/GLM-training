# Clean GRPO Implementation Plan

**Date:** January 19, 2026  
**Approach:** Start from scratch with minimal, clean code  
**Primary References:** HuggingFace TRL + nanoGRPO

---

## Philosophy: Minimal and Correct

Instead of fixing broken code, we'll build GRPO correctly from the ground up:

1. **Start Simple** - Minimal working implementation first
2. **Test Everything** - Each component tested before moving on
3. **Follow TRL** - Use proven patterns from production code
4. **Keep it Clean** - Easy to understand and maintain

---

## Phase 1: Minimal GRPO (Week 1)

### Goal: Get GRPO working with absolute minimum code

**What we need:**
1. Generate multiple samples per prompt
2. Compute rewards
3. Calculate advantages (z-score)
4. PPO loss with clipping
5. Basic training loop

**What we DON'T need yet:**
- ❌ Reference model (use old policy from generation)
- ❌ Complex memory optimization
- ❌ Multi-GPU support (start single GPU)
- ❌ Advanced logging

### File Structure
```
glm_training/
├── grpo/                      # NEW: Clean GRPO package
│   ├── __init__.py
│   ├── trainer.py            # NEW: Minimal GRPO trainer
│   ├── loss.py               # NEW: GRPO loss functions
│   └── utils.py              # NEW: Helper functions
├── models/
│   └── glm_wrapper.py        # MODIFY: Add minimal log prob support
├── data/                      # KEEP: Already works
└── utils/                     # KEEP: Already works

tests/
└── test_grpo_minimal.py      # NEW: Tests for minimal GRPO
```

---

## Step-by-Step Implementation

### Step 1: GRPO Loss Function (30 minutes)

**File:** `glm_training/grpo/loss.py`

```python
"""
Minimal GRPO loss implementation.

Following nanoGRPO and TRL patterns.
"""
import torch
import torch.nn.functional as F


def compute_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Compute z-score advantages within groups.
    
    This is the KEY innovation of GRPO - no value function needed!
    
    Args:
        rewards: [batch_size * group_size]
        group_size: Number of samples per prompt
        
    Returns:
        advantages: [batch_size * group_size]
    """
    # Reshape to groups
    batch_size = len(rewards) // group_size
    rewards = rewards.view(batch_size, group_size)
    
    # Z-score per group
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean) / (std + 1e-6)
    
    return advantages.view(-1)


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss (PPO-style clipped surrogate).
    
    Args:
        new_logprobs: Current policy log probs (with gradients)
        old_logprobs: Old policy log probs (from generation, no gradients)
        advantages: Normalized advantages
        clip_range: PPO clipping epsilon
        
    Returns:
        loss: Policy loss
        metrics: Dictionary of metrics for logging
    """
    # Compute ratio
    ratio = torch.exp(new_logprobs - old_logprobs.detach())
    
    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # Policy loss (negative because we maximize)
    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    loss = -torch.min(loss_unclipped, loss_clipped).mean()
    
    # Metrics
    metrics = {
        "loss": loss.item(),
        "ratio_mean": ratio.mean().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }
    
    return loss, metrics
```

**Test:**
```python
def test_grpo_loss():
    """Test GRPO loss computation."""
    # Create dummy data
    new_lp = torch.randn(8, requires_grad=True)
    old_lp = torch.randn(8)
    advantages = torch.randn(8)
    
    loss, metrics = compute_grpo_loss(new_lp, old_lp, advantages)
    
    # Check loss has gradients
    assert loss.requires_grad
    
    # Check can backprop
    loss.backward()
    assert new_lp.grad is not None
    
    print("✅ GRPO loss test passed")
```

### Step 2: Minimal GRPO Trainer (2 hours)

**File:** `glm_training/grpo/trainer.py`

```python
"""
Minimal GRPO trainer - starting from scratch.

Based on nanoGRPO + TRL patterns, but simplified.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from PIL import Image
import numpy as np

from .loss import compute_advantages, compute_grpo_loss


class MinimalGRPOTrainer:
    """
    Minimal GRPO trainer - just the essentials.
    
    Start simple, add features later.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        reward_function,
        group_size: int = 4,
        clip_range: float = 0.2,
        learning_rate: float = 5e-6,
    ):
        """
        Initialize minimal GRPO trainer.
        
        Args:
            model: GLM-Image model (or wrapper)
            tokenizer: Tokenizer
            train_dataloader: Training data
            reward_function: Function that takes (images, targets) -> rewards
            group_size: Samples per prompt
            clip_range: PPO clipping epsilon
            learning_rate: Learning rate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.reward_function = reward_function
        self.group_size = group_size
        self.clip_range = clip_range
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        self.device = next(model.parameters()).device
        
    def generate_samples(self, prompts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple samples per prompt.
        
        Returns:
            images: [batch_size * group_size, C, H, W]
            old_logprobs: [batch_size * group_size]
        """
        # Repeat prompts for group sampling
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.group_size)
        
        # Tokenize
        inputs = self.tokenizer(
            repeated_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate (no gradients)
        with torch.no_grad():
            # This is where we need GLM-Image to generate
            # For now, placeholder
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
            )
            
            # Compute old log probs
            old_logprobs = self._compute_logprobs(generated_ids)
            
            # Convert to images (GLM-Image specific)
            images = self._ids_to_images(generated_ids)
        
        return images, old_logprobs
    
    def _compute_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log probabilities.
        
        This is simplified - real version needs proper masking.
        """
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mean over sequence
        return token_log_probs.mean(dim=-1)
    
    def _ids_to_images(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert generated token IDs to images.
        
        This is GLM-Image specific - placeholder for now.
        """
        # TODO: Implement GLM-Image decoding
        # For now, return dummy images
        batch_size = generated_ids.size(0)
        return torch.randn(batch_size, 3, 1024, 1024, device=self.device)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        This is the core training loop.
        """
        prompts = batch["prompts"]
        target_images = batch["target_images"].to(self.device)
        
        # Step 1: Generate samples (no gradients)
        images, old_logprobs = self.generate_samples(prompts)
        
        # Step 2: Compute rewards
        # Repeat targets for each sample
        repeated_targets = target_images.repeat_interleave(
            self.group_size, dim=0
        )
        rewards = self.reward_function(images, repeated_targets)
        
        # Step 3: Compute advantages
        advantages = compute_advantages(rewards, self.group_size)
        
        # Step 4: Recompute log probs (with gradients)
        # TODO: Need token IDs from generation
        # For now, placeholder
        new_logprobs = torch.randn_like(old_logprobs, requires_grad=True)
        
        # Step 5: Compute loss
        loss, metrics = compute_grpo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            self.clip_range,
        )
        
        # Step 6: Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Add reward metrics
        metrics["reward_mean"] = rewards.mean().item()
        
        return metrics
    
    def train(self, num_steps: int = 1000):
        """
        Main training loop.
        
        Keep it simple for now.
        """
        self.model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            if step >= num_steps:
                break
            
            metrics = self.train_step(batch)
            
            if step % 10 == 0:
                print(f"Step {step}: loss={metrics['loss']:.4f}, "
                      f"reward={metrics['reward_mean']:.4f}")
```

### Step 3: Integration with GLM-Image (3 hours)

**File:** `glm_training/models/glm_wrapper.py` (modifications)

Add these methods to existing `GLMImageWrapper`:

```python
def compute_sequence_logprobs(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute sequence log probabilities from token IDs.
    
    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        
    Returns:
        log_probs: [batch_size]
    """
    # Forward through AR model
    outputs = self.ar_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs, -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply mask and normalize
    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous()
        token_log_probs = token_log_probs * mask
        sequence_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
    else:
        sequence_log_probs = token_log_probs.mean(dim=-1)
    
    return sequence_log_probs


def generate_with_tracking(
    self,
    prompts: List[str],
    **generation_kwargs,
) -> dict:
    """
    Generate images and track intermediate values for GRPO.
    
    Returns dict with:
        - images: Generated PIL images
        - token_ids: Generated token IDs
        - log_probs: Sequence log probabilities
    """
    # Tokenize
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"].to(self.ar_model.device)
    attention_mask = inputs["attention_mask"].to(self.ar_model.device)
    
    # Generate with AR model
    with torch.no_grad():
        generated_ids = self.ar_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        
        # Compute log probs for generated sequence
        log_probs = self.compute_sequence_logprobs(
            generated_ids,
            attention_mask=torch.ones_like(generated_ids),
        )
    
    # Decode to images with DiT
    images = self._decode_with_dit(generated_ids, prompts)
    
    return {
        "images": images,
        "token_ids": generated_ids,
        "log_probs": log_probs,
    }


def _decode_with_dit(
    self,
    token_ids: torch.Tensor,
    prompts: List[str],
) -> List[Image.Image]:
    """
    Decode token IDs to images using DiT.
    
    This is GLM-Image pipeline specific.
    """
    # Use the pipeline's decode method
    # This is simplified - real implementation depends on GLM-Image API
    images = []
    for ids, prompt in zip(token_ids, prompts):
        # Extract visual tokens (GLM-Image specific format)
        visual_tokens = ids  # Placeholder
        
        # Decode with DiT
        image = self.pipe(prompt, visual_tokens=visual_tokens)
        images.append(image)
    
    return images
```

### Step 4: Simple Reward Function (30 minutes)

**File:** `glm_training/grpo/utils.py`

```python
"""
Utility functions for GRPO training.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def simple_reward_function(
    images: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Simple reward: negative MSE loss.
    
    Start with something simple that works.
    Later can add LPIPS, aesthetic scores, etc.
    
    Args:
        images: [N, C, H, W]
        targets: [N, C, H, W]
        
    Returns:
        rewards: [N]
    """
    # MSE per image
    mse = F.mse_loss(images, targets, reduction='none')
    mse_per_image = mse.view(images.size(0), -1).mean(dim=1)
    
    # Convert to reward (negative loss)
    rewards = -mse_per_image
    
    return rewards


def pil_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Convert list of PIL images to tensor."""
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors)


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """Convert tensor to list of PIL images."""
    images = []
    for t in tensor:
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        images.append(img)
    return images
```

### Step 5: Minimal Training Script (30 minutes)

**File:** `train_grpo_minimal.py`

```python
"""
Minimal GRPO training script.

Start from scratch - keep it simple!
"""
import torch
from glm_training.grpo.trainer import MinimalGRPOTrainer
from glm_training.grpo.utils import simple_reward_function
from glm_training.models import GLMImageWrapper
from glm_training.data import T2IDataset
from torch.utils.data import DataLoader


def main():
    print("=== Minimal GRPO Training ===")
    
    # Load model
    print("Loading GLM-Image...")
    model = GLMImageWrapper(
        model_name="zai-org/GLM-Image",
        component="ar",  # Start with AR only
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = model.tokenizer
    
    # Load data
    print("Loading dataset...")
    dataset = T2IDataset(
        prompts_file="data/t2i/prompts.txt",
        target_images_dir="data/t2i/target_images",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create trainer
    print("Creating GRPO trainer...")
    trainer = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
        clip_range=0.2,
        learning_rate=5e-6,
    )
    
    # Train
    print("Starting training...")
    trainer.train(num_steps=100)
    
    print("Done!")


if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Test 1: GRPO Loss
```bash
python -c "from glm_training.grpo.loss import *; test_grpo_loss()"
```

### Test 2: Advantage Computation
```bash
python -c "
from glm_training.grpo.loss import compute_advantages
import torch

rewards = torch.tensor([1., 2., 3., 4., 5., 5., 5., 5.])
advantages = compute_advantages(rewards, group_size=4)

# Check group 1 has variance
assert advantages[:4].std() > 0
# Check group 2 is zero (constant)
assert torch.allclose(advantages[4:], torch.zeros(4), atol=1e-5)

print('✅ Advantages test passed')
"
```

### Test 3: End-to-End
```bash
python train_grpo_minimal.py
```

---

## Success Criteria for Phase 1

- [ ] GRPO loss computes correctly
- [ ] Advantages are z-score normalized per group
- [ ] Training loop runs without errors
- [ ] Loss decreases over steps
- [ ] Rewards increase over steps
- [ ] Code is clean and understandable

---

## What's Next (Phase 2)

Once Phase 1 works:

1. **Add Reference Model** - For proper KL divergence
2. **Add KL Penalty** - Keep close to reference policy
3. **Better Rewards** - LPIPS, aesthetic scores
4. **Memory Optimization** - Micro-batching, gradient accumulation
5. **Multi-GPU** - Distributed training
6. **Logging** - TensorBoard, W&B

But first: **Get Phase 1 working!**

---

## Key Differences from Old Implementation

| Old Approach | New Approach |
|--------------|--------------|
| Complex GRPO with many features | Minimal GRPO - just essentials |
| Tried to fix broken code | Start fresh with clean code |
| Reconstruction loss | Proper policy gradient |
| Single sample per prompt | Group-based sampling |
| No log probability tracking | Proper log prob computation |
| Mixed up components | Clear separation of concerns |

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | GRPO loss + tests | Working loss function |
| 2 | Minimal trainer skeleton | Trainer structure |
| 3 | GLM-Image integration | Generation with log probs |
| 4 | Training script | End-to-end training |
| 5 | Testing & debugging | Working minimal GRPO |

**Total: 1 week for minimal working GRPO**

---

## Files Created (Summary)

**New:**
```
glm_training/grpo/
  __init__.py
  loss.py          # GRPO loss functions
  trainer.py       # Minimal GRPO trainer
  utils.py         # Helper functions

train_grpo_minimal.py  # Training script

tests/
  test_grpo_minimal.py  # Tests
```

**Modified:**
```
glm_training/models/glm_wrapper.py  # Add log prob methods
```

**Total new lines:** ~500 (minimal!)

---

## Next Steps

1. ✅ Create `glm_training/grpo/` package
2. ✅ Implement GRPO loss
3. ✅ Test loss function
4. 🔄 Implement minimal trainer
5. 🔄 Integrate with GLM-Image
6. 🔄 Test end-to-end
7. 🔄 Add features incrementally

Let's start coding!
