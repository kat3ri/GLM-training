# HuggingFace TRL GRPO Adoption Plan for GLM-Training

**Date:** January 19, 2026  
**Status:** Implementation Planning  
**Primary Reference:** HuggingFace TRL GRPOTrainer  
**Target:** GLM-Image Training with proper GRPO

---

## Executive Summary

This document outlines the plan to adopt HuggingFace TRL's GRPO implementation patterns into GLM-Training. TRL is the gold standard for GRPO implementation, with production-ready code, excellent documentation, and active maintenance.

### Why HuggingFace TRL?

✅ **Production-Ready:** Used by thousands of projects  
✅ **Well-Documented:** Comprehensive docs and examples  
✅ **Actively Maintained:** Regular updates and bug fixes  
✅ **Feature-Complete:** Multi-GPU, vLLM, async rewards  
✅ **Battle-Tested:** Proven in real-world applications  
✅ **Best Practices:** Follows PyTorch and HF conventions  

---

## Current State Analysis

### What We Have (Working)
```python
# glm_training/trainers/reward_trainer.py
class RewardTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.reward_calculator = RewardCalculator(...)
        self.num_samples = config["reward"]["grpo"]["num_samples"]  # ✅
        self.kl_coef = config["reward"]["grpo"]["kl_coef"]          # ✅
        self.clip_range = config["reward"]["grpo"]["clip_range"]    # ✅
```

**Strengths:**
- ✅ Basic GRPO config parameters defined
- ✅ Reward calculator system in place
- ✅ Multi-GPU DDP support
- ✅ Separate AR/DiT training
- ✅ T2I and I2I modes

### What's Missing (Critical Gaps)

**From REVERT_SUMMARY.md:**
> "The complex GRPO implementation with `_generate_latents_with_log_probs()` method" was removed due to errors. Current version uses "simple reconstruction loss approach instead of policy gradient."

**Specific Gaps:**
1. ❌ No reference model (frozen copy)
2. ❌ No log probability tracking during generation
3. ❌ No group-based advantage computation
4. ❌ No PPO-style clipped loss
5. ❌ No proper KL divergence with reference model
6. ❌ Single sample generation (not group-based)

**Current train_step() approach:**
```python
def train_step(self, batch):
    # Generate samples (but only 1 per prompt, not a group)
    samples = self._generate_samples(prompts, source_images)
    
    # Compute rewards
    rewards = self.reward_calculator.compute_grpo_rewards(...)
    
    # PROBLEM: Uses reconstruction loss, not policy gradient!
    best_sample_idx = rewards.mean(dim=1).argmax()
    best_sample = samples[best_sample_idx]
    recon_loss = F.mse_loss(best_sample, target_images)
    reward_loss = -rewards[best_sample_idx].mean()
    loss = recon_loss + reward_loss  # ❌ Not GRPO!
```

---

## TRL GRPO Architecture Overview

### TRL's GRPOTrainer Structure

```python
# From: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

class GRPOTrainer(BaseTrainer):
    """
    Key components:
    1. Model (policy) - what we're training
    2. Ref Model (frozen) - for KL divergence
    3. Reward Functions - compute rewards
    4. GRPO Config - hyperparameters
    """
    
    def __init__(self, model, reward_funcs, args, ...):
        self.model = model
        self.ref_model = self._create_reference_model(model)  # ✅ Frozen copy
        self.reward_funcs = self._prepare_reward_functions(reward_funcs)
        
    def training_step(self, batch):
        """Main training loop - this is what we need to adopt."""
        # 1. Generate completions (group-based, no gradients)
        with torch.no_grad():
            completions, old_log_probs = self.generate_completions(
                batch, num_samples=self.args.num_samples
            )
        
        # 2. Compute rewards for all completions
        rewards = self.compute_rewards(completions, batch)
        
        # 3. Compute advantages (z-score within groups)
        advantages = self.compute_advantages(rewards)
        
        # 4. Compute policy loss (with gradients)
        new_log_probs = self.get_log_probs(self.model, completions)
        ref_log_probs = self.get_log_probs(self.ref_model, completions)
        loss = self.compute_grpo_loss(
            new_log_probs, old_log_probs, ref_log_probs, advantages
        )
        
        # 5. Backward and optimize
        self.accelerator.backward(loss)
        self.optimizer.step()
```

### Key TRL Patterns to Adopt

#### 1. Reference Model Creation
```python
def _create_reference_model(self, model):
    """Create frozen copy for KL divergence computation."""
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model
```

#### 2. Log Probability Computation
```python
def get_log_probs(self, model, input_ids, attention_mask=None):
    """
    Compute per-token log probabilities.
    
    This is THE critical missing piece in current implementation.
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift logits and input_ids for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs, -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Sum over sequence (or mean, depending on normalization)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous()
        token_log_probs = token_log_probs * mask
        sequence_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1)
    else:
        sequence_log_probs = token_log_probs.mean(dim=-1)
    
    return sequence_log_probs
```

#### 3. Group-Based Generation
```python
def generate_completions(self, prompts, num_samples=4):
    """
    Generate multiple completions per prompt (group-based sampling).
    
    Key: Each prompt generates N samples for relative comparison.
    """
    # Repeat each prompt num_samples times
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * num_samples)
    
    # Tokenize
    inputs = self.tokenizer(repeated_prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(self.device)
    
    # Generate
    with torch.no_grad():
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            do_sample=True,
        )
        
        # Compute old log probs (needed for PPO ratio)
        old_log_probs = self.get_log_probs(self.model, outputs)
    
    return outputs, old_log_probs
```

#### 4. Advantage Computation
```python
def compute_advantages(self, rewards, num_samples):
    """
    Compute advantages using group-relative normalization (z-score).
    
    This is the KEY INNOVATION of GRPO over PPO.
    """
    # Reshape: [batch_size * num_samples] -> [batch_size, num_samples]
    batch_size = rewards.size(0) // num_samples
    rewards = rewards.view(batch_size, num_samples)
    
    # Compute mean and std per group (per prompt)
    mean_reward = rewards.mean(dim=1, keepdim=True)
    std_reward = rewards.std(dim=1, keepdim=True)
    
    # Z-score normalization
    advantages = (rewards - mean_reward) / (std_reward + 1e-6)
    
    # Flatten back
    return advantages.view(-1)
```

#### 5. GRPO Loss Function
```python
def compute_grpo_loss(
    self,
    new_log_probs,
    old_log_probs,
    ref_log_probs,
    advantages,
    clip_range=0.2,
    kl_coef=0.05,
):
    """
    Compute GRPO loss: PPO clipped surrogate + KL penalty.
    
    Args:
        new_log_probs: Current policy log probs (with gradients)
        old_log_probs: Old policy log probs (from generation, no gradients)
        ref_log_probs: Reference model log probs (no gradients)
        advantages: Normalized advantages per sample
        clip_range: PPO clipping epsilon
        kl_coef: KL divergence penalty coefficient
    """
    # Compute probability ratio
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # PPO clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # Policy loss (maximize advantages)
    policy_loss_unclipped = ratio * advantages
    policy_loss_clipped = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
    
    # KL divergence penalty (keep close to reference model)
    kl_div = (new_log_probs - ref_log_probs).mean()
    kl_penalty = kl_coef * kl_div
    
    # Total loss
    total_loss = policy_loss + kl_penalty
    
    # Metrics for logging
    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/kl": kl_penalty.item(),
        "loss/total": total_loss.item(),
        "ppo/ratio_mean": ratio.mean().item(),
        "ppo/ratio_max": ratio.max().item(),
        "ppo/ratio_min": ratio.min().item(),
        "ppo/clipped_ratio_mean": clipped_ratio.mean().item(),
        "advantages/mean": advantages.mean().item(),
        "advantages/std": advantages.std().item(),
    }
    
    return total_loss, metrics
```

---

## Implementation Plan

### Phase 1: Core GRPO Components (Week 1-2)

#### Task 1.1: Reference Model Creation
**File:** `glm_training/trainers/reward_trainer.py`

```python
class RewardTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Build main model
        self.model = self._build_model()
        
        # NEW: Create reference model (frozen copy)
        self.ref_model = self._create_reference_model()
        
        # ... rest of init
    
    def _create_reference_model(self) -> GLMImageWrapper:
        """
        Create frozen reference model for KL divergence.
        
        Following TRL pattern: deep copy and freeze all parameters.
        """
        import copy
        
        logger.info("Creating reference model for GRPO...")
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        
        # Freeze all parameters
        for param in ref_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Reference model created with {sum(p.numel() for p in ref_model.parameters())} parameters (frozen)")
        
        return ref_model
```

**Testing:**
```python
def test_reference_model_creation():
    """Test that reference model is properly frozen."""
    trainer = RewardTrainer(config)
    
    # Check reference model exists
    assert trainer.ref_model is not None
    
    # Check all params are frozen
    for param in trainer.ref_model.parameters():
        assert param.requires_grad == False
    
    # Check it's a separate copy
    assert id(trainer.model) != id(trainer.ref_model)
```

#### Task 1.2: Log Probability Tracking
**File:** `glm_training/models/glm_wrapper.py`

Add method to compute log probabilities:

```python
class GLMImageWrapper(nn.Module):
    # ... existing code ...
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities.
        
        This is needed for GRPO's policy gradient computation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            sequence_log_probs: Log probability per sequence [batch_size]
        """
        # Forward pass through AR model (only AR has log probs)
        if self.component in ["ar", "both"]:
            outputs = self.ar_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        else:
            raise ValueError("Log probs only available for AR component")
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
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
    
    def generate_with_log_probs(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        **generation_kwargs,
    ) -> tuple[List[Image.Image], torch.Tensor]:
        """
        Generate images and return log probabilities.
        
        Returns:
            generated_images: List of generated PIL images
            log_probs: Log probabilities for generated sequences
        """
        # Tokenize prompts
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
        
        # Compute log probs for generated sequences
        log_probs = self.compute_log_probs(generated_ids, attention_mask)
        
        # Decode with DiT (if using both components)
        if self.component in ["dit", "both"]:
            # Extract visual tokens from generated_ids
            # (This is GLM-Image specific - need to understand format)
            visual_tokens = self._extract_visual_tokens(generated_ids)
            
            # Generate images with DiT
            images = self._dit_decode(visual_tokens, prompts)
        else:
            # AR-only mode: convert tokens directly to images
            images = self._ar_to_images(generated_ids)
        
        return images, log_probs
```

**Testing:**
```python
def test_log_prob_computation():
    """Test log probability computation."""
    model = GLMImageWrapper("zai-org/GLM-Image", component="ar")
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 50))
    attention_mask = torch.ones_like(input_ids)
    
    # Compute log probs
    log_probs = model.compute_log_probs(input_ids, attention_mask)
    
    # Check shape
    assert log_probs.shape == (2,)
    
    # Check values are negative (log probs)
    assert (log_probs <= 0).all()
```

#### Task 1.3: Group-Based Generation
**File:** `glm_training/trainers/reward_trainer.py`

Update `_generate_samples` to generate groups:

```python
def _generate_samples_with_logprobs(
    self,
    prompts: List[str],
    source_images: Optional[List[Image.Image]] = None,
) -> tuple[List[torch.Tensor], torch.Tensor]:
    """
    Generate multiple samples per prompt for GRPO.
    
    Returns:
        samples: Generated images [batch_size * num_samples, C, H, W]
        old_log_probs: Log probabilities [batch_size * num_samples]
    """
    # Repeat each prompt num_samples times
    repeated_prompts = []
    repeated_source_images = []
    
    for i, prompt in enumerate(prompts):
        repeated_prompts.extend([prompt] * self.num_samples)
        if source_images is not None:
            repeated_source_images.extend([source_images[i]] * self.num_samples)
    
    # Generate images with log probs
    with torch.no_grad():
        if source_images is not None:
            generated_images, old_log_probs = self.model.generate_with_log_probs(
                prompts=repeated_prompts,
                images=repeated_source_images,
                height=self.config["data"]["image_size"]["height"],
                width=self.config["data"]["image_size"]["width"],
                num_inference_steps=self.config["model"]["dit"]["num_inference_steps"],
                guidance_scale=self.config["model"]["dit"]["guidance_scale"],
            )
        else:
            generated_images, old_log_probs = self.model.generate_with_log_probs(
                prompts=repeated_prompts,
                height=self.config["data"]["image_size"]["height"],
                width=self.config["data"]["image_size"]["width"],
                num_inference_steps=self.config["model"]["dit"]["num_inference_steps"],
                guidance_scale=self.config["model"]["dit"]["guidance_scale"],
            )
    
    # Convert PIL images to tensors
    image_tensors = []
    for img in generated_images:
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        image_tensors.append(img_tensor)
    
    samples = torch.stack(image_tensors).to(self.device)
    
    return samples, old_log_probs
```

#### Task 1.4: Advantage Computation
**File:** `glm_training/trainers/reward_trainer.py`

```python
def _compute_advantages(
    self,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Compute advantages using group-relative normalization (z-score).
    
    Following TRL pattern: normalize within each group (per prompt).
    
    Args:
        rewards: Tensor of shape [batch_size * num_samples]
        
    Returns:
        advantages: Z-score normalized advantages [batch_size * num_samples]
    """
    # Reshape to [batch_size, num_samples]
    batch_size = rewards.size(0) // self.num_samples
    rewards = rewards.view(batch_size, self.num_samples)
    
    # Compute statistics per group
    mean_reward = rewards.mean(dim=1, keepdim=True)
    std_reward = rewards.std(dim=1, keepdim=True)
    
    # Z-score normalization
    # Add small epsilon to prevent division by zero
    advantages = (rewards - mean_reward) / (std_reward + 1e-6)
    
    # Flatten back to [batch_size * num_samples]
    advantages = advantages.view(-1)
    
    return advantages
```

**Testing:**
```python
def test_advantage_computation():
    """Test advantage computation."""
    trainer = RewardTrainer(config)
    
    # Create test rewards (2 groups of 4 samples)
    rewards = torch.tensor([
        1.0, 2.0, 3.0, 4.0,  # Group 1: mean=2.5, std~1.29
        5.0, 5.0, 5.0, 5.0,  # Group 2: mean=5.0, std=0
    ])
    
    advantages = trainer._compute_advantages(rewards)
    
    # Group 1 should have ~zero mean, unit variance
    group1_adv = advantages[:4]
    assert torch.allclose(group1_adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(group1_adv.std(), torch.tensor(1.0), atol=1e-1)
    
    # Group 2 should be zero (constant rewards)
    group2_adv = advantages[4:]
    assert torch.allclose(group2_adv, torch.zeros(4), atol=1e-6)
```

#### Task 1.5: GRPO Loss Function
**File:** `glm_training/trainers/reward_trainer.py`

```python
def _compute_grpo_loss(
    self,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute GRPO loss following TRL pattern.
    
    Args:
        new_log_probs: Current policy log probs (with gradients)
        old_log_probs: Old policy log probs (no gradients)
        ref_log_probs: Reference model log probs (no gradients)
        advantages: Normalized advantages
        
    Returns:
        loss: Total GRPO loss
        metrics: Dictionary of metrics for logging
    """
    # Compute probability ratio
    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    
    # PPO clipped surrogate objective
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - self.clip_range,
        1.0 + self.clip_range,
    )
    
    # Policy loss (maximize advantages weighted by ratio)
    policy_loss_unclipped = ratio * advantages
    policy_loss_clipped = clipped_ratio * advantages
    policy_loss = -torch.min(
        policy_loss_unclipped,
        policy_loss_clipped,
    ).mean()
    
    # KL divergence penalty
    kl_div = (new_log_probs - ref_log_probs.detach()).mean()
    kl_penalty = self.kl_coef * kl_div
    
    # Total loss
    total_loss = policy_loss + kl_penalty
    
    # Compile metrics
    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/kl": kl_penalty.item(),
        "loss/total": total_loss.item(),
        "ppo/ratio_mean": ratio.mean().item(),
        "ppo/ratio_max": ratio.max().item(),
        "ppo/ratio_min": ratio.min().item(),
        "ppo/clipped_fraction": ((ratio < 1 - self.clip_range) | (ratio > 1 + self.clip_range)).float().mean().item(),
        "advantages/mean": advantages.mean().item(),
        "advantages/std": advantages.std().item(),
    }
    
    return total_loss, metrics
```

#### Task 1.6: Update train_step()
**File:** `glm_training/trainers/reward_trainer.py`

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Single training step with GRPO following TRL pattern.
    
    Args:
        batch: Training batch
        
    Returns:
        Dictionary of metrics
    """
    prompts = batch["prompts"]
    target_images = batch["target_images"].to(self.device)
    
    # Get source images for i2i mode
    source_images = None
    if self.mode == "i2i" and "source_images" in batch:
        source_images_tensor = batch["source_images"].to(self.device)
        # Convert to PIL images for generation
        source_images = []
        for img_tensor in source_images_tensor:
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            source_images.append(Image.fromarray(img_np))
    
    # Step 1: Generate multiple samples per prompt (no gradients)
    samples, old_log_probs = self._generate_samples_with_logprobs(
        prompts, source_images
    )
    
    # Step 2: Compute rewards for all samples
    # Repeat target images for each sample
    repeated_targets = target_images.repeat_interleave(self.num_samples, dim=0)
    
    rewards = self.reward_calculator.compute_grpo_rewards(
        samples=samples,
        target_images=repeated_targets,
        prompts=prompts * self.num_samples,  # Repeat prompts
        source_images=source_images_tensor.repeat_interleave(self.num_samples, dim=0) if source_images else None,
    )
    
    # Step 3: Compute advantages (z-score within groups)
    advantages = self._compute_advantages(rewards)
    
    # Step 4: Compute new log probs (with gradients)
    # For GLM-Image: need to recompute AR model log probs
    # Note: samples are images, need to convert back to token IDs
    # This is a challenge specific to vision models...
    
    # CHALLENGE: We generated images, but need token-level log probs
    # Solution: Store token IDs during generation, recompute log probs
    # This requires modifying generate_with_log_probs to return token_ids
    
    # For now, let's assume we can get log probs from images
    # (This is the AR model's job - it generated these images)
    new_log_probs = self._compute_logprobs_from_samples(samples)
    
    # Step 5: Get reference model log probs (no gradients)
    with torch.no_grad():
        ref_log_probs = self._compute_logprobs_from_samples(
            samples, use_ref_model=True
        )
    
    # Step 6: Compute GRPO loss
    loss, metrics = self._compute_grpo_loss(
        new_log_probs,
        old_log_probs,
        ref_log_probs,
        advantages,
    )
    
    # Step 7: Backward pass
    self.optimizer.zero_grad()
    
    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config["training"]["max_grad_norm"]
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config["training"]["max_grad_norm"]
        )
        self.optimizer.step()
    
    # Add reward metrics
    metrics["reward/mean"] = rewards.mean().item()
    metrics["reward/std"] = rewards.std().item()
    metrics["reward/max"] = rewards.max().item()
    metrics["reward/min"] = rewards.min().item()
    
    return metrics
```

---

### Phase 2: Testing & Validation (Week 3)

#### Unit Tests
**File:** `tests/test_grpo_components.py`

```python
import pytest
import torch
from glm_training.trainers.reward_trainer import RewardTrainer

def test_reference_model_frozen():
    """Test reference model is properly frozen."""
    pass  # Implemented above

def test_log_prob_computation():
    """Test log probability computation."""
    pass  # Implemented above

def test_advantage_computation():
    """Test advantage z-score normalization."""
    pass  # Implemented above

def test_grpo_loss_computation():
    """Test GRPO loss calculation."""
    trainer = RewardTrainer(config)
    
    # Create dummy data
    new_lp = torch.randn(8)
    old_lp = torch.randn(8)
    ref_lp = torch.randn(8)
    advantages = torch.randn(8)
    
    loss, metrics = trainer._compute_grpo_loss(
        new_lp, old_lp, ref_lp, advantages
    )
    
    assert loss.requires_grad
    assert "loss/policy" in metrics
    assert "loss/kl" in metrics
    assert "ppo/ratio_mean" in metrics

def test_ppo_clipping_activates():
    """Test PPO clipping works when ratio is large."""
    trainer = RewardTrainer(config)
    
    # Create scenario where ratio >> 1.2
    new_lp = torch.tensor([2.0])  # ratio = exp(2.0) = 7.4
    old_lp = torch.tensor([0.0])
    ref_lp = torch.tensor([0.0])
    advantages = torch.tensor([1.0])
    
    loss, metrics = trainer._compute_grpo_loss(
        new_lp, old_lp, ref_lp, advantages
    )
    
    # Should be clipped
    assert metrics["ppo/ratio_mean"] > 1.2
    assert metrics["ppo/clipped_fraction"] > 0
```

#### Integration Tests
**File:** `tests/test_grpo_integration.py`

```python
def test_full_training_pipeline():
    """Test complete GRPO training pipeline."""
    config = load_config("configs/t2i_training.yaml")
    trainer = RewardTrainer(config)
    
    batch = {
        "prompts": ["test prompt 1", "test prompt 2"],
        "target_images": torch.randn(2, 3, 1024, 1024),
    }
    
    # Run one training step
    metrics = trainer.train_step(batch)
    
    # Check all expected metrics present
    assert "loss/total" in metrics
    assert "loss/policy" in metrics
    assert "loss/kl" in metrics
    assert "reward/mean" in metrics
    assert "ppo/ratio_mean" in metrics

def test_memory_usage():
    """Test training fits in GPU memory."""
    torch.cuda.reset_peak_memory_stats()
    
    trainer = RewardTrainer(config)
    for _ in range(10):
        metrics = trainer.train_step(batch)
        torch.cuda.empty_cache()
    
    peak = torch.cuda.max_memory_allocated()
    assert peak < 80 * 1024**3  # 80GB
```

---

### Phase 3: Memory Optimization (Week 4)

#### Micro-Batching
From nanoGRPO pattern:

```python
def _train_step_with_micro_batching(self, batch):
    """
    Train with micro-batching for memory efficiency.
    
    Following nanoGRPO pattern: split groups into micro-batches.
    """
    # Generate all samples
    samples, old_log_probs = self._generate_samples_with_logprobs(...)
    rewards = self.reward_calculator.compute_grpo_rewards(...)
    advantages = self._compute_advantages(rewards)
    
    # Reshape into groups
    batch_size = len(batch["prompts"])
    samples = samples.view(batch_size, self.num_samples, *samples.shape[1:])
    old_log_probs = old_log_probs.view(batch_size, self.num_samples)
    advantages = advantages.view(batch_size, self.num_samples)
    
    # Offload to CPU to save VRAM
    samples = samples.cpu()
    old_log_probs = old_log_probs.cpu()
    advantages = advantages.cpu()
    torch.cuda.empty_cache()
    
    # Process each group with micro-batching
    total_loss = 0
    for i in range(batch_size):
        group_samples = samples[i]  # [num_samples, C, H, W]
        group_old_lp = old_log_probs[i]  # [num_samples]
        group_adv = advantages[i]  # [num_samples]
        
        # Split into micro-batches
        micro_batch_size = self.config["reward"]["micro_batch_size"]
        num_micro_batches = (self.num_samples + micro_batch_size - 1) // micro_batch_size
        
        for j in range(num_micro_batches):
            start = j * micro_batch_size
            end = min(start + micro_batch_size, self.num_samples)
            
            # Move micro-batch to GPU
            micro_samples = group_samples[start:end].to(self.device)
            micro_old_lp = group_old_lp[start:end].to(self.device)
            micro_adv = group_adv[start:end].to(self.device)
            
            # Compute loss
            micro_new_lp = self._compute_logprobs_from_samples(micro_samples)
            with torch.no_grad():
                micro_ref_lp = self._compute_logprobs_from_samples(
                    micro_samples, use_ref_model=True
                )
            
            loss, _ = self._compute_grpo_loss(
                micro_new_lp, micro_old_lp, micro_ref_lp, micro_adv
            )
            
            # Backward (accumulate gradients)
            loss.backward()
            total_loss += loss.item()
            
            # Clear GPU memory
            del micro_samples, micro_old_lp, micro_adv, micro_new_lp, micro_ref_lp
            torch.cuda.empty_cache()
        
        # Update weights after processing full group
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    return total_loss / batch_size
```

---

### Phase 4: Documentation & Examples (Week 5)

#### Documentation Updates
1. **GRPO_IMPLEMENTATION.md** - Technical details
2. **TRAINING_GUIDE.md** - User guide with examples
3. **MIGRATION_GUIDE.md** - From old to new implementation
4. **API_REFERENCE.md** - Updated API docs

#### Example Scripts
**File:** `examples/grpo_training_t2i.py`

```python
"""
Example: GRPO training for text-to-image generation.

This example shows how to train GLM-Image using GRPO with HuggingFace TRL patterns.
"""
from glm_training.trainers import RewardTrainer
from glm_training.data import T2IDataset
import yaml

# Load configuration
with open("configs/t2i_training.yaml") as f:
    config = yaml.safe_load(f)

# Create trainer (follows TRL pattern)
trainer = RewardTrainer(config)

# Train
trainer.train()
```

---

## Configuration Updates

### Updated YAML Schema
**File:** `configs/t2i_training.yaml`

```yaml
# GRPO Configuration (following TRL patterns)
reward:
  enabled: true
  
  # GRPO hyperparameters (from TRL)
  grpo:
    num_samples: 4              # Samples per prompt (group size)
    clip_range: 0.2             # PPO clipping epsilon
    kl_coef: 0.05               # KL divergence penalty
    
    # TRL-specific options
    normalize_advantages: true  # Apply z-score normalization
    advantage_epsilon: 1e-6     # Epsilon for advantage computation
    
    # Memory optimization
    micro_batch_size: 2         # Process N samples at a time
    offload_ref_model: false    # Offload reference model to CPU
  
  # Reward metrics
  metrics:
    lpips: 0.4
    ssim: 0.2
    text_accuracy: 0.3
    aesthetic: 0.1
  
  # Logging
  log_advantages: true          # Log advantage statistics
  log_ratios: true              # Log PPO ratios
  log_kl: true                  # Log KL divergence
```

---

## Migration Path

### From Current to TRL-Based Implementation

**Step 1:** Add reference model (non-breaking change)
```python
# Old: No reference model
self.model = self._build_model()

# New: Add reference model
self.model = self._build_model()
self.ref_model = self._create_reference_model()  # NEW
```

**Step 2:** Update generation (non-breaking)
```python
# Old: Generate without log probs
samples = self._generate_samples(prompts)

# New: Generate with log probs
samples, old_log_probs = self._generate_samples_with_logprobs(prompts)  # NEW
```

**Step 3:** Add advantage computation (non-breaking)
```python
# Old: Use best sample
best_idx = rewards.argmax()

# New: Compute advantages
advantages = self._compute_advantages(rewards)  # NEW
```

**Step 4:** Replace loss (BREAKING)
```python
# Old: Reconstruction loss
recon_loss = F.mse_loss(best_sample, target)
loss = recon_loss - rewards.mean()

# New: GRPO loss
new_lp = self._compute_logprobs(samples)
ref_lp = self._compute_logprobs_ref(samples)
loss, metrics = self._compute_grpo_loss(new_lp, old_lp, ref_lp, advantages)  # NEW
```

---

## Success Criteria

### Phase 1 Success Metrics
- [ ] Reference model created and frozen
- [ ] Log probabilities computed correctly
- [ ] Group-based generation working
- [ ] Advantages have zero mean per group
- [ ] GRPO loss computes without errors
- [ ] All unit tests passing

### Phase 2 Success Metrics
- [ ] Integration tests passing
- [ ] Training runs without OOM errors
- [ ] Rewards increase over training
- [ ] PPO clipping activates appropriately
- [ ] KL divergence stays bounded

### Phase 3 Success Metrics
- [ ] Memory usage < 80GB on A100
- [ ] Micro-batching works correctly
- [ ] Training speed acceptable
- [ ] Can train on 40GB GPU (AR only)

### Phase 4 Success Metrics
- [ ] Documentation complete
- [ ] Examples working
- [ ] Migration guide tested
- [ ] Performance benchmarks documented

---

## Timeline

| Week | Phase | Tasks | Deliverable |
|------|-------|-------|-------------|
| 1-2  | Phase 1 | Core GRPO components | Working GRPO implementation |
| 3    | Phase 2 | Testing & validation | Tested, verified implementation |
| 4    | Phase 3 | Memory optimization | Production-ready performance |
| 5    | Phase 4 | Documentation | Complete documentation & examples |

---

## Risks and Mitigations

### Risk 1: GLM-Image Specific Challenges
**Problem:** GLM-Image is vision model, TRL is for text models
**Mitigation:** 
- Use AR model log probs (text-like)
- Treat DiT as deterministic decoder (no gradients)
- Store token IDs during generation for recomputation

### Risk 2: Memory Constraints
**Problem:** 16B parameters + multiple samples = OOM
**Mitigation:**
- Micro-batching (from nanoGRPO)
- Gradient accumulation
- Reference model offloading
- Component-specific training (AR or DiT only)

### Risk 3: Log Probability Computation
**Problem:** Need to recompute log probs from generated images
**Mitigation:**
- Store token IDs during generation
- Recompute AR model forward pass
- Cache intermediate representations

---

## Open Questions

1. **How to handle DiT decoder in policy gradient?**
   - Option A: Only use AR log probs (treat DiT as deterministic)
   - Option B: Develop diffusion-specific policy gradient
   - **Recommended:** Option A (simpler, follows vision+language model patterns)

2. **What's the best way to store/retrieve token IDs?**
   - Option A: Modify generate() to return token IDs
   - Option B: Store in separate cache
   - **Recommended:** Option A (cleaner API)

3. **Should we support vLLM integration like TRL?**
   - Benefit: Faster generation (2-3x speedup)
   - Cost: Additional complexity
   - **Recommended:** Phase 3 or later (after core GRPO works)

---

## References

### Primary
- **HuggingFace TRL:** https://github.com/huggingface/trl
- **TRL GRPOTrainer:** https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
- **TRL Docs:** https://huggingface.co/docs/trl/grpo_trainer

### Supporting
- **nanoGRPO:** https://github.com/joey00072/nanoGRPO (memory optimization)
- **DeepSeekMath Paper:** https://arxiv.org/abs/2402.03300 (original GRPO)
- **DeepSeek R1 Paper:** https://arxiv.org/abs/2501.12948 (recent application)

---

## Next Steps

1. **Review this plan** with team
2. **Set up development branch** for TRL adoption
3. **Begin Phase 1, Task 1.1** (reference model)
4. **Incremental implementation** following the plan
5. **Test after each task** before moving forward

---

**Status:** Ready for implementation  
**Estimated Timeline:** 5 weeks  
**Confidence:** High (following proven TRL patterns)  
**Risk Level:** Medium (vision model adaptations needed)
