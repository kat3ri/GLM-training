# GRPO Quick Reference Guide

This is a condensed reference for the comprehensive research in `GRPO_IMPLEMENTATIONS_RESEARCH.md`.

## What is GRPO?

**Group Relative Policy Optimization** - A reinforcement learning algorithm from DeepSeekMath (arxiv:2402.03300) that's essentially PPO with advantages computed as z-scores within groups instead of using a value function.

**Key Advantage:** Simpler than PPO (no critic/value network needed), more memory-efficient.

## Core Formula

```
Advantage_i = (reward_i - mean_reward_in_group) / (std_reward_in_group + 1e-6)

Loss = -min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
) + β * KL_divergence

where ratio = exp(new_log_prob - old_log_prob)
```

## Top 3 Implementations to Study

### 1. HuggingFace TRL (BEST for Production)
- **URL:** https://github.com/huggingface/trl
- **File:** `trl/trainer/grpo_trainer.py`
- **Why:** Production-ready, full-featured, well-maintained
- **Features:** Multi-GPU, vLLM, multiple rewards, async rewards

### 2. nanoGRPO (BEST for Learning)
- **URL:** https://github.com/joey00072/nanoGRPO
- **File:** `grpo.py` (single file!)
- **Why:** Clean, minimal, easy to understand
- **Features:** Works on 8GB GPU, LoRA support, micro-batching

### 3. Google Tunix (BEST for Research)
- **URL:** https://github.com/google/tunix
- **File:** `tunix/rl/grpo/grpo_learner.py`
- **Why:** Clean design, JAX-based, optimized
- **Features:** TPU support, modular architecture

## Key Implementation Steps

### 1. Reference Model
```python
# Create frozen copy for KL divergence
ref_model = copy.deepcopy(model)
for param in ref_model.parameters():
    param.requires_grad = False
```

### 2. Generate with Log Probs
```python
def get_per_token_logprobs(model, input_ids):
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_log_probs = torch.gather(
        log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs

# Generate multiple samples per prompt
with torch.no_grad():
    for _ in range(group_size):  # e.g., 4-8
        completion = model.generate(prompt)
        old_log_probs = get_per_token_logprobs(model, completion)
```

### 3. Compute Advantages (Z-Score)
```python
# rewards shape: [batch_size * group_size]
batch_size = len(rewards) // group_size
rewards = rewards.view(batch_size, group_size)

# Z-score within each group
mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean) / (std + 1e-6)

advantages = advantages.view(-1)  # Flatten back
```

### 4. Compute GRPO Loss
```python
def compute_grpo_loss(new_logprobs, old_logprobs, ref_logprobs, 
                      advantages, clip_range=0.2, kl_coef=0.05):
    # Policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # Clipped surrogate loss (PPO-style)
    clipped_ratio = torch.clamp(ratio, 1-clip_range, 1+clip_range)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    
    # KL divergence penalty
    kl_div = (new_logprobs - ref_logprobs).mean()
    kl_penalty = kl_coef * kl_div
    
    return policy_loss + kl_penalty
```

## Memory Optimization (from nanoGRPO)

### Micro-Batching
```python
# Split group into micro-batches
for i in range(0, group_size, micro_batch_size):
    micro_batch = group[i:i+micro_batch_size]
    loss = compute_loss(micro_batch)
    loss.backward()
    torch.cuda.empty_cache()

optimizer.step()
optimizer.zero_grad()
```

### Offloading
```python
# Offload to CPU when not in use
batch_inputs = batch_inputs.cpu()
rewards = rewards.cpu()
torch.cuda.empty_cache()

# Move back when needed
batch_inputs = batch_inputs.to(device)
```

## Configuration Template

```yaml
reward:
  enabled: true
  grpo:
    num_samples: 4           # Completions per prompt
    clip_range: 0.2          # PPO clip ε (typically 0.1-0.3)
    kl_coef: 0.05            # KL penalty coefficient
    normalize_advantages: true
    
  metrics:
    lpips: 0.4
    ssim: 0.2
    text_accuracy: 0.3
    aesthetic: 0.1
    
  micro_batch_size: 2        # For memory efficiency
```

## Typical Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| group_size | 4-8 | More = better estimation, more memory |
| clip_range (ε) | 0.1-0.3 | 0.2 is standard |
| kl_coef (β) | 0.01-0.1 | Higher = stay closer to reference |
| learning_rate | 1e-6 to 5e-6 | Lower than supervised learning |
| micro_batch_size | 1-4 | Depends on available memory |

## Common Pitfalls

### ❌ Don't
```python
# Global normalization (wrong!)
advantages = (rewards - rewards.mean()) / rewards.std()

# Tracking gradients through generation
completions = model.generate(prompts)
loss = compute_loss(completions)  # Gradients?

# No reference model
kl = new_log_probs - old_log_probs  # Wrong!
```

### ✅ Do
```python
# Group-wise normalization (correct!)
rewards = rewards.view(batch_size, group_size)
advantages = (rewards - rewards.mean(1, keepdim=True)) / \
             rewards.std(1, keepdim=True)

# Generate without gradients, recompute for training
with torch.no_grad():
    completions, old_lp = model.generate(prompts)
new_lp = compute_logprobs(model, completions)  # With gradients

# Use frozen reference model
kl = new_log_probs - ref_model_log_probs  # Correct!
```

## For GLM-Image Specifically

### Challenge: Vision + Diffusion Model
```python
# AR model generates visual tokens (trackable log probs)
ar_outputs = model.ar_model.generate(prompts, return_log_probs=True)

# DiT decoder converts to images (deterministic)
images = model.dit_model.decode(ar_outputs.tokens)

# Use AR log probs for policy gradient
# DiT is deterministic decoder, doesn't need gradients
return images, ar_outputs.log_probs
```

### Vision Rewards
```python
class VisualRewardCalculator:
    def compute_rewards(self, generated, target, prompts):
        return (
            0.4 * self.lpips_similarity(generated, target) +
            0.2 * self.ssim_score(generated, target) +
            0.3 * self.ocr_text_accuracy(generated, prompts) +
            0.1 * self.aesthetic_score(generated)
        )
```

## Testing Checklist

- [ ] Advantage computation gives zero mean per group
- [ ] PPO clipping activates when ratio > 1.2 or < 0.8
- [ ] KL divergence is positive and bounded
- [ ] Memory usage stays under GPU limit
- [ ] Training loss decreases over time
- [ ] Reward increases over time

## Quick Start (Pseudocode)

```python
# 1. Setup
model = load_model()
ref_model = freeze(copy(model))
optimizer = AdamW(model.parameters(), lr=5e-6)

# 2. Training loop
for batch in dataloader:
    prompts = batch["prompts"]
    targets = batch["targets"]
    
    # Generate multiple completions per prompt
    with torch.no_grad():
        completions, old_lp = generate_group(model, prompts, n=4)
    
    # Compute rewards
    rewards = reward_fn(completions, targets)
    
    # Compute advantages (z-score per group)
    advantages = compute_advantages(rewards, group_size=4)
    
    # Compute GRPO loss
    new_lp = get_logprobs(model, completions)
    ref_lp = get_logprobs(ref_model, completions)
    loss = grpo_loss(new_lp, old_lp, ref_lp, advantages)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Resources

- **Full Research:** `GRPO_IMPLEMENTATIONS_RESEARCH.md` (27KB detailed analysis)
- **HF TRL Docs:** https://huggingface.co/docs/trl/grpo_trainer
- **Original Paper:** DeepSeekMath (arxiv:2402.03300)
- **Latest Paper:** DeepSeek R1 (arxiv:2501.12948)

## Next Steps for GLM-Training

1. Study HuggingFace TRL implementation
2. Study nanoGRPO for core concepts
3. Add reference model creation
4. Implement log probability tracking
5. Add advantage computation
6. Replace current loss with GRPO loss
7. Test on small model first
8. Add memory optimizations
9. Adapt for vision/diffusion architecture

---

**Quick Reference Version 1.0**  
**Based on research:** January 19, 2026  
**For details:** See `GRPO_IMPLEMENTATIONS_RESEARCH.md`
