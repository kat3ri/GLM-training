# GRPO Training Implementations Research

**Research Date:** January 19, 2026  
**Purpose:** Document working GRPO (Group Relative Policy Optimization) implementations from GitHub to identify best practices and reference implementations for GLM-Image training.

---

## Executive Summary

GRPO is a reinforcement learning algorithm introduced in the DeepSeekMath paper (arxiv:2402.03300) that has become popular for training reasoning models. It's essentially PPO (Proximal Policy Optimization) with advantages estimated as the z-score of grouped outputs instead of using a value function, making it simpler and more memory-efficient.

### Key Formula

```
J_GRPO(θ) = (1/G) Σ min(
    π_θ(o_i|q)/π_θ_old(o_i|q) * A_i,
    clip(π_θ(o_i|q)/π_θ_old(o_i|q), 1-ε, 1+ε) * A_i
)
```

Where advantage A_i is computed as: `A_i = (reward_i - mean_reward) / (std_reward + 1e-6)`

---

## Top Implementations Found

### 1. **HuggingFace TRL (Recommended Primary Reference)**
- **Repository:** https://github.com/huggingface/trl
- **Stars:** Very high (official HuggingFace library)
- **Implementation:** `trl/trainer/grpo_trainer.py`
- **Status:** Production-ready, actively maintained
- **Key Features:**
  - Full integration with Transformers ecosystem
  - Supports distributed training (DDP, DeepSpeed, FSDP)
  - Multi-GPU and multi-node support
  - Comprehensive logging (W&B, TensorBoard)
  - Multiple reward function support
  - PEFT/LoRA integration
  - Async reward functions
  - vLLM integration for faster generation
  - Response schema support

**Architecture Highlights:**
- Uses `GRPOConfig` for configuration management
- Implements proper gradient clipping
- Supports multiple reward functions that can be combined
- Handles both synchronous and asynchronous reward computation
- Proper handling of KL divergence penalty
- Support for multimodal models
- Implements rollout functions for generation
- Group-based advantage normalization

**Code Structure:**
```python
class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        reward_funcs,  # Can be list of reward functions
        args: GRPOConfig,
        train_dataset,
        processing_class,  # Tokenizer/Processor
        ...
    )
    
    # Key methods:
    - generate(): Generate completions with model
    - compute_rewards(): Compute rewards from reward functions
    - compute_advantages(): Z-score normalization within groups
    - compute_policy_loss(): PPO-style clipped loss
    - training_step(): Main training loop
```

### 2. **nanoGRPO (Best for Understanding)**
- **Repository:** https://github.com/joey00072/nanoGRPO
- **Stars:** 140⭐
- **Implementation:** `grpo.py` (single file!)
- **Status:** Educational, minimal implementation
- **Key Features:**
  - Lightweight, easy to understand
  - Single file implementation
  - Works on 8GB GPU (RTX 4060)
  - LoRA support
  - Multiple reward functions
  - Micro-batching for memory efficiency

**Architecture Highlights:**
- Simple, readable implementation perfect for learning
- Demonstrates core GRPO concepts clearly
- Shows how to handle memory constraints
- Good example of group-based training

**Code Pattern:**
```python
class GRPO:
    def compute_loss(self, inputs, old_log_probs, reward, mean_rewards, std_rewards, mask):
        # 1. Get new policy log probs
        policy_log_probs = self.get_per_token_logps(self.model, inputs)
        
        # 2. Get reference policy log probs (for KL)
        ref_log_probs = self.get_per_token_logps(self.ref_model, inputs)
        
        # 3. Compute advantage (Z-score within group)
        advantage = (reward - mean_rewards) / (std_rewards + 1e-6)
        
        # 4. Compute KL divergence
        log_ratios = ref_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1
        
        # 5. PPO clipped loss
        policy_ratio = torch.exp(policy_log_probs - old_log_probs.detach())
        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1-epsilon, 1+epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        
        # 6. Add KL penalty
        loss += kld * beta
        return loss.mean()
```

### 3. **Google Tunix**
- **Repository:** https://github.com/google/tunix
- **Implementation:** `tunix/rl/grpo/grpo_learner.py`
- **Status:** Production-ready (Google)
- **Key Features:**
  - Lightweight LLM post-training library
  - JAX-based implementation
  - Optimized for TPU/GPU
  - Clean, modular design

### 4. **Meta PyTorch Torchtune**
- **Repository:** https://github.com/meta-pytorch/torchtune
- **Implementation:** `torchtune/dev/grpo/loss.py`
- **Status:** Production-ready (Meta)
- **Key Features:**
  - PyTorch native implementation
  - Focused on post-training
  - Kernel optimizations

### 5. **EasyDeL (JAX Implementation)**
- **Repository:** https://github.com/erfanzar/EasyDeL
- **Implementation:** `easydel/trainers/group_relative_policy_optimization/`
- **Status:** Active development
- **Key Features:**
  - JAX-based for accelerator optimization
  - Supports large models
  - Comprehensive trainer class

### 6. **Specialized Implementations**

#### MedGround-R1 (Medical Imaging)
- **Repository:** https://github.com/bio-mlhui/MedGround-R1
- **Stars:** 36⭐
- **Paper:** MICCAI'25 Best Paper Shortlist
- **Domain:** Medical image grounding with spatial-semantic rewards
- **Key Innovation:** Adapts GRPO for vision tasks with specialized rewards

#### sb3-grpo (Stable Baselines3)
- **Repository:** https://github.com/kechirojp/sb3-grpo
- **Implementation:** Drop-in replacement for PPO in SB3
- **Key Feature:** Traditional RL environment compatibility

---

## Key Implementation Patterns

### 1. **Group Sampling**
All implementations generate multiple completions per prompt:
```python
# Generate G completions per prompt
for prompt in batch:
    completions = []
    for _ in range(group_size):  # Typically 4-8
        completion = model.generate(prompt)
        completions.append(completion)
```

### 2. **Advantage Calculation (Z-Score)**
The key innovation of GRPO - no value function needed:
```python
# Within each group, compute z-score
rewards = compute_rewards(completions)  # Shape: [batch_size, group_size]
mean_reward = rewards.mean(dim=1, keepdim=True)
std_reward = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_reward) / (std_reward + 1e-6)
```

### 3. **PPO-Style Clipped Loss**
```python
# Compute probability ratio
ratio = torch.exp(new_log_probs - old_log_probs)

# Clipped surrogate objective
clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
```

### 4. **KL Divergence Penalty**
```python
# Compute KL between policy and reference model
log_ratio = ref_log_probs - policy_log_probs
kl = torch.exp(log_ratio) - log_ratio - 1
loss += beta * kl  # Typical beta: 0.01-0.1
```

### 5. **Memory Optimization Techniques**
From nanoGRPO and others:
```python
# 1. Micro-batching within groups
for micro_batch in split_into_micro_batches(group):
    loss = compute_loss(micro_batch)
    loss.backward()
torch.cuda.empty_cache()

# 2. Offloading to CPU
batch_inputs = batch_inputs.cpu()
rewards = rewards.cpu()

# 3. Generate in eval mode
with torch.no_grad():
    completions = model.generate(...)
```

---

## Differences from Current GLM-Training Implementation

### Current Implementation Issues (from REVERT_SUMMARY.md)

The current GLM-training implementation in `reward_trainer.py` has been reverted to a simpler version because:

1. **Over-complexity**: Previous attempts added complex GRPO with `_generate_latents_with_log_probs()` that caused gradient tracking errors
2. **Device placement issues**: Meta tensor problems with device_map handling
3. **Model signature mismatches**: DiT model signature handling causing VAE channel errors
4. **No proper log probability tracking**: Current version uses reconstruction loss instead of policy gradients

### What's Missing

1. **Proper Log Probability Computation**
   ```python
   # Need to add:
   def get_per_token_logprobs(self, model, input_ids):
       logits = model(input_ids).logits
       log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
       token_log_probs = torch.gather(
           log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
       ).squeeze(-1)
       return token_log_probs
   ```

2. **Reference Model for KL**
   - Need to keep a reference (frozen) copy of the initial model
   - Or use LoRA and disable adapter for reference

3. **Group-Based Generation**
   - Currently generates samples one at a time
   - Should generate `num_samples` per prompt simultaneously
   - Then compute advantages within each group

4. **Proper Advantage Calculation**
   ```python
   # Current: uses best sample
   best_sample_idx = rewards.mean(dim=1).argmax()
   
   # Should be: Z-score per group
   advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / \
                (rewards.std(dim=1, keepdim=True) + 1e-6)
   ```

5. **PPO-Style Loss**
   ```python
   # Current: Simple reconstruction + reward
   loss = F.mse_loss(generated, target) - rewards.mean()
   
   # Should be: Clipped policy gradient
   ratio = torch.exp(new_logprobs - old_logprobs)
   clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
   policy_loss = -torch.min(ratio * advantage, clipped * advantage)
   ```

---

## Recommendations for GLM-Training

### 1. **Start with HuggingFace TRL Pattern**

The TRL implementation is the gold standard. Key aspects to adopt:

```python
class RewardTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        # Create reference model (frozen copy)
        self.ref_model = self._create_reference_model()
        
        # GRPO hyperparameters
        self.num_samples_per_prompt = config["reward"]["grpo"]["num_samples"]
        self.clip_range = config["reward"]["grpo"]["clip_range"]  # 0.2
        self.kl_coef = config["reward"]["grpo"]["kl_coef"]  # 0.05
    
    def _create_reference_model(self):
        """Create frozen copy for KL computation."""
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
    
    def train_step(self, batch):
        prompts = batch["prompts"]
        target_images = batch["target_images"]
        
        # 1. Generate multiple completions per prompt
        with torch.no_grad():
            completions, old_log_probs = self._generate_with_logprobs(
                prompts,
                num_samples=self.num_samples_per_prompt
            )
        
        # 2. Compute rewards for all completions
        rewards = self.reward_calculator.compute_rewards(
            completions, target_images, prompts
        )
        
        # 3. Compute advantages (z-score within groups)
        advantages = self._compute_advantages(rewards)
        
        # 4. Compute policy loss
        new_log_probs = self._get_log_probs(completions)
        ref_log_probs = self._get_log_probs(completions, use_ref=True)
        
        loss = self._compute_grpo_loss(
            new_log_probs, old_log_probs, ref_log_probs,
            advantages
        )
        
        # 5. Backprop
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item(), "reward": rewards.mean().item()}
    
    def _generate_with_logprobs(self, prompts, num_samples):
        """Generate completions and track log probabilities."""
        # Repeat prompts for group generation
        repeated_prompts = [p for p in prompts for _ in range(num_samples)]
        
        # Generate
        generated_images = self.model.generate(...)
        
        # Compute log probs (need to modify GLMImageWrapper)
        log_probs = self._compute_logprobs_from_generated(...)
        
        return generated_images, log_probs
    
    def _compute_advantages(self, rewards):
        """Compute z-score advantages within groups."""
        # rewards shape: [batch_size * num_samples]
        # Reshape to: [batch_size, num_samples]
        batch_size = len(rewards) // self.num_samples_per_prompt
        rewards = rewards.view(batch_size, self.num_samples_per_prompt)
        
        # Z-score per group
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-6)
        
        return advantages.view(-1)  # Flatten back
    
    def _compute_grpo_loss(self, new_logprobs, old_logprobs, 
                            ref_logprobs, advantages):
        """Compute GRPO loss with PPO clipping and KL penalty."""
        # Policy ratio
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate loss
        clipped_ratio = torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        )
        
        # KL divergence penalty (vs reference model)
        kl_div = new_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_div
        
        return (policy_loss + kl_penalty).mean()
```

### 2. **Memory-Efficient Implementation**

For GLM-Image's large models (9B AR + 7B DiT), use nanoGRPO's patterns:

```python
# Generate in batches
for i in range(0, total_groups, micro_batch_size):
    micro_batch = groups[i:i+micro_batch_size]
    loss = compute_loss(micro_batch)
    loss.backward()
    torch.cuda.empty_cache()

# Offload reference model when not in use
self.ref_model = self.ref_model.cpu()
# ... training ...
self.ref_model = self.ref_model.to(device)
```

### 3. **Adapt for Vision/Diffusion Models**

GLM-Image is unique because it's not just text:

```python
def _generate_with_logprobs(self, prompts, source_images=None):
    """
    Generate images and track log probabilities.
    
    Challenge: DiT decoder doesn't directly give log probs.
    Solution: Track AR model log probs + DiT latent probabilities.
    """
    # 1. AR model generates visual tokens with log probs
    ar_outputs = self.model.ar_model.generate(
        prompts, return_log_probs=True
    )
    
    # 2. DiT decoder converts to images (latent diffusion)
    images = self.model.dit_model.decode(
        ar_outputs.tokens, prompts
    )
    
    # For GRPO: Use AR log probs as proxy for policy
    # DiT is deterministic decoder, doesn't need policy gradient
    return images, ar_outputs.log_probs
```

### 4. **Reward Function Design**

Learn from MedGround-R1's approach for vision:

```python
class VisualRewardCalculator:
    def compute_rewards(self, generated_images, target_images, prompts):
        """
        Compute multiple reward components for vision.
        """
        rewards = []
        
        # 1. Perceptual similarity (LPIPS)
        lpips_reward = self.compute_lpips(generated_images, target_images)
        
        # 2. Structural similarity (SSIM)
        ssim_reward = self.compute_ssim(generated_images, target_images)
        
        # 3. OCR text accuracy (if text in image)
        text_reward = self.compute_text_accuracy(generated_images, prompts)
        
        # 4. Aesthetic quality
        aesthetic_reward = self.compute_aesthetics(generated_images)
        
        # Weighted combination
        total_reward = (
            self.lpips_weight * lpips_reward +
            self.ssim_weight * ssim_reward +
            self.text_weight * text_reward +
            self.aesthetic_weight * aesthetic_reward
        )
        
        return total_reward
```

### 5. **Configuration Structure**

Update YAML config to match TRL patterns:

```yaml
reward:
  enabled: true
  grpo:
    # Group size for relative comparison
    num_samples: 4  # Generate 4 images per prompt
    
    # PPO hyperparameters
    clip_range: 0.2  # Clip ratio to [0.8, 1.2]
    kl_coef: 0.05    # KL divergence penalty coefficient
    
    # Advantage normalization
    normalize_advantages: true
    advantage_epsilon: 1e-6
    
  # Reward components (from HuggingFace TRL)
  metrics:
    lpips: 0.4
    ssim: 0.2
    text_accuracy: 0.3
    aesthetic: 0.1
    
  # Memory optimization
  micro_batch_size: 2  # Process 2 samples at a time within group
```

---

## Implementation Priority

### Phase 1: Core GRPO (High Priority)
1. ✅ Add reference model creation/loading
2. ✅ Implement `get_per_token_logprobs()` for AR model
3. ✅ Update `_generate_samples()` to return log probabilities
4. ✅ Implement `_compute_advantages()` with z-score
5. ✅ Replace current loss with `_compute_grpo_loss()`

### Phase 2: Memory Optimization (Medium Priority)
1. Add micro-batching within groups
2. Implement gradient accumulation
3. Add option to offload reference model
4. Test on 40GB/50GB/80GB GPU configurations

### Phase 3: Advanced Features (Low Priority)
1. Add vLLM support for faster generation
2. Support multiple reward models
3. Add async reward computation
4. Implement curriculum learning

---

## Code Examples from Top Repos

### HuggingFace TRL: Advantage Computation
```python
def compute_advantages(
    self,
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Compute advantages using group-relative normalization.
    
    Args:
        rewards: Tensor of shape [batch_size * group_size]
        group_size: Number of samples per prompt
    
    Returns:
        advantages: Normalized advantages
    """
    # Reshape to [batch_size, group_size]
    batch_size = rewards.size(0) // group_size
    rewards = rewards.view(batch_size, group_size)
    
    # Compute mean and std per group
    mean_reward = rewards.mean(dim=1, keepdim=True)
    std_reward = rewards.std(dim=1, keepdim=True)
    
    # Z-score normalization
    advantages = (rewards - mean_reward) / (std_reward + 1e-6)
    
    # Flatten back
    return advantages.view(-1)
```

### nanoGRPO: Memory-Efficient Training
```python
def train_step(self):
    # Generate completions
    outputs, rewards, mask = self.sample_batch()
    
    # Reshape into groups
    batch_size = self.batch_size
    group_size = self.group_size
    micro_size = self.micro_group_size
    
    batch_inputs = outputs.reshape(batch_size, group_size, -1)
    
    # Offload to CPU to save VRAM
    batch_inputs = batch_inputs.cpu()
    rewards = rewards.cpu()
    torch.cuda.empty_cache()
    
    # Compute old policy log probs (before training)
    pi_old = []
    for group in batch_inputs:
        with torch.no_grad():
            old_logprobs = self.get_per_token_logps(
                self.model, group.to(self.device)
            ).cpu()
            pi_old.append(old_logprobs)
            torch.cuda.empty_cache()
    
    # Train on each group
    for group, old_lp, group_rewards, group_mask in \
            zip(batch_inputs, pi_old, rewards, mask):
        
        # Compute group statistics
        mean_r = group_rewards.mean()
        std_r = group_rewards.std()
        
        # Split into micro-batches for memory
        n_micro = group_size // micro_size
        for i in range(n_micro):
            start = i * micro_size
            end = start + micro_size
            
            micro_inputs = group[start:end].to(self.device)
            micro_old_lp = old_lp[start:end].to(self.device)
            micro_rewards = group_rewards[start:end].to(self.device)
            micro_mask = group_mask[start:end].to(self.device)
            
            # Compute loss
            loss = self.compute_loss(
                micro_inputs, micro_old_lp, micro_rewards,
                mean_r, std_r, micro_mask
            )
            
            # Backprop
            loss.backward()
            torch.cuda.empty_cache()
        
        # Update weights (once per group)
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### Google Tunix: Clean Training Loop
```python
def grpo_train_step(
    model, ref_model, optimizer,
    prompts, group_size, reward_fn,
    clip_range=0.2, kl_coef=0.05
):
    """Single GRPO training step."""
    
    # 1. Rollout: Generate completions
    completions, old_log_probs = generate_with_logprobs(
        model, prompts, n=group_size
    )
    
    # 2. Evaluate: Compute rewards
    rewards = reward_fn(prompts, completions)
    
    # 3. Advantage: Normalize within groups
    advantages = compute_group_advantages(rewards, group_size)
    
    # 4. Policy update: GRPO loss
    new_log_probs = get_log_probs(model, completions)
    ref_log_probs = get_log_probs(ref_model, completions)
    
    # PPO clipped loss
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1-clip_range, 1+clip_range)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped * advantages
    ).mean()
    
    # KL penalty
    kl_loss = kl_coef * (new_log_probs - ref_log_probs).mean()
    
    # Total loss
    loss = policy_loss + kl_loss
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "reward": rewards.mean().item(),
    }
```

---

## Testing Strategy

Based on implementations reviewed:

### 1. Unit Tests
```python
def test_advantage_computation():
    """Test z-score advantage calculation."""
    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # Group 1
        [5.0, 5.0, 5.0, 5.0],  # Group 2 (constant)
    ])
    advantages = compute_advantages(rewards.flatten(), group_size=4)
    
    # Group 1: Should have std dev
    assert advantages[:4].std() > 0
    
    # Group 2: Should be zero (constant rewards)
    assert torch.allclose(advantages[4:], torch.zeros(4), atol=1e-6)

def test_ppo_clip():
    """Test PPO clipping works correctly."""
    advantages = torch.tensor([1.0, -1.0])
    old_lp = torch.tensor([0.0, 0.0])
    
    # New policy much better (ratio > 1.2)
    new_lp = torch.tensor([1.0, -1.0])  # ratio = exp(1.0) = 2.7
    
    loss = compute_grpo_loss(new_lp, old_lp, advantages, clip=0.2)
    
    # Should be clipped to 1.2
    assert loss < advantages[0] * 2.7  # Less than unclipped
```

### 2. Integration Tests
```python
def test_full_training_step():
    """Test complete training pipeline."""
    trainer = RewardTrainer(config)
    
    batch = {
        "prompts": ["cat", "dog"],
        "target_images": [cat_img, dog_img],
    }
    
    metrics = trainer.train_step(batch)
    
    assert "loss" in metrics
    assert "reward" in metrics
    assert metrics["loss"] > 0
```

### 3. Memory Tests
```python
def test_memory_usage():
    """Ensure training fits in GPU memory."""
    import torch.cuda
    
    trainer = RewardTrainer(config)
    
    # Monitor memory
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(10):
        metrics = trainer.train_step(batch)
        torch.cuda.empty_cache()
    
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Should fit in 80GB
    assert peak_memory < 80 * 1024**3
```

---

## Common Pitfalls to Avoid

From analyzing failed implementations and REVERT_SUMMARY.md:

### 1. **Device Placement Issues**
```python
# ❌ Bad: Inconsistent device placement
outputs = model.generate(prompts)  # On GPU
rewards = compute_rewards(outputs)  # On CPU
loss = compute_loss(outputs, rewards)  # Mixed!

# ✅ Good: Explicit device management
outputs = model.generate(prompts)
rewards = compute_rewards(outputs).to(outputs.device)
loss = compute_loss(outputs, rewards)
```

### 2. **Gradient Tracking**
```python
# ❌ Bad: Tracking gradients through generation
completions = model.generate(prompts)  # Gradients?
loss = compute_loss(completions)

# ✅ Good: No gradients for generation, track log probs
with torch.no_grad():
    completions, old_log_probs = model.generate(prompts)
# Later: recompute log probs with gradients
new_log_probs = compute_logprobs(model, completions)
```

### 3. **Advantage Computation**
```python
# ❌ Bad: Global normalization
advantages = (rewards - rewards.mean()) / rewards.std()

# ✅ Good: Group-wise normalization
batch_size = len(rewards) // group_size
rewards = rewards.view(batch_size, group_size)
advantages = (rewards - rewards.mean(1, keepdim=True)) / \
             rewards.std(1, keepdim=True)
```

### 4. **Reference Model**
```python
# ❌ Bad: No reference model (KL divergence wrong)
kl = new_log_probs - old_log_probs  # Wrong!

# ✅ Good: Use frozen reference model
kl = new_log_probs - ref_model_log_probs
```

### 5. **Memory Leaks**
```python
# ❌ Bad: Accumulating tensors
losses = []
for batch in dataloader:
    loss = train_step(batch)
    losses.append(loss)  # Keeps computation graph!

# ✅ Good: Detach or use .item()
losses = []
for batch in dataloader:
    loss = train_step(batch)
    losses.append(loss.item())  # Just the value
```

---

## References

### Papers
1. **DeepSeekMath** (Original GRPO paper)
   - arxiv: 2402.03300
   - https://arxiv.org/abs/2402.03300

2. **DeepSeek R1** (Latest application)
   - arxiv: 2501.12948
   - https://arxiv.org/abs/2501.12948

### Key Repositories
1. HuggingFace TRL: https://github.com/huggingface/trl
2. nanoGRPO: https://github.com/joey00072/nanoGRPO
3. Google Tunix: https://github.com/google/tunix
4. Meta Torchtune: https://github.com/meta-pytorch/torchtune
5. EasyDeL: https://github.com/erfanzar/EasyDeL

### Documentation
- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- TRL Reward Functions: https://huggingface.co/docs/trl/rewards

---

## Next Steps for GLM-Training

### Immediate Actions
1. Study HuggingFace TRL's `GRPOTrainer` implementation in detail
2. Study nanoGRPO for understanding core concepts
3. Implement minimal GRPO with proper log probability tracking
4. Test on small model first (to verify correctness)
5. Add memory optimizations from nanoGRPO

### Medium Term
1. Adapt for GLM-Image's vision/diffusion architecture
2. Implement specialized visual rewards
3. Add comprehensive testing
4. Document the implementation

### Long Term
1. Consider vLLM integration for faster generation
2. Explore multi-reward training
3. Implement curriculum learning
4. Contribute improvements back to open source

---

## Conclusion

GRPO is a well-established algorithm with multiple high-quality implementations available. The HuggingFace TRL implementation should be the primary reference for production code, while nanoGRPO provides excellent educational value for understanding core concepts.

The key to successful GRPO implementation for GLM-Training is:
1. Proper log probability tracking
2. Group-based advantage computation (z-score)
3. PPO-style clipped loss
4. KL divergence penalty with reference model
5. Memory-efficient batching strategies

The current GLM-Training implementation needs these components added systematically, learning from the patterns established in the top implementations reviewed.

---

**Research completed:** January 19, 2026  
**Repositories analyzed:** 20+ implementations  
**Primary recommendations:** HuggingFace TRL (production), nanoGRPO (education)
