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
    
    Example:
        >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0])
        >>> advantages = compute_advantages(rewards, group_size=4)
        >>> # Group 1: [1,2,3,4] -> mean=2.5, std~1.29 -> z-scores
        >>> # Group 2: [5,5,5,5] -> mean=5.0, std=0.0 -> all zeros
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
    
    Example:
        >>> new_lp = torch.randn(8, requires_grad=True)
        >>> old_lp = torch.randn(8)
        >>> advantages = torch.randn(8)
        >>> loss, metrics = compute_grpo_loss(new_lp, old_lp, advantages)
        >>> loss.backward()  # Should work
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
        "ratio_max": ratio.max().item(),
        "ratio_min": ratio.min().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }
    
    return loss, metrics
