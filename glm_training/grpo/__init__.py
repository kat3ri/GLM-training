"""
GRPO (Group Relative Policy Optimization) implementation.

A clean, minimal implementation following HuggingFace TRL and nanoGRPO patterns.
"""

from .loss import compute_advantages, compute_grpo_loss
from .utils import simple_reward_function, pil_to_tensor, tensor_to_pil
from .trainer import MinimalGRPOTrainer

__all__ = [
    "compute_advantages",
    "compute_grpo_loss",
    "simple_reward_function",
    "pil_to_tensor",
    "tensor_to_pil",
    "MinimalGRPOTrainer",
]
