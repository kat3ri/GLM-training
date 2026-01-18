"""
Distributed training utilities for multi-GPU support.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """
    Initialize distributed training environment.
    
    Returns:
        tuple: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def init_distributed(backend="nccl"):
    """
    Initialize the distributed process group.
    
    Args:
        backend: Distributed backend (nccl or gloo)
    """
    rank, world_size, local_rank = setup_distributed()
    
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduce operation
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor):
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def wrap_model_ddp(model, device_ids=None, find_unused_parameters=False):
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        Wrapped model
    """
    if not dist.is_initialized():
        return model
    
    if device_ids is None:
        device_ids = [get_rank() % torch.cuda.device_count()]
    
    model = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters
    )
    
    return model


def save_on_main_process(save_fn, *args, **kwargs):
    """
    Execute save function only on main process.
    
    Args:
        save_fn: Function to call for saving
        *args, **kwargs: Arguments to pass to save_fn
    """
    if is_main_process():
        save_fn(*args, **kwargs)
    barrier()
