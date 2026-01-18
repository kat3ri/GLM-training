# Fix Summary: Meta Tensor Issue in Model Initialization

## Problem
The training crashed with:
```
NotImplementedError: Cannot copy out of meta tensor; no data! 
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() 
when moving module from meta to a different device.
```
at line 155 in `train_dit.py` when calling `DiTTrainer(config)`.

## Root Cause
When loading models using `GlmImagePipeline.from_pretrained()` with a `device_map` parameter (e.g., "auto", "balanced"), the pipeline may use meta tensors for efficient memory management or distributed loading. Meta tensors are placeholder tensors without actual data that allow models to be loaded without immediately allocating memory.

The original code in `reward_trainer.py` always called `.to(self.device)` after loading the model:

```python
# Line 62-69: Load model with device_map
model = GLMImageWrapper(
    model_name=model_config["name"],
    component=model_config["component"],
    torch_dtype=torch_dtype,
    device_map="cpu" if self.world_size > 1 else model_config["device_map"],
)

# Line 69: ALWAYS called .to() - this causes the error with meta tensors!
model = model.to(self.device)
```

**The Issue:** When `device_map` is "auto" or "balanced", the model components are already placed on the correct device(s) by the `from_pretrained` method. Calling `.to()` afterwards:
1. Is unnecessary (model is already on the right device)
2. Causes errors if meta tensors are used (cannot copy data from meta tensors)

## Solution

Modified `_build_model()` in `reward_trainer.py` to only call `.to()` when necessary:

```python
# Determine device_map based on distributed training
if self.world_size > 1:
    # In distributed training, use cpu device_map and let DDP handle device placement
    device_map = "cpu"
else:
    device_map = model_config["device_map"]

model = GLMImageWrapper(
    model_name=model_config["name"],
    component=model_config["component"],
    torch_dtype=torch_dtype,
    device_map=device_map,
)

# Move to device only when using device_map="cpu"
# When device_map is "auto", "balanced", etc., the model is already on the correct device(s)
# and calling .to() can cause meta tensor errors
if device_map == "cpu":
    # Model was loaded on CPU, move to the target device
    model = model.to(self.device)
```

### Why This Works

**Scenario 1: Single GPU with device_map="auto"**
- Model is loaded with device_map="auto" (mapped to "balanced" in GLMImageWrapper)
- HuggingFace places the model on the available GPU automatically
- `.to()` is NOT called (device_map != "cpu")
- ✓ Model is already on GPU, no meta tensor error

**Scenario 2: Distributed Training (Multi-GPU)**
- device_map is overridden to "cpu"
- Model is loaded on CPU without meta tensors
- `.to(self.device)` IS called to move to cuda:{local_rank}
- Model is then wrapped with DDP
- ✓ Proper device placement for distributed training

**Scenario 3: Single GPU with device_map="cpu"**
- Model is loaded on CPU
- `.to(self.device)` IS called to move to GPU
- ✓ Respects user's intent to start on CPU but moves to GPU for training

## Benefits

✅ **No Meta Tensor Errors**: Avoids calling `.to()` on models with meta tensors  
✅ **Correct Device Placement**: Models end up on the right device(s) in all scenarios  
✅ **Distributed Training Support**: Works correctly with DDP  
✅ **Single GPU Support**: Works with all device_map values ("auto", "balanced", "cpu")  
✅ **Backward Compatible**: Doesn't break existing functionality  

## Scope

This fix applies to all trainers:
- `RewardTrainer` - Fixed in `_build_model()`
- `DiTTrainer` - Inherits from RewardTrainer, automatically fixed
- `ARTrainer` - Inherits from RewardTrainer, automatically fixed

## Testing

Created `test_meta_tensor_fix.py` that validates:
1. **Logic Test**: Verifies device_map determination and `.to()` call logic
2. **Mock Initialization Test**: Tests model initialization with different device_map values

Test cases covered:
- Single GPU with auto, balanced, and cpu device_map
- Multi-GPU with various device_map configurations
- Verification that `.to()` is only called when device_map="cpu"

## Files Modified

- `glm_training/trainers/reward_trainer.py`:
  - Updated `_build_model()` to conditionally call `.to()` based on device_map
  - Added detailed comments explaining the fix
- `test_meta_tensor_fix.py`:
  - New test file to verify the fix logic

## How to Use

No changes needed in user code. The fix is transparent:

```bash
# Works now without meta tensor errors!
python train_dit.py --config configs/t2i_training.yaml

# Works in distributed mode too!
torchrun --nproc_per_node=4 train_dit.py --config configs/t2i_training.yaml
```

## References

- [PyTorch Meta Tensors](https://pytorch.org/docs/stable/meta.html)
- [HuggingFace device_map](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [PyTorch Module.to() documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)
