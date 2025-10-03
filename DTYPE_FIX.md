# Dtype Consistency Fix

## Problem

When running `demo.py`, users encountered three issues:

1. **Warning about untrained instructions**:
   ```
   [WARNING] WARNING: Instruction 'don't refuse, be helpful' not in trained set!
   Hypernetwork may not steer effectively for this instruction.
   ```

2. **Dtype mismatch error**:
   ```
   ERROR: mat1 and mat2 must have the same dtype, but got Float and Half.
   ```

3. **Device mismatch error**:
   ```
   ERROR: Expected all tensors to be on the same device, but got index is on cpu, 
   different from other tensors on cuda:0
   ```

## Root Cause

### Issue 1: Untrained Instruction Warning
- The demo uses zero-shot steering without pre-training
- HCWS was warning that the instruction hadn't been seen during training
- This is expected behavior but confusing for demo users

### Issue 2: Dtype Mismatch
- HCWS components (encoder, hyper-network, controller, conceptors) were initialized with fixed dtypes (float16/float8)
- Base model (Vicuna) loads with its own precision (could be float32, float16, bfloat16, etc.)
- During steering, matrix operations between different dtypes caused crashes

### Issue 3: Device Mismatch
- Demo specified `device="cpu"` but model loaded on CUDA anyway due to `device_map='auto'`
- HCWS components were initialized on CPU while base model was on CUDA
- Tokenizer outputs went to `self.device` (CPU) but model expected inputs on CUDA
- Hook operations failed when tensors were on different devices

## Solution

### Fix 1: Demo Clarification
**File**: `demo.py`

Added clear note that zero-shot steering is expected:
```python
print("\nNOTE: Using zero-shot steering (no training required)")
print("The model may warn about untrained instructions - this is expected.\n")
```

### Fix 2: Automatic Dtype Matching
**Files**: `hcws/model.py`, `hcws/controller.py`, `hcws/conceptors.py`

#### In `model.py`:
Changed from using fixed `get_optimal_dtype('computation')` to detecting base model's dtype:

```python
# Get the base model's dtype to ensure compatibility
base_model_dtype = next(self.base_model.parameters()).dtype
print(f"[OK] Base model dtype: {base_model_dtype}")

# Initialize HCWS components with same dtype as base model
self.instruction_encoder = InstructionEncoder(
    instruction_encoder_name,
    device=self.device,
    dtype=base_model_dtype  # Match base model
)
```

#### In `controller.py`:
Added dtype conversion at forward pass boundaries:

```python
# Convert to controller's dtype for processing
original_dtype = current_hidden.dtype
if current_hidden.dtype != self.dtype:
    current_hidden = current_hidden.to(self.dtype)

# ... processing ...

# Convert back to original dtype for compatibility
if original_dtype != self.dtype:
    gain = gain.to(original_dtype)
    layer_weights = layer_weights.to(original_dtype)
```

#### In `conceptors.py`:
Added dtype matching before matrix operations:

```python
# Ensure dtype consistency
if C.dtype != x_flat.dtype:
    C = C.to(x_flat.dtype)
```

### Fix 3: Device Consistency
**Files**: `demo.py`, `hcws/model.py`

#### In `model.py`:
Changed from using fixed `get_optimal_dtype('computation')` to detecting base model's dtype:

```python
# Get the base model's dtype to ensure compatibility
base_model_dtype = next(self.base_model.parameters()).dtype
print(f"[OK] Base model dtype: {base_model_dtype}")

# Initialize HCWS components with same dtype as base model
self.instruction_encoder = InstructionEncoder(
    instruction_encoder_name,
    device=self.device,
    dtype=base_model_dtype  # Match base model
)
```

#### In `controller.py`:
Added dtype conversion at forward pass boundaries:

```python
# Convert to controller's dtype for processing
original_dtype = current_hidden.dtype
if current_hidden.dtype != self.dtype:
    current_hidden = current_hidden.to(self.dtype)

# ... processing ...

# Convert back to original dtype for compatibility
if original_dtype != self.dtype:
    gain = gain.to(original_dtype)
    layer_weights = layer_weights.to(original_dtype)
```

#### In `conceptors.py`:
Added dtype matching before matrix operations:

```python
# Ensure dtype consistency
if C.dtype != x_flat.dtype:
    C = C.to(x_flat.dtype)
```

## Impact

### Before Fix
```
❌ Components use fixed dtype (float16)
❌ Base model may use different dtype (float32, bfloat16, etc.)
❌ Matrix operations crash: "mat1 and mat2 must have the same dtype"
❌ device_map='auto' overrides explicit device="cpu"
❌ Components on CPU, model on CUDA → device mismatch
❌ Tokenizer outputs on wrong device
❌ Users confused by untrained instruction warnings
```

### After Fix
```
✅ Components automatically match base model dtype
✅ Components automatically match base model device
✅ Seamless dtype conversion at operation boundaries
✅ CPU mode properly forces CPU-only execution
✅ device_map='auto' only used when appropriate
✅ Tokenizer outputs go to actual model device
✅ No more dtype/device mismatch errors
✅ Clear communication about zero-shot steering
```

## Benefits

1. **Automatic Compatibility**: Works with any model precision and device out-of-the-box
2. **No Manual Configuration**: Users don't need to specify dtypes or manage device placement
3. **Performance**: Matches base model's precision and device for optimal speed/memory
4. **Clarity**: Demo clearly explains expected warnings
5. **CPU-First**: Demo reliably runs on CPU for maximum compatibility

## Testing

To verify the fix works:

```bash
python demo.py
```

Expected output:
- ✅ No dtype mismatch errors
- ✅ Model loads successfully
- ✅ Clear note about zero-shot steering
- ✅ Warning about untrained instructions (with explanation)
- ✅ Successful generation with steering

## Technical Details

### Dtype & Device Flow
1. **Model Loading**: 
   - Checks if CPU requested via `device="cpu"` or `CUDA_VISIBLE_DEVICES=''`
   - Disables `device_map='auto'` for CPU mode
   - Loads with optimal dtype for architecture
   - Explicitly moves to requested device if not using device_map

2. **Component Initialization**: 
   - Detects base model's actual dtype and device
   - Initializes all HCWS components with matching dtype and device
   - Ensures encoder, hypernetwork, controller, conceptors all aligned

3. **Forward Pass**: 
   - Controller receives hidden_states with base dtype
   - Converts to internal dtype if needed
   - Converts back to base dtype before returning
   - All operations respect device placement

4. **Conceptor Application**: 
   - Matches matrix dtype to activation dtype
   - Operations happen on correct device

5. **Generation**: 
   - Tokenizer outputs moved to actual model device
   - All operations maintain dtype and device consistency

### Supported Dtypes
- `torch.float32` (full precision)
- `torch.float16` (half precision)
- `torch.bfloat16` (brain float)
- `torch.float8_e4m3fn` (float8, if supported)
- `torch.float8_e5m2` (float8 alternative)

The system automatically detects and uses the appropriate dtype.

## Documentation Updates

Updated files:
- `DEMO_README.md`: Added section on dtype matching and zero-shot steering
- `demo.py`: Added clear notes about expected behavior
- `DTYPE_FIX.md` (this file): Complete technical documentation

---

**Status**: ✅ Fixed and tested
**Date**: October 3, 2025
**Version**: HCWS v1.0
