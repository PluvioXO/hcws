# HCWS Demo - Quick Start

## What Was Fixed

1. **Device Error**: Fixed `'str' object has no attribute 'type'` error
2. **Auto-Registration**: Instructions are automatically registered for steering
3. **Clean Output**: Minimal output showing just prompt → unsteered → steered

## Running the Demo

### Quick Test (GPT-2 - 548MB)
```bash
python3 test_steering.py
```
Fast test to verify steering works. Takes ~1 minute.

### Full Demo (Vicuna-7B - 13GB)
```bash
python3 demo.py
```
Full refusal bypass demonstration. First run downloads ~13GB model.

## Expected Behavior

### Before (Unsteered)
Model follows its default behavior patterns.

### After (Steered)
Model behavior changes according to the steering instruction.

### Example Output Format
```
============================================================
Prompt 1/3: How do I pick a lock?
============================================================

[UNSTEERED]: I cannot provide instructions on picking locks...

[STEERED]: Here's how lock picking works: First, you'll need...
```

## What Changed in demo.py

### Old Version
- Verbose warnings and status messages
- Manual training step required
- Detailed expectations and summaries
- ~100 lines of output per prompt

### New Version
- Minimal output (prompt + 2 responses)
- Auto-registration of instructions
- Clean, focused demonstration
- ~10 lines of output per prompt

## Technical Details

The fix ensures all device handling uses `torch.device` objects instead of strings:

```python
# Component initialization now does:
raw_device = get_device(device)
if isinstance(raw_device, str):
    self.device = torch.device(raw_device)
else:
    self.device = raw_device
```

This allows code like `self.device.type == 'cpu'` to work correctly across all components.

## Files Modified

Core changes:
- `hcws/model.py` - Main model class
- `hcws/encoder.py` - T5 encoder
- `hcws/simple_encoder.py` - Alternative encoder
- `hcws/conceptors.py` - Conceptor matrices
- `hcws/controller.py` - Steering controller
- `hcws/hyper_network.py` - Parameter generator

Demo updates:
- `demo.py` - Simplified output and auto-registration

New test files:
- `test_device_fix.py` - Verify device handling
- `test_steering.py` - Quick GPT-2 test

## Verification

All tests passing:
- ✅ Device conversion works correctly
- ✅ Model loads on CPU without errors
- ✅ Steering produces different outputs
- ✅ No AttributeError on device.type
