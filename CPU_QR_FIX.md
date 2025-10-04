# CPU QR Decomposition Fix

## Problem
When running on CPU with float16 precision, PyTorch's QR decomposition (`torch.qr`) fails with:
```
ERROR: "geqrf_cpu" not implemented for 'Half'
```

This is a known PyTorch limitation - the `geqrf_cpu` operation (QR decomposition) doesn't support float16 on CPU.

## Root Cause
The HCWS Conceptor operations use QR decomposition to ensure orthogonality of the U matrix:
1. **Conceptor initialization**: `_initialize_orthogonal()` uses `torch.qr`
2. **Conceptor matrix computation**: `get_matrix()` uses `torch.qr` to normalize U
3. **HyperNetwork**: `generate_conceptor_bank()` uses `torch.qr` to orthogonalize U parameters
4. **HyperNetwork**: `generate_layer_conceptor()` uses `torch.qr`

When the model loads in float16 on CPU, these operations fail.

## Solution
Implemented dtype conversion for QR operations on CPU:

1. **Detect compute dtype**:
   ```python
   compute_dtype = torch.float32 if self.device.type == 'cpu' and tensor.dtype == torch.float16 else tensor.dtype
   ```

2. **Convert to float32 for QR**:
   ```python
   tensor_compute = tensor.to(compute_dtype) if compute_dtype != tensor.dtype else tensor
   result, _ = torch.qr(tensor_compute)
   ```

3. **Convert back to float16**:
   ```python
   if compute_dtype != tensor.dtype:
       result = result.to(tensor.dtype)
   ```

## Files Modified

### hcws/conceptors.py
1. **`_initialize_orthogonal()`**: Convert to float32 for QR, then back to target dtype
2. **`get_matrix()`**: Convert U and s to float32 for QR, convert C back to original dtype

### hcws/hyper_network.py
1. **`generate_conceptor_bank()`**: Convert U_layer to float32 for QR, then back
2. **`generate_layer_conceptor()`**: Convert U to float32 for QR, convert results back

## Technical Details

### Why Float32 on CPU?
- **GPU**: Supports float16 for all operations including QR decomposition
- **CPU**: Limited float16 support - many linear algebra operations (like `geqrf`) only implemented for float32/float64
- **Solution**: Use float32 for these specific CPU operations, maintain float16 elsewhere for memory efficiency

### Performance Impact
- **Minimal**: QR operations are infrequent (only during initialization and conceptor generation)
- **Memory**: Temporary float32 tensors are small (hidden_dim × rank matrices)
- **Accuracy**: Float32 provides better numerical stability for QR decomposition anyway

### Why Not Just Use Float32 Everywhere?
- Model weights (13GB Vicuna) benefit from float16 memory savings
- Most operations (matmul, attention) work fine in float16 on CPU
- Only specific operations like QR need float32

## Testing

Run the demo to verify:
```bash
python demo.py
```

**Expected behavior**:
- ✅ Model loads in float16 on CPU
- ✅ No "geqrf_cpu not implemented for 'Half'" error
- ✅ Conceptors initialize successfully
- ✅ Steering operations work (with warning about untrained instructions)
- ✅ Text generation produces visible output

**Known limitations**:
- Zero-shot steering (no training) may be less effective
- Warnings about untrained instructions are expected
- Baseline responses may still be minimal due to model behavior

## Related Fixes
This fix complements the previous dtype consistency fixes:
- **FINAL_DTYPE_FIX.md**: Overall dtype matching across components
- **CPU_QR_FIX.md**: Specific CPU operation compatibility

Both are needed for full CPU compatibility with float16 models.
