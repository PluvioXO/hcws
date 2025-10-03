#!/usr/bin/env python3
"""
Quick test script to verify device and dtype fixes work correctly.
"""

# MUST set BEFORE any torch imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("=" * 60)
print("Testing HCWS Device & Dtype Fixes")
print("=" * 60)

print("\n1. Checking environment...")
print(f"   CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}'")

print("\n2. Importing torch...")
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

print("\n3. Testing simple tensor creation...")
test_tensor = torch.randn(3, 3)
print(f"   Test tensor device: {test_tensor.device}")
print(f"   Test tensor dtype: {test_tensor.dtype}")

print("\n4. Loading GPT-2 (smaller model for quick test)...")
from hcws import HCWSModel

try:
    model = HCWSModel("gpt2", device="cpu", steering_strength=2.0)
    print("   ✓ Model loaded successfully!")
    
    # Check actual devices
    print(f"\n5. Checking component devices...")
    base_device = next(model.base_model.parameters()).device
    print(f"   Base model device: {base_device}")
    print(f"   Base model dtype: {next(model.base_model.parameters()).dtype}")
    
    if hasattr(model, 'controller'):
        controller_device = next(model.controller.parameters()).device
        controller_dtype = next(model.controller.parameters()).dtype
        print(f"   Controller device: {controller_device}")
        print(f"   Controller dtype: {controller_dtype}")
    
    print(f"\n6. Testing generation (baseline)...")
    output = model.generate(
        "Hello, how are you?",
        max_length=20,
        temperature=0.7
    )
    print(f"   Output: {output[:100]}...")
    print("   ✓ Baseline generation works!")
    
    print(f"\n7. Testing generation (with steering)...")
    output_steered = model.generate(
        "Hello, how are you?",
        steering_instruction="be very enthusiastic",
        max_length=20,
        temperature=0.7
    )
    print(f"   Output: {output_steered[:100]}...")
    print("   ✓ Steered generation works!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nYour fixes are working correctly!")
    print("The demo should now run without device/dtype errors.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("TEST FAILED")
    print("=" * 60)
