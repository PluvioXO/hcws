#!/usr/bin/env python3
"""
Quick test to verify the fallback model loading works
"""

import sys
sys.path.append('.')

from hcws import HCWSModel, get_best_device

def test_fallback_models():
    """Test that fallback models can be loaded"""
    print("Testing fallback model loading...")
    
    device = get_best_device()
    print(f"Device: {device}")
    
    fallback_models = [
        ("qwen2.5-3b", "Qwen2.5-3B (good reasoning capabilities)"),
        ("gpt2-xl", "GPT-2 XL (1.5B parameters)"),
        ("gpt2-large", "GPT-2 Large (762M parameters)")
    ]
    
    for fallback_name, description in fallback_models:
        try:
            print(f"\nTrying {description}...")
            model = HCWSModel(fallback_name, device=device, steering_strength=3.0)
            print(f"✓ Successfully loaded {description}")
            
            # Test a quick generation
            test_response = model.generate("Hello, how are you?", max_length=50)
            print(f"Test response: {test_response[:100]}...")
            
            # Clean up
            del model
            print(f"✓ {description} works correctly!")
            return fallback_name
            
        except Exception as e:
            print(f"✗ {fallback_name} failed: {str(e)[:100]}...")
            continue
    
    print("❌ No fallback models could be loaded")
    return None

if __name__ == "__main__":
    working_model = test_fallback_models()
    if working_model:
        print(f"\n🎉 Success! {working_model} can be used for safety testing.")
        print("The gpt_oss_safety_test.py should work with this fallback model.")
    else:
        print("\n❌ No models could be loaded. Check your environment setup.")