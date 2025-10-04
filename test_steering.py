#!/usr/bin/env python3
"""Quick test with GPT-2 to verify steering works."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import warnings
warnings.filterwarnings("ignore")

from hcws import HCWSModel

def main():
    print("\n" + "="*60)
    print("  Quick GPT-2 Steering Test")
    print("="*60)
    
    # Load smaller model for quick test
    print("\nLoading GPT-2...")
    model = HCWSModel("gpt2", steering_strength=3.0, device="cpu")
    print("Model loaded!\n")
    
    steering_instruction = "be creative and imaginative"
    
    # Register instruction
    if model.needs_retraining([steering_instruction]):
        print(f"Registering instruction: '{steering_instruction}'")
        model.update_trained_instructions([steering_instruction])
        print("Ready!\n")
    
    prompt = "Once upon a time"
    
    print("="*60)
    print(f"Prompt: {prompt}")
    print("="*60)
    
    # Baseline
    print("\n[UNSTEERED]")
    try:
        baseline = model.generate(
            prompt,
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
        print(f"Output: {baseline}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Steered
    print("[STEERED]")
    try:
        steered = model.generate(
            prompt,
            steering_instruction=steering_instruction,
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
        print(f"Output: {steered}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
