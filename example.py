#!/usr/bin/env python3
"""
HCWS Quick Start Example

This is the simplest, most straightforward example of using HCWS to steer 
language model generation. Perfect for getting started!

Usage:
    python example.py
"""

from hcws import HCWSModel, print_device_info


def main():
    """Simple demonstration of HCWS steering capabilities."""
    
    print("HCWS Quick Start Example")
    print("=" * 60)
    
    # Show what device we're using
    print("\nDevice Information:")
    print_device_info()
    
    # Initialize HCWS with GPT-2 (small, fast, works on CPU)
    print("\nLoading model...")
    model = HCWSModel("gpt2", steering_strength=5.0)
    print("Model loaded successfully!")
    
    # Example 1: Basic Steering
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Steering")
    print("="*60)
    
    prompt = "The future of artificial intelligence is"
    
    print(f"\nPrompt: {prompt}")
    
    # Generate without steering
    print("\nWithout steering:")
    unsteered = model.generate(
        prompt,
        max_length=50,
        temperature=0.8
    )
    print(f"   {unsteered}")
    
    # Generate with optimistic steering
    print("\nWith steering (be optimistic and enthusiastic):")
    steered = model.generate(
        prompt,
        steering_instruction="be optimistic and enthusiastic",
        max_length=50,
        temperature=0.8
    )
    print(f"   {steered}")
    
    # Example 2: Different Steering Instructions
    print("\n" + "="*60)
    print("EXAMPLE 2: Different Steering Styles")
    print("="*60)
    
    test_cases = [
        {
            "prompt": "The weather today is",
            "instruction": "be poetic and descriptive",
            "description": "Poetic Style"
        },
        {
            "prompt": "Machine learning works by",
            "instruction": "explain simply for a beginner",
            "description": "Simple Explanation"
        },
        {
            "prompt": "The best way to learn is",
            "instruction": "be encouraging and supportive",
            "description": "Encouraging Tone"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   Prompt: {case['prompt']}")
        print(f"   Instruction: {case['instruction']}")
        
        response = model.generate(
            case["prompt"],
            steering_instruction=case["instruction"],
            max_length=40,
            temperature=0.7
        )
        print(f"   Result: {response}")
    
    # Example 3: Steering Strength Comparison
    print("\n" + "="*60)
    print("EXAMPLE 3: Steering Strength Levels")
    print("="*60)
    
    prompt = "Technology has changed our lives by"
    instruction = "be enthusiastic and positive"
    
    print(f"\nPrompt: {prompt}")
    print(f"Instruction: {instruction}\n")
    
    strengths = [0.0, 2.0, 5.0, 8.0]
    
    for strength in strengths:
        model.steering_strength = strength
        
        if strength == 0.0:
            response = model.generate(prompt, max_length=35, temperature=0.7)
            print(f"Strength {strength} (no steering): {response}")
        else:
            response = model.generate(
                prompt,
                steering_instruction=instruction,
                max_length=35,
                temperature=0.7
            )
            print(f"Strength {strength}: {response}")
    
    # Summary
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - HCWS steers model outputs without retraining")
    print("  - Use clear, descriptive steering instructions")
    print("  - Adjust steering_strength for intensity (3-5 is typical)")
    print("  - Works with any transformer model (GPT-2, LLaMA, etc.)")
    print("\nNext Steps:")
    print("  - Try your own prompts and instructions")
    print("  - Explore examples/ folder for more use cases")
    print("  - Run 'python test.py --model gpt2' for comprehensive tests")
    print("  - Check README.md for advanced usage")


if __name__ == "__main__":
    main()
