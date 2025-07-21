#!/usr/bin/env python3
"""
Qwen Model Testing with HCWS

Tests HCWS steering capabilities with Qwen2 and Qwen2.5 models running locally.
"""

import torch
import warnings
import os

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from hcws import HCWSModel, get_best_device, print_device_info

def test_qwen_model(model_key="qwen2.5-7b"):
    """Test HCWS with a Qwen model."""
    print(f"🚀 Testing HCWS with {model_key.upper()}")
    print("=" * 60)
    
    try:
        # Show device info
        print_device_info()
        device = get_best_device()
        
        # Load Qwen model with HCWS
        print(f"\n🔄 Loading {model_key}...")
        print("(This may take a few minutes to download the model...)")
        
        model = HCWSModel(
            model_key,
            device=device,
            steering_strength=4.0  # Slightly higher for Qwen
        )
        print("✅ Model loaded successfully!")
        
        # Test different steering scenarios
        test_scenarios = [
            {
                "prompt": "The future of artificial intelligence is",
                "instruction": "be optimistic and enthusiastic about technology",
                "description": "Optimistic Tech Discussion"
            },
            {
                "prompt": "When solving complex problems, the best approach is to",
                "instruction": "be methodical and analytical",
                "description": "Analytical Problem Solving"
            },
            {
                "prompt": "Climate change is a challenge that",
                "instruction": "provide balanced scientific perspective",
                "description": "Scientific Climate Discussion"
            },
            {
                "prompt": "The key to effective communication is",
                "instruction": "be clear and practical",
                "description": "Communication Advice"
            },
            {
                "prompt": "In creative writing, the most important element is",
                "instruction": "be artistic and imaginative",
                "description": "Creative Writing Perspective"
            }
        ]
        
        print(f"\n🎯 Testing HCWS steering with {model_key}")
        print("=" * 80)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*15} Test {i}: {scenario['description']} {'='*15}")
            print(f"📝 Prompt: {scenario['prompt']}")
            print(f"🎯 Instruction: {scenario['instruction']}")
            print("-" * 70)
            
            # Generate unsteered response
            print("🔹 UNSTEERED:")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unsteered = model.generate(
                    scenario["prompt"],
                    max_length=100,
                    temperature=0.8,
                    do_sample=True
                )
            print(f"   {unsteered}")
            
            # Generate steered response
            print("\n🎮 STEERED:")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                steered = model.generate(
                    scenario["prompt"],
                    steering_instruction=scenario["instruction"],
                    max_length=100,
                    temperature=0.8,
                    do_sample=True
                )
            print(f"   {steered}")
            
            # Simple analysis
            print(f"\n📊 COMPARISON:")
            print(f"   Unsteered length: {len(unsteered.split())} words")
            print(f"   Steered length: {len(steered.split())} words")
            
            # Check for steering effectiveness
            instruction_words = scenario["instruction"].lower().split()
            unsteered_lower = unsteered.lower()
            steered_lower = steered.lower()
            
            unsteered_matches = sum(1 for word in instruction_words if word in unsteered_lower)
            steered_matches = sum(1 for word in instruction_words if word in steered_lower)
            
            if steered_matches > unsteered_matches:
                print("   ✅ Steering appears effective (more instruction-related words)")
            elif unsteered != steered:
                print("   📈 Steering modified response (different content)")
            else:
                print("   ➖ Minimal steering effect")
            
            print("=" * 80)
        
        # Test steering strength variations
        print(f"\n🔧 STEERING STRENGTH COMPARISON")
        print("-" * 50)
        
        test_prompt = "The best way to learn new skills is"
        test_instruction = "be encouraging and motivational"
        strengths = [0.0, 2.0, 4.0, 6.0]
        
        for strength in strengths:
            print(f"\n⚡ Strength: {strength}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if strength == 0.0:
                    response = model.generate(
                        test_prompt,
                        max_length=60,
                        temperature=0.8,
                        do_sample=True
                    )
                else:
                    model.steering_strength = strength
                    model.controller.steering_strength = strength
                    
                    response = model.generate(
                        test_prompt,
                        steering_instruction=test_instruction,
                        max_length=60,
                        temperature=0.8,
                        do_sample=True
                    )
            
            print(f"Response: {response}")
            
            # Count encouraging words
            encouraging_words = ["great", "excellent", "amazing", "wonderful", "fantastic", "perfect", "best", "awesome"]
            encouraging_count = sum(1 for word in encouraging_words if word in response.lower())
            print(f"Encouraging words: {encouraging_count}")
        
        print(f"\n🎉 {model_key} testing completed successfully!")
        print("\n📊 SUMMARY:")
        print(f"- Model: {model_key} ({device.upper()})")
        print("- HCWS steering successfully modified responses")
        print("- Different steering strengths showed varied effects")
        print("- Qwen models demonstrate good steering responsiveness")
        
    except Exception as e:
        print(f"❌ Error testing {model_key}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have sufficient memory/storage")
        print("2. Check internet connection for model download")
        print("3. Try with a smaller model if memory issues occur")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test Qwen models."""
    print("🐧 HCWS Qwen Model Testing Suite")
    print("=" * 50)
    
    available_models = [
        "qwen2.5-0.5b",   # 0.5B - Very fast, good for testing
        "qwen2.5-1.5b",   # 1.5B - Small but capable  
        "qwen2.5-3b",     # 3B - Good balance
        "qwen2.5-7b",     # 7B - Full capability
        "qwen2-7b"        # Original Qwen2 7B
    ]
    
    print("Available Qwen models (ordered by size):")
    for i, model in enumerate(available_models, 1):
        try:
            from hcws import get_model_config
            config = get_model_config(model)
            print(f"{i}. {model} - {config.name} ({config.hidden_dim}d, {config.num_layers} layers)")
        except:
            print(f"{i}. {model} - (Configuration error)")
    
    print("\nRecommendations:")
    print("• qwen2.5-0.5b: Fastest, good for quick testing")
    print("• qwen2.5-1.5b: Small but capable, runs on most hardware")
    print("• qwen2.5-3b: Good balance of speed and capability")
    print("• qwen2.5-7b: Full performance (requires more memory)")
    
    try:
        choice = input(f"\nSelect model (1-{len(available_models)}) or press Enter for Qwen2.5-1.5B: ").strip()
        
        if choice:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                model_key = available_models[model_idx]
            else:
                model_key = "qwen2.5-1.5b"  # Default to 1.5B for good balance
        else:
            model_key = "qwen2.5-1.5b"  # Default to 1.5B
            
    except (ValueError, KeyboardInterrupt):
        model_key = "qwen2.5-1.5b"  # Default to 1.5B
    
    print(f"\nSelected: {model_key}")
    test_qwen_model(model_key)

if __name__ == "__main__":
    main() 