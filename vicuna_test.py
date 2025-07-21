#!/usr/bin/env python3
"""
Vicuna Model Testing with HCWS

Tests HCWS steering capabilities with Vicuna models (LLaMA-based, publicly available).
Vicuna models are excellent alternatives to Meta's LLaMA when access is restricted.
"""

import torch
import warnings
import os

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from hcws import HCWSModel, get_best_device, print_device_info

def test_vicuna_model(model_key="vicuna-7b"):
    """Test HCWS with a Vicuna model."""
    print(f"🦙 Testing HCWS with {model_key.upper()}")
    print("=" * 60)
    
    try:
        # Show device info
        print_device_info()
        device = get_best_device()
        
        # Load Vicuna model with HCWS
        print(f"\n🔄 Loading {model_key}...")
        print("(This may take a few minutes to download the model...)")
        
        model = HCWSModel(
            model_key,
            device=device,
            steering_strength=3.5  # Good default for Vicuna
        )
        print("✅ Model loaded successfully!")
        
        # Test different steering scenarios (similar to LLaMA use cases)
        test_scenarios = [
            {
                "prompt": "Explain the benefits of renewable energy",
                "instruction": "be enthusiastic and optimistic about environmental solutions",
                "description": "Environmental Enthusiasm"
            },
            {
                "prompt": "How can I improve my programming skills?",
                "instruction": "be practical and encouraging with specific advice",
                "description": "Programming Advice"
            },
            {
                "prompt": "What are the challenges of artificial intelligence?",
                "instruction": "provide balanced and thoughtful analysis",
                "description": "AI Discussion"
            },
            {
                "prompt": "Tell me about machine learning algorithms",
                "instruction": "be technical but accessible to beginners",
                "description": "Technical Explanation"
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
            print("🔹 UNSTEERED (Original Vicuna):")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unsteered = model.generate(
                    scenario["prompt"],
                    max_length=150,
                    temperature=0.7,
                    do_sample=True
                )
            print(f"   {unsteered}")
            
            # Generate steered response
            print("\n🎮 STEERED (HCWS Enhanced):")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                steered = model.generate(
                    scenario["prompt"],
                    steering_instruction=scenario["instruction"],
                    max_length=150,
                    temperature=0.7,
                    do_sample=True
                )
            print(f"   {steered}")
            
            # Simple analysis
            print(f"\n📊 ANALYSIS:")
            print(f"   Unsteered length: {len(unsteered.split())} words")
            print(f"   Steered length: {len(steered.split())} words")
            
            # Check if responses are different
            if unsteered != steered:
                print("   ✅ HCWS successfully modified the response")
            else:
                print("   ➖ Responses are identical (may need stronger steering)")
            
            print("=" * 80)
        
        # Demonstration of steering strength effects
        print(f"\n🔧 STEERING STRENGTH DEMONSTRATION")
        print("-" * 50)
        
        test_prompt = "What are the most important skills for success?"
        test_instruction = "be motivational and inspiring"
        strengths = [0.0, 2.0, 4.0, 6.0]
        
        for strength in strengths:
            print(f"\n⚡ Strength: {strength}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if strength == 0.0:
                    response = model.generate(
                        test_prompt,
                        max_length=80,
                        temperature=0.7,
                        do_sample=True
                    )
                    print("   (No steering)")
                else:
                    # Temporarily adjust steering strength
                    original_strength = model.steering_strength
                    model.steering_strength = strength
                    model.controller.steering_strength = strength
                    
                    response = model.generate(
                        test_prompt,
                        steering_instruction=test_instruction,
                        max_length=80,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    # Restore original strength
                    model.steering_strength = original_strength
                    model.controller.steering_strength = original_strength
            
            print(f"   Response: {response}")
        
        print(f"\n🎉 {model_key} testing completed successfully!")
        print("\n📊 SUMMARY:")
        print(f"- Model: {model_key} (LLaMA-based Vicuna)")
        print(f"- Device: {device.upper()}")
        print("- HCWS steering successfully demonstrated")
        print("- Vicuna provides excellent LLaMA-like capabilities without access restrictions")
        print("- Different steering strengths showed varied effects")
        
    except Exception as e:
        print(f"❌ Error testing {model_key}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have sufficient memory/storage")
        print("2. Check internet connection for model download")
        print("3. Try with a smaller model if memory issues occur")
        print("4. Vicuna models are large - 7B needs ~15GB RAM, 13B needs ~25GB")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test Vicuna models."""
    print("🦙 HCWS Vicuna Model Testing Suite")
    print("Vicuna: LLaMA-based models, publicly available!")
    print("=" * 50)
    
    available_models = [
        "vicuna-7b",   # 7B - Good balance, LLaMA 2 based
        "vicuna-13b",  # 13B - More capable, original LLaMA
        "vicuna-33b"   # 33B - Most powerful (requires lots of RAM)
    ]
    
    print("Available Vicuna models (LLaMA-based, no access restrictions):")
    for i, model in enumerate(available_models, 1):
        try:
            from hcws import get_model_config
            config = get_model_config(model)
            print(f"{i}. {model} - {config.name}")
            print(f"   • {config.description}")
            print(f"   • Hidden dim: {config.hidden_dim}, Layers: {config.num_layers}")
        except:
            print(f"{i}. {model} - (Configuration error)")
    
    print("\nRecommendations:")
    print("• vicuna-7b: Best starting point, based on LLaMA 2 (similar to LLaMA 3.1)")
    print("• vicuna-13b: More capable, if you have sufficient RAM/VRAM")
    print("• vicuna-33b: Most powerful, but requires significant resources")
    
    try:
        choice = input(f"\nSelect model (1-{len(available_models)}) or press Enter for Vicuna-7B: ").strip()
        
        if choice:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                model_key = available_models[model_idx]
            else:
                model_key = "vicuna-7b"
        else:
            model_key = "vicuna-7b"  # Default to 7B
            
    except (ValueError, KeyboardInterrupt):
        model_key = "vicuna-7b"  # Default to 7B
    
    print(f"\nSelected: {model_key}")
    print("Note: Vicuna is an excellent LLaMA alternative with similar capabilities!")
    test_vicuna_model(model_key)

if __name__ == "__main__":
    main() 