#!/usr/bin/env python3
"""
Model Selection Demo for HCWS

This script demonstrates the new model selection capabilities, showing how to:
1. List available models
2. Get model information
3. Use different models including DeepSeek V3
4. Compare outputs from different models
"""

import torch
import argparse
from hcws import HCWSModel, print_available_models, get_model_config

def demo_model_info():
    """Demonstrate model information features."""
    print("=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    print_available_models()
    
    print("\n" + "=" * 60)
    print("DEEPSEEK V3 MODEL INFO")
    print("=" * 60)
    
    try:
        config = get_model_config("deepseek-v3")
        print(f"Model: {config.name}")
        print(f"ID: {config.model_id}")
        print(f"Architecture: {config.architecture}")
        print(f"Hidden Dim: {config.hidden_dim}")
        print(f"Layers: {config.num_layers}")
        print(f"Default Steering Strength: {config.default_steering_strength}")
        print(f"Requires Trust Remote Code: {config.requires_trust_remote_code}")
        print(f"Recommended Dtype: {config.torch_dtype}")
        print(f"Description: {config.description}")
    except Exception as e:
        print(f"Error getting model info: {e}")


def demo_model_generation(model_key: str, prompt: str = "The future of AI is"):
    """Demonstrate text generation with a specific model."""
    print(f"\n" + "=" * 60)
    print(f"GENERATION WITH {model_key.upper()}")
    print("=" * 60)
    
    try:
        # Get model config
        config = get_model_config(model_key)
        print(f"Using: {config.name}")
        print(f"Model ID: {config.model_id}")
        print(f"Steering Strength: {config.default_steering_strength}")
        
        # Initialize model
        print("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HCWSModel(model_key, device=device)
        print("✅ Model loaded successfully")
        
        # Test basic generation
        print(f"\nPrompt: {prompt}")
        
        unsteered = model.generate(
            prompt,
            max_length=50,
            temperature=0.8,
            do_sample=True
        )
        print(f"Unsteered: {unsteered}")
        
        # Test steered generation
        instruction = "be optimistic and enthusiastic about technology"
        steered = model.generate(
            prompt,
            steering_instruction=instruction,
            max_length=50,
            temperature=0.8,
            do_sample=True
        )
        print(f"Steered ({instruction}): {steered}")
        
    except Exception as e:
        print(f"Error with model {model_key}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="HCWS Model Selection Demo")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2",
        help="Model to test (use 'deepseek-v3' for DeepSeek V3)"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="Show model information and exit"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for text generation"
    )
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare multiple models (GPT-2 and specified model)"
    )
    
    args = parser.parse_args()
    
    print("🚀 HCWS Model Selection Demo")
    print("=" * 60)
    
    if args.list_models:
        demo_model_info()
        return
    
    if args.compare:
        print("Comparing models...")
        # Test with GPT-2 first
        demo_model_generation("gpt2", args.prompt)
        
        # Test with specified model if different
        if args.model != "gpt2":
            demo_model_generation(args.model, args.prompt)
    else:
        demo_model_generation(args.model, args.prompt)
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("# List all available models:")
    print("python model_selection_demo.py --list-models")
    print()
    print("# Test with DeepSeek V3:")
    print("python model_selection_demo.py --model deepseek-v3")
    print()
    print("# Compare GPT-2 vs DeepSeek V3:")
    print("python model_selection_demo.py --model deepseek-v3 --compare")
    print()
    print("# Use custom prompt:")
    print('python model_selection_demo.py --model deepseek-v3 --prompt "Write a poem about"')
    print()
    print("# Use demo.py with DeepSeek V3:")
    print("python demo.py --model deepseek-v3 --steering-strength 2.0")


if __name__ == "__main__":
    main() 