#!/usr/bin/env python3
"""
Simple demonstration of HCWS functionality.

This script provides a quick way to test the HCWS system with basic examples.
"""

import torch
import logging
from hcws import HCWSModel

# Set up logging to only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_comparison(prompt, unsteered, steered, instruction):
    """Print a clean comparison between unsteered and steered outputs."""
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Instruction: {instruction}")
    print(f"\n🔹 Unsteered: {unsteered}")
    print(f"🎭 Steered:    {steered}")
    print("-" * 60)


def main():
    """Run a simple HCWS demonstration."""
    
    print("🚀 HCWS (Hyper-Conceptor Weighted Steering) Demo")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    try:
        # Initialize HCWS model
        print("\n🔄 Initializing HCWS model with GPT-2...")
        model = HCWSModel("gpt2", device=device)
        print("✅ Model loaded successfully")
        
        # Test basic generation
        print_section("BASIC GENERATION TEST")
        
        prompt = "The future of artificial intelligence is"
        
        # Generate without steering
        unsteered_output = model.generate(
            prompt,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        
        # Generate with steering
        steering_instruction = "be optimistic and enthusiastic"
        steered_output = model.generate(
            prompt,
            steering_instruction=steering_instruction,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        
        print_comparison(prompt, unsteered_output, steered_output, steering_instruction)
        
        # Test different steering instructions
        print_section("MULTIPLE STEERING INSTRUCTIONS")
        
        test_prompt = "The weather today is"
        instructions = [
            "be poetic and metaphorical",
            "be scientific and precise", 
            "be negative and gloomy",
            "be cheerful and upbeat"
        ]
        
        # Generate unsteered version first
        unsteered_output = model.generate(
            test_prompt,
            max_length=25,
            temperature=0.7
        )
        
        print(f"📝 Prompt: {test_prompt}")
        print(f"🔹 Unsteered: {unsteered_output}")
        print("-" * 60)
        
        # Generate each steered version
        for i, instruction in enumerate(instructions, 1):
            steered_output = model.generate(
                test_prompt,
                steering_instruction=instruction,
                max_length=25,
                temperature=0.7
            )
            print(f"{i}. 🎯 '{instruction}':")
            print(f"   🎭 {steered_output}")
            print()
        
        # Test steering strength analysis
        print_section("STEERING STRENGTH ANALYSIS")
        
        for instruction in instructions:
            metrics = model.compute_steering_strength(instruction)
            print(f"🎯 '{instruction}':")
            print(f"   📊 Mean aperture: {metrics['mean_aperture']:.4f}")
            print(f"   📈 Aperture std:   {metrics['aperture_std']:.4f}")
            print()
        
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        print(f"❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 