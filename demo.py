#!/usr/bin/env python3
"""
Simple demonstration of HCWS functionality.

This script provides a quick way to test the HCWS system with basic examples.
"""

import torch
import logging
from hcws import HCWSModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple HCWS demonstration."""
    
    print("HCWS (Hyper-Conceptor Weighted Steering) Demo")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Initialize HCWS model
        print("\nInitializing HCWS model with GPT-2...")
        model = HCWSModel("gpt2", device=device)
        print("✓ Model loaded successfully")
        
        # Test basic generation
        print("\n" + "=" * 50)
        print("BASIC GENERATION TEST")
        print("=" * 50)
        
        prompt = "The future of artificial intelligence is"
        print(f"Input: {prompt}")
        
        # Generate without steering
        normal_output = model.generate(
            prompt,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        print(f"Normal: {normal_output}")
        
        # Generate with steering
        steering_instruction = "be optimistic and enthusiastic"
        steered_output = model.generate(
            prompt,
            steering_instruction=steering_instruction,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        print(f"Steered ('{steering_instruction}'): {steered_output}")
        
        # Test different steering instructions
        print("\n" + "=" * 50)
        print("MULTIPLE STEERING INSTRUCTIONS")
        print("=" * 50)
        
        test_prompt = "The weather today is"
        instructions = [
            "be poetic and metaphorical",
            "be scientific and precise",
            "be negative and gloomy",
            "be cheerful and upbeat"
        ]
        
        print(f"Input: {test_prompt}")
        print()
        
        for instruction in instructions:
            output = model.generate(
                test_prompt,
                steering_instruction=instruction,
                max_length=25,
                temperature=0.7
            )
            print(f"'{instruction}': {output}")
        
        # Test steering strength analysis
        print("\n" + "=" * 50)
        print("STEERING STRENGTH ANALYSIS")
        print("=" * 50)
        
        for instruction in instructions:
            metrics = model.compute_steering_strength(instruction)
            print(f"'{instruction}':")
            print(f"  Mean aperture: {metrics['mean_aperture']:.4f}")
            print(f"  Aperture std: {metrics['aperture_std']:.4f}")
            print()
        
        print("✓ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        print(f"❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 