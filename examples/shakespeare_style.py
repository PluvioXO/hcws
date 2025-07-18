#!/usr/bin/env python3
"""
Example: Shakespearean Style Generation with HCWS

This script demonstrates how to use HCWS to generate text in Shakespearean style.
"""

import torch
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hcws import HCWSModel, print_available_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate Shakespearean style generation."""
    
    parser = argparse.ArgumentParser(description="Shakespearean Style Generation with HCWS")
    parser.add_argument("--model", "-m", default="gpt2", help="Model to use")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        print_available_models()
        return
    
    # Initialize HCWS model
    print(f"Loading HCWS model with {args.model}...")
    model = HCWSModel(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Define input prompts
    prompts = [
        "The weather today is",
        "I think that artificial intelligence will",
        "In the future, humans and robots will",
        "The most important thing in life is",
        "Technology has changed our world by"
    ]
    
    # Define Shakespearean instructions
    shakespeare_instructions = [
        "answer in Shakespearean English",
        "respond like William Shakespeare",
        "use Elizabethan language and style",
        "speak with the eloquence of Shakespeare",
        "write in the style of Romeo and Juliet"
    ]
    
    print("\n" + "="*60)
    print("SHAKESPEAREAN STYLE GENERATION EXAMPLES")
    print("="*60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {prompt}")
        print()
        
        # Generate normal response
        normal_response = model.generate(
            prompt,
            max_length=50,
            temperature=0.8,
            do_sample=True
        )
        print(f"Normal: {normal_response}")
        
        # Generate with Shakespearean steering
        for j, instruction in enumerate(shakespeare_instructions[:2]):  # Use first 2 instructions
            shakespeare_response = model.generate(
                prompt,
                steering_instruction=instruction,
                max_length=50,
                temperature=0.8,
                do_sample=True
            )
            print(f"Shakespeare {j+1}: {shakespeare_response}")
        
        print()
    
    # Demonstrate multiple instructions comparison
    print("\n" + "="*60)
    print("COMPARISON OF DIFFERENT SHAKESPEARE INSTRUCTIONS")
    print("="*60)
    
    test_prompt = "What is love?"
    print(f"Input: {test_prompt}")
    print()
    
    # Generate with all Shakespeare instructions
    responses = model.generate_with_multiple_instructions(
        test_prompt,
        shakespeare_instructions,
        max_length=60,
        temperature=0.9
    )
    
    for instruction, response in zip(shakespeare_instructions, responses):
        print(f"'{instruction}': {response}")
        print()
    
    # Analyze steering strength
    print("\n" + "="*60)
    print("STEERING STRENGTH ANALYSIS")
    print("="*60)
    
    for instruction in shakespeare_instructions:
        metrics = model.compute_steering_strength(instruction)
        print(f"Instruction: '{instruction}'")
        print(f"  Mean aperture: {metrics['mean_aperture']:.4f}")
        print(f"  Max aperture: {metrics['max_aperture']:.4f}")
        print(f"  Min aperture: {metrics['min_aperture']:.4f}")
        print(f"  Aperture std: {metrics['aperture_std']:.4f}")
        print()


if __name__ == "__main__":
    main() 