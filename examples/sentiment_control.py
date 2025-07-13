#!/usr/bin/env python3
"""
Example: Sentiment Control with HCWS

This script demonstrates how to use HCWS to control the sentiment of generated text.
"""

import torch
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hcws import HCWSModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate sentiment control."""
    
    # Initialize HCWS model with GPT-2
    print("Loading HCWS model with GPT-2...")
    model = HCWSModel("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Define neutral prompts
    neutral_prompts = [
        "The movie was",
        "I went to the restaurant and",
        "The weather forecast shows",
        "My experience with the product was",
        "The political situation seems",
        "The book I read was",
        "Working from home has been"
    ]
    
    # Define sentiment instructions
    sentiment_instructions = {
        "positive": [
            "respond with positive sentiment",
            "be optimistic and upbeat",
            "focus on the bright side",
            "emphasize positive aspects"
        ],
        "negative": [
            "respond with negative sentiment", 
            "be pessimistic and critical",
            "focus on problems and issues",
            "emphasize negative aspects"
        ],
        "neutral": [
            "respond with neutral sentiment",
            "be balanced and objective",
            "present facts without bias",
            "maintain neutrality"
        ]
    }
    
    print("\n" + "="*70)
    print("SENTIMENT CONTROL EXAMPLES")
    print("="*70)
    
    for i, prompt in enumerate(neutral_prompts):
        print(f"\n--- Example {i+1}: '{prompt}' ---")
        print()
        
        # Generate with different sentiments
        for sentiment, instructions in sentiment_instructions.items():
            instruction = instructions[0]  # Use first instruction for each sentiment
            
            response = model.generate(
                prompt,
                steering_instruction=instruction,
                max_length=40,
                temperature=0.8,
                do_sample=True
            )
            
            print(f"{sentiment.capitalize()}: {response}")
        
        print()
    
    # Demonstrate sentiment strength comparison
    print("\n" + "="*70)
    print("SENTIMENT STRENGTH COMPARISON")
    print("="*70)
    
    test_prompt = "The new policy will"
    print(f"Input: {test_prompt}")
    print()
    
    # Test different levels of sentiment intensity
    positive_intensity = [
        "be slightly positive",
        "be moderately positive", 
        "be very positive",
        "be extremely positive and enthusiastic"
    ]
    
    negative_intensity = [
        "be slightly negative",
        "be moderately negative",
        "be very negative", 
        "be extremely negative and critical"
    ]
    
    print("Positive intensity levels:")
    for i, instruction in enumerate(positive_intensity):
        response = model.generate(
            test_prompt,
            steering_instruction=instruction,
            max_length=35,
            temperature=0.7
        )
        print(f"  Level {i+1}: {response}")
    
    print("\nNegative intensity levels:")
    for i, instruction in enumerate(negative_intensity):
        response = model.generate(
            test_prompt,
            steering_instruction=instruction,
            max_length=35,
            temperature=0.7
        )
        print(f"  Level {i+1}: {response}")
    
    # Analyze different sentiment instruction formulations
    print("\n" + "="*70)
    print("SENTIMENT INSTRUCTION FORMULATIONS")
    print("="*70)
    
    test_prompt = "The company's performance this year"
    print(f"Input: {test_prompt}")
    print()
    
    # Different ways to express positive sentiment
    positive_formulations = [
        "be positive",
        "respond optimistically",
        "emphasize good news",
        "highlight successes",
        "find the silver lining",
        "celebrate achievements"
    ]
    
    print("Different positive formulations:")
    for formulation in positive_formulations:
        response = model.generate(
            test_prompt,
            steering_instruction=formulation,
            max_length=40,
            temperature=0.8
        )
        print(f"  '{formulation}': {response}")
    
    # Steering strength analysis
    print("\n" + "="*70)
    print("STEERING STRENGTH ANALYSIS")
    print("="*70)
    
    all_instructions = []
    for instructions in sentiment_instructions.values():
        all_instructions.extend(instructions)
    
    print("Analyzing steering strength for sentiment instructions...")
    for instruction in all_instructions:
        metrics = model.compute_steering_strength(instruction)
        print(f"'{instruction}':")
        print(f"  Mean aperture: {metrics['mean_aperture']:.4f}")
        print(f"  Aperture range: [{metrics['min_aperture']:.4f}, {metrics['max_aperture']:.4f}]")
        print()


if __name__ == "__main__":
    main() 