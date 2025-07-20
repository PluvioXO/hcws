#!/usr/bin/env python3
"""
Example: Balancing Factual vs Creative Responses with HCWS

This script demonstrates how to use HCWS to balance between factual accuracy
and creative expression in generated text.
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
    """Main function to demonstrate factual vs creative balance."""
    
    # Initialize HCWS model with GPT-2
    print("Loading HCWS model with GPT-2...")
    from hcws.device_utils import get_best_device
    model = HCWSModel("gpt2", device=get_best_device())
    
    # Define prompts that can be answered factually or creatively
    prompts = [
        "The sun is",
        "Water boils at",
        "The capital of France is",
        "Photosynthesis is the process by which",
        "Gravity works by",
        "The human brain contains",
        "DNA stands for"
    ]
    
    # Define factual vs creative instructions
    factual_instructions = [
        "respond with factual accuracy",
        "provide precise scientific information",
        "be objective and informative",
        "stick to established facts",
        "answer like a textbook"
    ]
    
    creative_instructions = [
        "respond creatively and imaginatively",
        "use poetic and metaphorical language",
        "be artistic and expressive",
        "answer like a creative writer",
        "use colorful and vivid descriptions"
    ]
    
    balanced_instructions = [
        "balance facts with creativity",
        "be informative but engaging",
        "combine accuracy with imagination",
        "make facts interesting and memorable",
        "educate while entertaining"
    ]
    
    print("\n" + "="*70)
    print("FACTUAL VS CREATIVE BALANCE EXAMPLES")
    print("="*70)
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Example {i+1}: '{prompt}' ---")
        print()
        
        # Generate factual response
        factual_response = model.generate(
            prompt,
            steering_instruction=factual_instructions[0],
            max_length=50,
            temperature=0.5,
            do_sample=True
        )
        print(f"Factual: {factual_response}")
        
        # Generate creative response
        creative_response = model.generate(
            prompt,
            steering_instruction=creative_instructions[0],
            max_length=50,
            temperature=0.9,
            do_sample=True
        )
        print(f"Creative: {creative_response}")
        
        # Generate balanced response
        balanced_response = model.generate(
            prompt,
            steering_instruction=balanced_instructions[0],
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
        print(f"Balanced: {balanced_response}")
        
        print()
    
    # Demonstrate creativity spectrum
    print("\n" + "="*70)
    print("CREATIVITY SPECTRUM")
    print("="*70)
    
    test_prompt = "The ocean is"
    print(f"Input: {test_prompt}")
    print()
    
    creativity_levels = [
        "answer with pure scientific facts",
        "be mostly factual with some description",
        "balance facts and creative description",
        "be mostly creative with some facts",
        "answer with pure creative imagination"
    ]
    
    temperatures = [0.3, 0.5, 0.7, 0.9, 1.2]
    
    for i, (instruction, temp) in enumerate(zip(creativity_levels, temperatures)):
        response = model.generate(
            test_prompt,
            steering_instruction=instruction,
            max_length=45,
            temperature=temp,
            do_sample=True
        )
        print(f"Level {i+1}: {response}")
    
    # Demonstrate domain-specific balance
    print("\n" + "="*70)
    print("DOMAIN-SPECIFIC FACTUAL-CREATIVE BALANCE")
    print("="*70)
    
    domains = {
        "science": [
            "Explain black holes",
            "be scientifically accurate but engaging",
            "balance physics with wonder"
        ],
        "history": [
            "Describe the Renaissance",
            "be historically accurate but vivid",
            "bring history to life creatively"
        ],
        "technology": [
            "Explain artificial intelligence",
            "be technically correct but accessible",
            "make complex concepts engaging"
        ],
        "nature": [
            "Describe a forest ecosystem",
            "be biologically accurate but poetic",
            "combine science with natural beauty"
        ]
    }
    
    for domain, (prompt, instruction, description) in domains.items():
        print(f"\n{domain.upper()}: {prompt}")
        print(f"Instruction: {instruction}")
        
        response = model.generate(
            prompt,
            steering_instruction=instruction,
            max_length=60,
            temperature=0.7,
            do_sample=True
        )
        print(f"Response: {response}")
    
    # Compare different formulations
    print("\n" + "="*70)
    print("INSTRUCTION FORMULATION COMPARISON")
    print("="*70)
    
    test_prompt = "Evolution is"
    print(f"Input: {test_prompt}")
    print()
    
    factual_formulations = [
        "explain scientifically",
        "provide factual information",
        "be precise and accurate",
        "answer like a scientist",
        "use technical language"
    ]
    
    creative_formulations = [
        "explain poetically",
        "use creative metaphors",
        "be imaginative and artistic",
        "answer like a storyteller",
        "use vivid imagery"
    ]
    
    print("Factual formulations:")
    for formulation in factual_formulations:
        response = model.generate(
            test_prompt,
            steering_instruction=formulation,
            max_length=40,
            temperature=0.6
        )
        print(f"  '{formulation}': {response}")
    
    print("\nCreative formulations:")
    for formulation in creative_formulations:
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
    
    all_instructions = factual_instructions + creative_instructions + balanced_instructions
    
    print("Analyzing steering strength for different instruction types...")
    for instruction in all_instructions:
        metrics = model.compute_steering_strength(instruction)
        print(f"'{instruction}':")
        print(f"  Mean aperture: {metrics['mean_aperture']:.4f}")
        print(f"  Aperture std: {metrics['aperture_std']:.4f}")
        print()


if __name__ == "__main__":
    main() 