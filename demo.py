#!/usr/bin/env python3
"""
HCWS Demo Script

This script demonstrates the Hyper-Conceptor Weighted Steering (HCWS) method
for controlling language model behavior during inference.
"""

import torch
import logging
from hcws import HCWSModel, ActAddModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*20} {title} {'='*20}")


def print_comparison(prompt: str, unsteered: str, steered: str, instruction: str, strength: float):
    """Print a formatted comparison between unsteered and steered outputs."""
    print(f"📝 Prompt: {prompt}")
    print(f"🎯 Instruction: {instruction}")
    print(f"⚡ Strength: {strength}")
    print(f"🔹 Unsteered: {unsteered}")
    print(f"🎮 Steered: {steered}")
    print("-" * 60)


def print_method_comparison(prompt: str, hcws_output: str, actadd_output: str, instruction: str, behavior: str):
    """Print a formatted comparison between HCWS and ActAdd methods."""
    print(f"📝 Prompt: {prompt}")
    print(f"🎯 Instruction: {instruction}")
    print(f"🎭 Behavior: {behavior}")
    print(f"🧠 HCWS: {hcws_output}")
    print(f"➕ ActAdd: {actadd_output}")
    print("-" * 60)


def main():
    """Run a comprehensive HCWS and ActAdd demonstration."""
    
    print("🚀 HCWS vs ActAdd Steering Demo")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")
    
    try:
        # Initialize both models
        print("\n🔄 Initializing models...")
        hcws_model = HCWSModel("gpt2", device=device, steering_strength=5.0)
        actadd_model = ActAddModel("gpt2", device=device, steering_strength=5.0)
        print("✅ Models loaded successfully")
        
        # Test basic generation comparison
        print_section("BASIC GENERATION COMPARISON")
        
        prompt = "The future of artificial intelligence is"
        instruction = "be optimistic and enthusiastic"
        behavior = "optimistic"
        
        # Generate unsteered version
        unsteered_output = hcws_model.generate(
            prompt,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        
        # Generate with HCWS
        hcws_output = hcws_model.generate(
            prompt,
            steering_instruction=instruction,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        
        # Generate with ActAdd
        actadd_output = actadd_model.generate(
            prompt,
            behavior=behavior,
            max_length=30,
            temperature=0.8,
            do_sample=True
        )
        
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Instruction: {instruction}")
        print(f"🔹 Unsteered: {unsteered_output}")
        print(f"🧠 HCWS: {hcws_output}")
        print(f"➕ ActAdd: {actadd_output}")
        print("-" * 60)
        
        # Test different behaviors/styles
        print_section("MULTIPLE BEHAVIOR COMPARISON")
        
        test_cases = [
            {
                "prompt": "The weather today is",
                "instruction": "be cheerful and upbeat",
                "behavior": "optimistic"
            },
            {
                "prompt": "The economic situation is",
                "instruction": "be pessimistic and gloomy",
                "behavior": "pessimistic"
            },
            {
                "prompt": "The scientific discovery shows",
                "instruction": "be formal and precise",
                "behavior": "formal"
            },
            {
                "prompt": "The party was",
                "instruction": "be casual and relaxed",
                "behavior": "casual"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔸 Test Case {i}:")
            
            # Generate with HCWS
            hcws_output = hcws_model.generate(
                case["prompt"],
                steering_instruction=case["instruction"],
                max_length=25,
                temperature=0.7
            )
            
            # Generate with ActAdd
            actadd_output = actadd_model.generate(
                case["prompt"],
                behavior=case["behavior"],
                max_length=25,
                temperature=0.7
            )
            
            print_method_comparison(
                case["prompt"],
                hcws_output,
                actadd_output,
                case["instruction"],
                case["behavior"]
            )
        
        # Test steering strength comparison
        print_section("STEERING STRENGTH COMPARISON")
        
        test_prompt = "The new technology will"
        instruction = "be creative and innovative"
        behavior = "creative"
        
        # Generate unsteered version
        unsteered_output = hcws_model.generate(
            test_prompt,
            max_length=25,
            temperature=0.7
        )
        
        print(f"📝 Prompt: {test_prompt}")
        print(f"🎯 Instruction: {instruction}")
        print(f"🔹 Unsteered: {unsteered_output}")
        print("-" * 60)
        
        # Test different steering strengths
        strengths = [1.0, 3.0, 5.0, 8.0]
        
        for strength in strengths:
            # Update steering strength for both models
            hcws_model.steering_strength = strength
            actadd_model.steering_strength = strength
            
            # Generate with HCWS
            hcws_output = hcws_model.generate(
                test_prompt,
                steering_instruction=instruction,
                max_length=25,
                temperature=0.7
            )
            
            # Generate with ActAdd
            actadd_output = actadd_model.generate(
                test_prompt,
                behavior=behavior,
                max_length=25,
                temperature=0.7
            )
            
            print(f"⚡ Strength: {strength}")
            print(f"🧠 HCWS: {hcws_output}")
            print(f"➕ ActAdd: {actadd_output}")
            print("-" * 40)
        
        # Test available ActAdd behaviors
        print_section("AVAILABLE ACTADD BEHAVIORS")
        
        available_behaviors = actadd_model.get_available_behaviors()
        print(f"🎭 Available behaviors: {', '.join(available_behaviors)}")
        
        # Test a few more behaviors
        additional_tests = [
            {
                "prompt": "The poem describes",
                "instruction": "be poetic and metaphorical",
                "behavior": "poetic"
            },
            {
                "prompt": "The research indicates",
                "instruction": "be scientific and analytical",
                "behavior": "scientific"
            }
        ]
        
        for case in additional_tests:
            print(f"\n🔸 Testing {case['behavior']} behavior:")
            
            hcws_output = hcws_model.generate(
                case["prompt"],
                steering_instruction=case["instruction"],
                max_length=25,
                temperature=0.7
            )
            
            actadd_output = actadd_model.generate(
                case["prompt"],
                behavior=case["behavior"],
                max_length=25,
                temperature=0.7
            )
            
            print_method_comparison(
                case["prompt"],
                hcws_output,
                actadd_output,
                case["instruction"],
                case["behavior"]
            )
        
        print_section("DEMO COMPLETE")
        print("✅ Successfully demonstrated both HCWS and ActAdd steering methods!")
        print("📊 Key differences:")
        print("   • HCWS: Uses conceptor-based subspace steering")
        print("   • ActAdd: Uses direct activation vector addition")
        print("   • Both methods provide effective steering control")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 