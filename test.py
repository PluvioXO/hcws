#!/usr/bin/env python3
"""
HCWS Unified Model Testing

A clean, JSON-configured testing interface for HCWS that allows users to:
1. Select models from organized categories via JSON configuration
2. Choose test scenarios interactively
3. Compare steering effects with different strengths
4. Get detailed performance analysis

Usage:
    python test.py                    # Interactive mode
    python test.py --model gpt2       # Direct model selection
    python test.py --list-models      # Show all available models
    python test.py --scenario basic   # Run specific test scenario
"""

import json
import argparse
import warnings
import os
from typing import Dict, List, Optional, Any

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from hcws import HCWSModel, get_best_device, print_device_info, get_model_config


def load_models_config() -> Dict[str, Any]:
    """Load the models configuration from JSON file."""
    try:
        with open('models.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: models.json not found in current directory")
        print("Please run this script from the project root directory.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing models.json: {e}")
        exit(1)


def print_available_models(config: Dict[str, Any]):
    """Print all available models organized by category."""
    print("🚀 HCWS Supported Models")
    print("=" * 60)
    
    for category, category_data in config['models'].items():
        print(f"\n📦 {category.upper().replace('_', ' ')}")
        print(f"   {category_data['description']}")
        print("-" * 50)
        
        for model_key, model_info in category_data['models'].items():
            print(f"  🔹 {model_key:15} - {model_info['name']} ({model_info['size']})")
            print(f"     {model_info['description']}")
            
            if 'requires' in model_info:
                req_str = ', '.join(model_info['requires'])
                print(f"     ⚠️  Requires: {req_str}")
            
            if 'recommended_for' in model_info:
                rec_str = ', '.join(model_info['recommended_for'])
                print(f"     💡 Best for: {rec_str}")
            print()


def get_all_model_keys(config: Dict[str, Any]) -> List[str]:
    """Get list of all available model keys."""
    model_keys = []
    for category_data in config['models'].values():
        model_keys.extend(category_data['models'].keys())
    return model_keys


def print_test_scenarios(config: Dict[str, Any]):
    """Print available test scenarios."""
    print("🎯 Available Test Scenarios")
    print("=" * 40)
    
    for scenario_key, scenario_data in config['test_scenarios'].items():
        print(f"\n📋 {scenario_key}")
        print(f"   {scenario_data['name']}")
        print(f"   {scenario_data['description']}")
        print(f"   Tests: {len(scenario_data['prompts'])} prompts")


def select_model_interactive(config: Dict[str, Any]) -> str:
    """Interactive model selection."""
    print("\n🔍 Model Selection")
    print("=" * 30)
    
    # Show categories
    categories = list(config['models'].keys())
    print("Available categories:")
    for i, category in enumerate(categories, 1):
        category_data = config['models'][category]
        model_count = len(category_data['models'])
        print(f"  {i}. {category.replace('_', ' ').title()} ({model_count} models)")
        print(f"     {category_data['description']}")
    
    try:
        choice = input(f"\nSelect category (1-{len(categories)}) or 'all' to see all models: ").strip()
        
        if choice.lower() == 'all':
            print_available_models(config)
            model_key = input("\nEnter model key directly: ").strip()
            return model_key
        
        category_idx = int(choice) - 1
        if 0 <= category_idx < len(categories):
            selected_category = categories[category_idx]
            category_data = config['models'][selected_category]
            
            print(f"\n📦 {selected_category.replace('_', ' ').title()} Models:")
            models = list(category_data['models'].items())
            
            for i, (model_key, model_info) in enumerate(models, 1):
                print(f"  {i}. {model_key} - {model_info['name']} ({model_info['size']})")
                print(f"     {model_info['description']}")
                if 'requires' in model_info:
                    req_str = ', '.join(model_info['requires'])
                    print(f"     ⚠️  Requires: {req_str}")
            
            model_choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            model_idx = int(model_choice) - 1
            
            if 0 <= model_idx < len(models):
                return models[model_idx][0]
            else:
                print("Invalid selection, using default (gpt2)")
                return "gpt2"
        else:
            print("Invalid selection, using default (gpt2)")
            return "gpt2"
    
    except (ValueError, KeyboardInterrupt):
        print("Using default model: gpt2")
        return "gpt2"


def select_scenario_interactive(config: Dict[str, Any]) -> str:
    """Interactive scenario selection."""
    print("\n🎯 Test Scenario Selection")
    print("=" * 35)
    
    scenarios = list(config['test_scenarios'].items())
    
    for i, (scenario_key, scenario_data) in enumerate(scenarios, 1):
        print(f"  {i}. {scenario_data['name']}")
        print(f"     {scenario_data['description']}")
        print(f"     {len(scenario_data['prompts'])} test prompts")
    
    try:
        choice = input(f"\nSelect scenario (1-{len(scenarios)}) or press Enter for all: ").strip()
        
        if not choice:
            return "all"
        
        scenario_idx = int(choice) - 1
        if 0 <= scenario_idx < len(scenarios):
            return scenarios[scenario_idx][0]
        else:
            print("Invalid selection, running all scenarios")
            return "all"
    
    except (ValueError, KeyboardInterrupt):
        print("Running all scenarios")
        return "all"


def run_test_scenario(model: HCWSModel, scenario_key: str, scenario_data: Dict[str, Any], 
                     config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a specific test scenario."""
    print(f"\n🧪 Running: {scenario_data['name']}")
    print("=" * 60)
    
    results = {
        'scenario': scenario_key,
        'name': scenario_data['name'], 
        'tests': []
    }
    
    settings = config['default_settings']
    
    for i, test_prompt in enumerate(scenario_data['prompts'], 1):
        print(f"\n{'='*15} Test {i}: {test_prompt['description']} {'='*15}")
        print(f"📝 Prompt: {test_prompt['prompt']}")
        print(f"🎯 Steering: {test_prompt['instruction']}")
        print("-" * 70)
        
        test_result = {
            'description': test_prompt['description'],
            'prompt': test_prompt['prompt'],
            'instruction': test_prompt['instruction'],
            'unsteered': '',
            'steered': '',
            'analysis': {}
        }
        
        # Generate unsteered response
        print("🔹 UNSTEERED (Original Model):")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unsteered = model.generate(
                test_prompt["prompt"],
                max_length=settings['max_length'],
                temperature=settings['temperature'],
                do_sample=settings['do_sample']
            )
        print(f"   {unsteered}")
        test_result['unsteered'] = unsteered
        
        # Generate steered response
        print("\n🎮 STEERED (HCWS Enhanced):")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            steered = model.generate(
                test_prompt["prompt"],
                steering_instruction=test_prompt["instruction"],
                max_length=settings['max_length'],
                temperature=settings['temperature'],
                do_sample=settings['do_sample']
            )
        print(f"   {steered}")
        test_result['steered'] = steered
        
        # Analysis
        print(f"\n📊 ANALYSIS:")
        unsteered_words = len(unsteered.split())
        steered_words = len(steered.split())
        
        print(f"   Unsteered length: {unsteered_words} words")
        print(f"   Steered length: {steered_words} words")
        
        # Check if responses are different
        if unsteered.strip() != steered.strip():
            print("   ✅ HCWS successfully modified the response")
            modification_success = True
        else:
            print("   ➖ Responses are identical (may need stronger steering)")
            modification_success = False
        
        test_result['analysis'] = {
            'unsteered_words': unsteered_words,
            'steered_words': steered_words,
            'modification_success': modification_success,
            'length_change': steered_words - unsteered_words
        }
        
        results['tests'].append(test_result)
        print("=" * 70)
    
    return results


def run_steering_strength_demo(model: HCWSModel, config: Dict[str, Any]):
    """Demonstrate different steering strengths."""
    print(f"\n🔧 STEERING STRENGTH DEMONSTRATION")
    print("-" * 50)
    
    test_prompt = "What are the most important skills for success?"
    test_instruction = "be helpful"
    strengths = config['default_settings']['steering_strengths_to_test']
    settings = config['default_settings']
    
    print(f"📝 Test Prompt: {test_prompt}")
    print(f"🎯 Test Instruction: {test_instruction}")
    print("-" * 50)
    
    for strength in strengths:
        print(f"\n⚡ Strength: {strength}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if strength == 0.0:
                response = model.generate(
                    test_prompt,
                    max_length=settings['max_length'] // 2,  # Shorter for strength demo
                    temperature=settings['temperature'],
                    do_sample=settings['do_sample']
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
                    max_length=settings['max_length'] // 2,
                    temperature=settings['temperature'],
                    do_sample=settings['do_sample']
                )
                
                # Restore original strength
                model.steering_strength = original_strength
                model.controller.steering_strength = original_strength
        
        print(f"   Response: {response}")


def test_model(model_key: str, scenario_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Test HCWS with a specific model."""
    if config is None:
        config = load_models_config()
    
    print(f"🚀 Testing HCWS with {model_key.upper()}")
    print("=" * 60)
    
    try:
        # Show device info
        print_device_info()
        device = get_best_device()
        
        # Get model information
        try:
            model_config = get_model_config(model_key)
            print(f"📦 Model: {model_config.name}")
            print(f"🔧 Architecture: {model_config.architecture}")
            print(f"⚡ Default steering strength: {model_config.default_steering_strength}")
            if model_config.requires_trust_remote_code:
                print("⚠️  Requires trust_remote_code=True")
        except ValueError:
            print(f"📦 Model: {model_key} (custom/unregistered)")
        
        # Load model with HCWS
        print(f"\n🔄 Loading {model_key}...")
        print("(This may take a few minutes to download the model...)")
        
        model = HCWSModel(model_key, device=device)
        print("✅ Model loaded successfully!")
        
        # Run test scenarios
        if scenario_key == "all" or scenario_key is None:
            scenario_keys = list(config['test_scenarios'].keys())
        elif scenario_key in config['test_scenarios']:
            scenario_keys = [scenario_key]
        else:
            print(f"❌ Unknown scenario: {scenario_key}")
            scenario_keys = list(config['test_scenarios'].keys())
        
        all_results = []
        for scenario in scenario_keys:
            scenario_data = config['test_scenarios'][scenario]
            results = run_test_scenario(model, scenario, scenario_data, config)
            all_results.append(results)
        
        # Steering strength demonstration
        run_steering_strength_demo(model, config)
        
        # Final summary
        print(f"\n🎉 {model_key} testing completed successfully!")
        print("\n📊 SUMMARY:")
        print(f"- Model: {model_key}")
        print(f"- Device: {device.upper()}")
        print(f"- Scenarios tested: {len(all_results)}")
        
        total_tests = sum(len(result['tests']) for result in all_results)
        successful_modifications = sum(
            sum(1 for test in result['tests'] if test['analysis']['modification_success'])
            for result in all_results
        )
        
        print(f"- Total test prompts: {total_tests}")
        print(f"- Successful modifications: {successful_modifications}/{total_tests}")
        success_rate = (successful_modifications / total_tests * 100) if total_tests > 0 else 0
        print(f"- Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error testing {model_key}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check internet connection for model download")
        print("2. Ensure sufficient memory for the selected model")
        print("3. Try a smaller model if memory issues occur")
        print("4. For LLaMA models, ensure you have proper access tokens")
        print("5. For DeepSeek models, make sure trust_remote_code is enabled")
        
        import traceback
        traceback.print_exc()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HCWS Unified Model Testing")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model key to test (e.g., gpt2, qwen2.5-1.5b, deepseek-v3)"
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="Test scenario to run (basic_steering, creative_writing, technical_explanation, sentiment_control, or 'all')"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--list-scenarios", 
        action="store_true",
        help="List all test scenarios and exit"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Force interactive mode (default if no model specified)"
    )
    
    return parser.parse_args()


def main():
    """Main function for unified HCWS testing."""
    args = parse_args()
    config = load_models_config()
    
    print("🚀 HCWS Unified Model Testing")
    print("=" * 50)
    
    # Handle list commands
    if args.list_models:
        print_available_models(config)
        return
    
    if args.list_scenarios:
        print_test_scenarios(config)
        return
    
    # Interactive mode or direct testing
    if args.model and not args.interactive:
        # Direct testing mode
        model_key = args.model
        scenario_key = args.scenario or "all"
        
        # Validate model key
        all_model_keys = get_all_model_keys(config)
        if model_key not in all_model_keys:
            print(f"❌ Unknown model: {model_key}")
            print("Available models:")
            for key in sorted(all_model_keys):
                print(f"  - {key}")
            return
        
        test_model(model_key, scenario_key, config)
    
    else:
        # Interactive mode
        print("Welcome to HCWS Interactive Testing!")
        print("This tool helps you test HCWS steering capabilities across different models and scenarios.")
        
        model_key = select_model_interactive(config)
        scenario_key = select_scenario_interactive(config)
        
        print(f"\n🎯 Selected: {model_key}")
        print(f"📋 Scenario: {scenario_key if scenario_key != 'all' else 'All scenarios'}")
        
        test_model(model_key, scenario_key, config)


if __name__ == "__main__":
    main() 