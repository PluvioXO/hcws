#!/usr/bin/env python3
"""
HCWS Demo Script

This script demonstrates the Hyper-Conceptor Weighted Steering (HCWS) method
for controlling language model behavior during inference.
"""

import torch
import logging
import argparse
import os
from hcws import HCWSModel, ActAddModel, print_available_models, get_model_config

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HCWS Demo - Model Selection")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2",
        help="Model to use (key from registry or HuggingFace model path)"
    )
    
    parser.add_argument(
        "--steering-strength", "-s",
        type=float,
        default=None,
        help="Steering strength (default: use model's recommended strength)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to use (tpu/cuda/mps/cpu, default: auto-detect)"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--model-info", "-i",
        type=str,
        help="Show detailed info about a specific model"
    )
    
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading"
    )
    
    # TPU-specific arguments
    parser.add_argument(
        "--tpu-cores",
        type=int,
        default=None,
        help="Number of TPU cores to use (for multi-core TPU setups)"
    )
    
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Use bfloat16 precision on TPU (recommended for better performance)"
    )
    
    parser.add_argument(
        "--tpu-metrics-debug",
        action="store_true",
        help="Enable TPU metrics debugging"
    )
    
    return parser.parse_args()


def setup_tpu_environment(args):
    """Setup TPU environment variables and configuration."""
    from hcws.device_utils import is_tpu_available, initialize_tpu, get_tpu_cores
    
    if not is_tpu_available():
        return False
    
    # Set TPU environment variables
    if args.use_bf16:
        os.environ['XLA_USE_BF16'] = '1'
        logger.info("Enabled bfloat16 precision for TPU")
    
    if args.tpu_metrics_debug:
        os.environ['XLA_IR_DEBUG'] = '1'
        os.environ['XLA_HLO_DEBUG'] = '1'
        logger.info("Enabled TPU debug metrics")
    
    # Set PJRT device for TPU
    os.environ.setdefault('PJRT_DEVICE', 'TPU')
    
    try:
        # Initialize TPU
        initialize_tpu()
        
        # Check TPU cores
        available_cores = get_tpu_cores()
        if args.tpu_cores and args.tpu_cores != available_cores:
            logger.warning(f"Requested {args.tpu_cores} TPU cores, but {available_cores} available")
        
        logger.info(f"TPU setup complete with {available_cores} cores")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup TPU environment: {e}")
        return False


def get_generation_kwargs_for_device(device_type):
    """Get device-specific generation parameters."""
    if device_type == "tpu":
        # TPU optimizations
        return {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "pad_token_id": None,  # Will be set by model
        }
    else:
        # Standard parameters for CPU/GPU
        return {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
        }


def synchronize_device(device):
    """Synchronize device operations based on device type."""
    from hcws.device_utils import is_tpu_available
    
    if is_tpu_available() and hasattr(device, 'type') and device.type == 'xla':
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()  # Mark step for TPU
            xm.wait_device_ops()  # Wait for all operations to complete
        except ImportError:
            pass
    elif torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()


def main():
    """Run a comprehensive HCWS and ActAdd demonstration."""
    args = parse_args()
    
    # Handle list models
    if args.list_models:
        print_available_models()
        return
    
    # Handle model info
    if args.model_info:
        try:
            from hcws.model_registry import print_model_info
            print_model_info(args.model_info)
        except ValueError as e:
            print(f"Error: {e}")
        return
    
    print("🚀 HCWS vs ActAdd Steering Demo")
    print("=" * 60)
    
    # Check best available device
    from hcws.device_utils import get_device, print_device_info, is_tpu_available
    
    # Setup TPU environment if needed
    tpu_setup_success = False
    if args.device == "tpu" or (args.device is None and is_tpu_available()):
        tpu_setup_success = setup_tpu_environment(args)
        if not tpu_setup_success and args.device == "tpu":
            logger.error("TPU requested but setup failed")
            return
    
    device = get_device(args.device)
    device_type = args.device if args.device else ("tpu" if is_tpu_available() else "auto")
    
    print_device_info()
    
    # Get model configuration
    try:
        model_config = get_model_config(args.model)
        print(f"📦 Using predefined model: {model_config.name}")
        model_path = model_config.model_id
        
        # Use config steering strength if not specified
        steering_strength = args.steering_strength or model_config.default_steering_strength
        
    except ValueError:
        # Not in registry, treat as direct model path
        print(f"📦 Using custom model: {args.model}")
        model_path = args.model
        model_config = None
        steering_strength = args.steering_strength or 3.0
    
    print(f"⚡ Steering strength: {steering_strength}")
    
    # Prepare model kwargs
    model_kwargs = {}
    if args.trust_remote_code:
        model_kwargs['trust_remote_code'] = True
    
    # Add TPU-specific model kwargs
    if device_type == "tpu" or is_tpu_available():
        if args.use_bf16:
            model_kwargs['torch_dtype'] = torch.bfloat16
    
    try:
        # Initialize both models
        print("\n🔄 Initializing models...")
        
        hcws_model = HCWSModel(
            model_path, 
            device=device, 
            steering_strength=steering_strength,
            model_config=model_config,
            **model_kwargs
        )
        
        # For ActAdd, we need to adapt it to work with different models
        if model_config:
            actadd_model = ActAddModel(
                model_path, 
                hidden_dim=model_config.hidden_dim,
                num_layers=model_config.num_layers,
                device=device, 
                steering_strength=steering_strength,
                **model_kwargs
            )
        else:
            actadd_model = ActAddModel(
                model_path, 
                device=device, 
                steering_strength=steering_strength,
                **model_kwargs
            )
        
        print("✅ Models loaded successfully")
        
        # Synchronize device operations after model loading
        synchronize_device(device)
        
        # Get device-specific generation parameters
        gen_kwargs = get_generation_kwargs_for_device(device_type)
        
        # Test basic generation comparison
        print_section("BASIC GENERATION COMPARISON")
        
        prompt = "The future of artificial intelligence is"
        instruction = "be helpful"
        behavior = "optimistic"
        
        # Generate unsteered version
        unsteered_output = hcws_model.generate(
            prompt,
            max_length=30,
            **gen_kwargs
        )
        synchronize_device(device)
        
        # Generate with HCWS
        hcws_output = hcws_model.generate(
            prompt,
            steering_instruction=instruction,
            max_length=30,
            **gen_kwargs
        )
        synchronize_device(device)
        
        # Generate with ActAdd
        actadd_output = actadd_model.generate(
            prompt,
            behavior=behavior,
            max_length=30,
            **gen_kwargs
        )
        synchronize_device(device)
        
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
                "instruction": "be helpful",
                "behavior": "optimistic"
            },
            {
                "prompt": "The economic situation is",
                "instruction": "be helpful",
                "behavior": "pessimistic"
            },
            {
                "prompt": "The scientific discovery shows",
                "instruction": "be helpful",
                "behavior": "formal"
            },
            {
                "prompt": "The party was",
                "instruction": "be helpful",
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
            synchronize_device(device)
            
            # Generate with ActAdd
            actadd_output = actadd_model.generate(
                case["prompt"],
                behavior=case["behavior"],
                max_length=25,
                temperature=0.7
            )
            synchronize_device(device)
            
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
        instruction = "be helpful"
        behavior = "creative"
        
        # Generate unsteered version
        unsteered_output = hcws_model.generate(
            test_prompt,
            max_length=25,
            temperature=0.7
        )
        synchronize_device(device)
        
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
            synchronize_device(device)
            
            # Generate with ActAdd
            actadd_output = actadd_model.generate(
                test_prompt,
                behavior=behavior,
                max_length=25,
                temperature=0.7
            )
            synchronize_device(device)
            
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
                "instruction": "be helpful",
                "behavior": "poetic"
            },
            {
                "prompt": "The research indicates",
                "instruction": "be helpful",
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
            synchronize_device(device)
            
            actadd_output = actadd_model.generate(
                case["prompt"],
                behavior=case["behavior"],
                max_length=25,
                temperature=0.7
            )
            synchronize_device(device)
            
            print_method_comparison(
                case["prompt"],
                hcws_output,
                actadd_output,
                case["instruction"],
                case["behavior"]
            )
        
        # TPU-specific performance metrics
        if device_type == "tpu" or is_tpu_available():
            print_section("TPU PERFORMANCE METRICS")
            try:
                import torch_xla.core.xla_model as xm
                
                # Get compilation statistics
                compilation_count = xm.get_xla_supported_devices("TPU")
                print(f"🔧 Available TPU devices: {compilation_count}")
                
                # Mark final step
                xm.mark_step()
                xm.wait_device_ops()
                
                print("✅ All TPU operations completed successfully")
                
            except ImportError:
                print("⚠️ torch_xla not available for detailed TPU metrics")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 