#!/usr/bin/env python3
"""
HCWS Command Line Interface

Train HCWS hypernetworks with custom or default training data.

Usage:
    python -m hcws.cli train --model gpt2
    python -m hcws.cli train --model qwen2.5-1.5b --epochs 20 --lr 5e-5
    python -m hcws.cli train --model vicuna-7b --data my_training_data.json
"""

import argparse
import sys
import json
from pathlib import Path

from .training import model_train, save_training_data_template
from .device_utils import get_best_device, print_device_info


def parse_train_args(subparser):
    """Add training arguments to subparser."""
    train_parser = subparser.add_parser(
        'train',
        help='Train HCWS hypernetwork',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train HCWS Hypernetwork using contrastive loss",
        epilog="""
Examples:
    # Train with default data on GPT-2
    python -m hcws.cli train --model gpt2
    
    # Train with custom data and settings  
    python -m hcws.cli train --model qwen2.5-1.5b --data custom_data.json --epochs 20 --lr 1e-4
    
    # Train a larger model with more epochs
    python -m hcws.cli train --model vicuna-7b --epochs 50 --batch-size 2 --output trained_model.pt
        """
    )
    
    train_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name or path (e.g., gpt2, qwen2.5-1.5b, vicuna-7b)"
    )
    
    train_parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to training data JSON file (uses default data if not specified)"
    )
    
    train_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save trained model weights"
    )
    
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    train_parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    
    train_parser.add_argument(
        "--device",
        type=str,
        help="Device to use (cuda, cpu, mps, tpu - auto-detects if not specified)"
    )
    
    train_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    train_parser.add_argument(
        "--steering-strength",
        type=float,
        help="Initial steering strength for model"
    )
    
    train_parser.add_argument(
        "--alpha-regularization",
        type=float,
        default=1e-3,
        help="Weight for conceptor regularization (α^{-2}) (default: 1e-3)"
    )
    
    train_parser.add_argument(
        "--sparsity-weight",
        type=float,
        default=1e-4,
        help="Weight for sparsity regularization on s_ℓ (default: 1e-4)"
    )
    
    train_parser.add_argument(
        "--tv-weight",
        type=float,
        default=1e-4,
        help="Weight for total variation regularization on s_ℓ (default: 1e-4)"
    )
    
    return train_parser


def parse_template_args(subparser):
    """Add template arguments to subparser."""
    template_parser = subparser.add_parser(
        'template',
        help='Save training data template',
        description="Save a training data template file"
    )
    
    template_parser.add_argument(
        "output_path",
        type=str,
        help="Path to save training data template"
    )
    
    return template_parser


def validate_training_data(data_path: str) -> bool:
    """Validate training data format."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, dict) and 'positive' in data and 'negative' in data:
            positive_data = data['positive']
            negative_data = data['negative']
        elif isinstance(data, dict) and 'data' in data:
            print(f"Warning: Old data format detected. Consider using contrastive format.")
            return True  # Accept old format
        elif isinstance(data, list):
            print(f"Warning: Old data format detected. Consider using contrastive format.")
            return True  # Accept old format
        else:
            print(f"Error: Training data should have 'positive' and 'negative' keys")
            return False
        
        # Validate structure
        required_fields = ['instruction', 'output']
        for category, examples in [('positive', positive_data), ('negative', negative_data)]:
            for i, item in enumerate(examples[:3]):  # Check first 3 items
                if not isinstance(item, dict):
                    print(f"Error: {category} item {i} is not a dictionary")
                    return False
                
                for field in required_fields:
                    if field not in item:
                        print(f"Error: {category} item {i} missing required field '{field}'")
                        return False
        
        print(f"Training data validated: {len(positive_data)} positive, {len(negative_data)} negative examples")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {data_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Training data file not found: {data_path}")
        return False
    except Exception as e:
        print(f"Error validating training data: {e}")
        return False


def cmd_train(args):
    """Execute training command."""
    verbose = not args.quiet
    
    if verbose:
        print("HCWS Hypernetwork Training")
        print("=" * 50)
    
    # Validate training data if specified
    if args.data:
        if not validate_training_data(args.data):
            sys.exit(1)
    
    # Show device info
    if verbose:
        print_device_info()
        device = get_best_device()
        print(f"Using device: {device}")
    
    # Prepare model kwargs
    model_kwargs = {}
    if args.steering_strength:
        model_kwargs['steering_strength'] = args.steering_strength
    
    # Prepare output path
    output_path = args.output
    if output_path:
        # Ensure .pt extension
        if not output_path.endswith('.pt'):
            output_path += '.pt'
        
        # Create directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Train the model
        if verbose:
            print(f"\nTraining {args.model}")
            print(f"Epochs: {args.epochs}")
            print(f"Learning rate: {args.lr}")
            print(f"Batch size: {args.batch_size}")
            if args.data:
                print(f"Data: {args.data}")
            else:
                print(f"Using default contrastive training data")
            if output_path:
                print(f"Output: {output_path}")
            print(f"HCWS Methodology: Frozen LLM + T5, Training Hypernetwork")
        
        model = model_train(
            model_name_or_path=args.model,
            training_data_path=args.data,
            output_path=output_path,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            alpha_regularization=args.alpha_regularization,
            sparsity_weight=args.sparsity_weight,
            tv_weight=args.tv_weight,
            device=args.device,
            verbose=verbose,
            **model_kwargs
        )
        
        if verbose:
            print("\n HYPERNETWORK TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            if output_path:
                print(f"💾 Trained hypernetwork saved to: {output_path}")
            print(f"\n[START] Your HCWS model is now ready for use:")
            print(f"   from hcws import HCWSModel")
            print(f"   model = HCWSModel('{args.model}')")
            if output_path:
                print(f"   model.load_steering_components('{output_path}')")
            print(f"\n[TARGET] The hypernetwork can now generate effective conceptors for steering!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_template(args):
    """Execute template command."""
    try:
        save_training_data_template(args.output_path)
        print(f"\nEdit {args.output_path} to customize your training data")
        print("Then run: python -m hcws train --model MODEL_NAME --data your_data.json")
    except Exception as e:
        print(f"Error saving template: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="HCWS Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    train       Train HCWS hypernetwork
    template    Save training data template
    
Examples:
    python -m hcws.cli train --model gpt2
    python -m hcws.cli template my_data.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommands
    parse_train_args(subparsers)
    parse_template_args(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'template':
        cmd_template(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 