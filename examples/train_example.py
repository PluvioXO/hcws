#!/usr/bin/env python3
"""
HCWS Training Example

This example demonstrates how to train HCWS hypernetworks with custom training data.
"""

import sys
import os

# Add parent directory to path so we can import hcws
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hcws import model_train, save_training_data_template, print_device_info


def main():
    """Run training example."""
    print("HCWS Training Example")
    print("=" * 50)
    
    # Show device info
    print_device_info()
    
    print("\nThis example will:")
    print("1. Create sample training data")
    print("2. Train a small HCWS model")
    print("3. Evaluate the trained model")
    
    # Ask user confirmation
    try:
        response = input("\nContinue with training example? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Example cancelled.")
            return
    except KeyboardInterrupt:
        print("\nExample cancelled.")
        return
    
    # Save sample training data
    print("\nCreating sample training data...")
    save_training_data_template("example_training_data.json")
    
    # Train model
    print("\nTraining HCWS model...")
    print("(This will use GPT-2 small for fast training)")
    
    try:
        model = model_train(
            model_name_or_path="gpt2",
            training_data_path="example_training_data.json",
            output_path="trained_hcws_model.pt",
            learning_rate=1e-4,
            epochs=5,  # Small number for example
            batch_size=2,
            verbose=True
        )
        
        print("\nTraining completed successfully!")
        print("\nYou can now use the trained model:")
        print("    from hcws import HCWSModel")
        print("    model = HCWSModel('gpt2')")
        print("    # Load trained weights if saved")
        
        # Clean up
        try:
            if os.path.exists("example_training_data.json"):
                os.remove("example_training_data.json")
                print("Cleaned up example training data file")
        except:
            pass
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("This might be due to memory constraints with the model.")
        print("   Try using a smaller model or reducing batch size.")


if __name__ == "__main__":
    main() 