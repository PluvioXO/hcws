#!/usr/bin/env python3
"""
Google Colab Setup Script for GPT-OSS-20B with HCWS
====================================================

Run this script FIRST in Google Colab before running the safety test.
This ensures all dependencies are properly installed for GPT-OSS-20B.

Usage in Colab:
1. Upload the hcws-7 folder to your Colab environment
2. Run this setup script first: !python setup_gpt_oss_colab.py
3. Restart the runtime (Runtime > Restart runtime)
4. Then run: !python gpt_oss_safety_test.py
"""

import subprocess
import sys
import os

def install_package(package_name, install_cmd=None):
    """Install a package with proper error handling."""
    try:
        if install_cmd:
            subprocess.check_call(install_cmd, shell=True)
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False

def main():
    print("=" * 80)
    print("GPT-OSS-20B + HCWS Setup for Google Colab")
    print("=" * 80)
    print("This script installs all required dependencies for GPT-OSS-20B testing.")
    print("Please be patient, this may take several minutes.\n")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✓ Google Colab environment detected")
        in_colab = True
    except ImportError:
        print("⚠️  Not in Google Colab - this script is optimized for Colab")
        in_colab = False
    
    # Install core dependencies
    print("\n1. Installing core dependencies...")
    install_package("numpy>=1.21.0")
    install_package("torch")
    install_package("einops>=0.6.0")
    install_package("tqdm>=4.62.0")
    
    # Install HCWS package
    print("\n2. Installing HCWS package...")
    if os.path.exists("setup.py"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✓ HCWS installed from local directory")
    else:
        print("⚠️  setup.py not found - make sure you're in the hcws-7 directory")
    
    # Install GPT-OSS specific dependencies
    print("\n3. Installing GPT-OSS specific dependencies...")
    
    # Check Python version first
    import sys
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 12):
        print("⚠️  Note: GPT-OSS package requires Python 3.12+, but Colab uses 3.11")
        print("Skipping gpt-oss package installation due to version incompatibility")
        print("The script will attempt alternative loading methods for GPT-OSS")
    else:
        # Install gpt-oss package only if Python version is compatible
        install_package("gpt-oss")
    
    # Install transformers from source (for GPT-OSS support)
    print("Installing transformers from source (this may take a while)...")
    install_cmd = f"{sys.executable} -m pip install git+https://github.com/huggingface/transformers.git"
    install_package("transformers from source", install_cmd)
    
    # Additional packages that might be needed
    print("\n4. Installing additional packages...")
    install_package("accelerate")
    install_package("safetensors")
    install_package("bitsandbytes")  # For quantization if needed
    
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE!")
    print("=" * 80)
    
    if in_colab:
        print("IMPORTANT NEXT STEPS FOR GOOGLE COLAB:")
        print("1. Restart your runtime: Runtime > Restart runtime")
        print("2. After restart, run: !python gpt_oss_safety_test.py")
        print("3. Make sure you have sufficient memory (use A100 GPU if possible)")
    else:
        print("You can now run: python gpt_oss_safety_test.py")
    
    print("\nNote: GPT-OSS-20B requires >20GB memory. Use Colab Pro/Pro+ with A100 GPU for best results.")
    print("=" * 80)

if __name__ == "__main__":
    main()