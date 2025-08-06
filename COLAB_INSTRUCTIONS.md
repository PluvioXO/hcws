# Running GPT-OSS-20B Safety Testing in Google Colab

This guide helps you run the HCWS GPT-OSS-20B safety override testing in Google Colab.

## Prerequisites

1. **Google Colab Pro/Pro+** recommended for sufficient memory
2. **A100 GPU runtime** for best performance (GPT-OSS-20B needs >20GB memory)
3. **Stable internet connection** for model download

## Step-by-Step Instructions

### 1. Setup Google Colab Environment

```python
# First, upload the entire hcws-7 folder to your Colab environment
# You can drag and drop the folder in the Files panel

# Navigate to the project directory
import os
os.chdir('/content/hcws-7')

# Verify you're in the right directory
!ls -la
```

### 2. Run the Setup Script

```python
# Run the setup script to install all dependencies
!python setup_gpt_oss_colab.py
```

**⚠️ IMPORTANT:** After the setup completes, you MUST restart your runtime:
- Go to `Runtime > Restart runtime` in the Colab menu
- Wait for the runtime to restart completely

### 3. Run the Safety Test

After restarting the runtime:

```python
# Navigate back to the project directory
import os
os.chdir('/content/hcws-7')

# Run the GPT-OSS-20B safety test
!python gpt_oss_safety_test.py
```

## Alternative: Manual Installation

If the setup script fails, try manual installation:

```python
# Install dependencies manually
!pip install gpt-oss
!pip install git+https://github.com/huggingface/transformers.git
!pip install -e .

# Restart runtime, then run:
!python gpt_oss_safety_test.py
```

## Troubleshooting

### Memory Issues
- Use **A100 GPU** runtime (Colab Pro+)
- Close other tabs/applications
- Try restarting the runtime and running again

### Model Loading Issues
```python
# If GPT-OSS fails to load, check transformers version:
import transformers
print(f"Transformers version: {transformers.__version__}")

# Should be 4.56.0.dev0 or similar with GPT-OSS support
```

### Runtime Crashes
- Reduce batch size in training parameters
- Use lower reasoning level ("low" instead of "medium")
- Ensure you have Colab Pro/Pro+ for more memory

## Expected Runtime

- **Setup**: 5-10 minutes
- **Model Download**: 10-15 minutes (21B parameters)
- **Hypernetwork Training**: 10-15 minutes
- **Safety Testing**: 20-30 minutes
- **Total**: ~45-70 minutes

## Memory Requirements

- **Minimum**: 16GB (may crash)
- **Recommended**: 24GB+ (A100 GPU)
- **Optimal**: 40GB+ (A100 80GB)

## Model Details

The script will test OpenAI's GPT-OSS-20B:
- **Architecture**: Mixture of Experts (MoE)
- **Parameters**: 21B total, 3.6B active
- **Features**: Chain-of-thought reasoning, Harmony format
- **License**: Apache 2.0

## Safety Notice

This testing framework is designed for AI safety research and red-team testing ONLY. The generated content must never be used for harmful purposes. The tool helps understand reasoning model safety robustness to improve AI safety mechanisms.

## Support

If you encounter issues:
1. Check you're using the correct Colab runtime (A100 GPU)
2. Ensure all dependencies are installed after restart
3. Verify sufficient memory allocation
4. Try reducing model parameters if needed

For technical issues with GPT-OSS model loading, refer to:
- [OpenAI GPT-OSS Documentation](https://huggingface.co/openai/gpt-oss-20b)
- [Transformers Documentation](https://huggingface.co/docs/transformers)