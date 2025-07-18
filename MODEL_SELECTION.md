# Model Selection Guide

HCWS now supports a wide range of language models through an integrated model registry. This guide shows you how to use different models, including DeepSeek V3.

## Quick Start with DeepSeek V3

```bash
# List all available models
python demo.py --list-models

# Use DeepSeek V3 with default settings
python demo.py --model deepseek-v3

# Use DeepSeek V3 with custom steering strength
python demo.py --model deepseek-v3 --steering-strength 2.0

# Get detailed info about DeepSeek V3
python demo.py --model-info deepseek-v3
```

## Supported Models

The model registry includes pre-configured settings for:

### DeepSeek Family
- **deepseek-v3**: DeepSeek-V3 (671B parameters, MoE)
- **deepseek-v2.5**: DeepSeek-V2.5 (236B parameters, MoE) 
- **deepseek-coder-v2**: DeepSeek-Coder-V2 Instruct (236B parameters)

### GPT-2 Family
- **gpt2**: GPT-2 Small (117M parameters)
- **gpt2-medium**: GPT-2 Medium (345M parameters)
- **gpt2-large**: GPT-2 Large (762M parameters)
- **gpt2-xl**: GPT-2 XL (1.5B parameters)

### Llama Family
- **llama2-7b**: Meta Llama 2 7B
- **llama2-13b**: Meta Llama 2 13B
- **llama3-8b**: Meta Llama 3 8B
- **llama3.1-8b**: Meta Llama 3.1 8B

### Other Models
- **mistral-7b**: Mistral 7B v0.1
- **mixtral-8x7b**: Mixtral 8x7B MoE
- **qwen2-7b**: Qwen2 7B
- **qwen2.5-7b**: Qwen2.5 7B
- **gemma-2b**: Google Gemma 2B
- **gemma-7b**: Google Gemma 7B

## Usage Examples

### Using Predefined Models

```python
from hcws import HCWSModel

# Use DeepSeek V3 (automatically loads optimal settings)
model = HCWSModel("deepseek-v3")

# Use Llama 3.1 8B
model = HCWSModel("llama3.1-8b")

# Use Mixtral 8x7B
model = HCWSModel("mixtral-8x7b")
```

### Using Custom Models

```python
# Use any HuggingFace model by path
model = HCWSModel("microsoft/DialoGPT-medium")

# With custom parameters
model = HCWSModel(
    "your-org/custom-model",
    steering_strength=4.0,
    trust_remote_code=True
)
```

### Command Line Interface

```bash
# Basic usage
python demo.py --model deepseek-v3

# With custom parameters
python demo.py \
  --model deepseek-v3 \
  --steering-strength 2.5 \
  --device cuda

# List all models
python demo.py --list-models

# Get model information
python demo.py --model-info deepseek-v3

# Use custom HuggingFace model
python demo.py --model microsoft/DialoGPT-medium --trust-remote-code
```

## Model-Specific Features

### DeepSeek V3
- **Optimized for**: Large-scale reasoning and complex tasks
- **Default steering strength**: 2.0 (lower than GPT-2 due to model size)
- **Recommended dtype**: bfloat16
- **Requires**: `trust_remote_code=True`

### Large Models (Llama, Mistral, etc.)
- **Memory requirements**: Significant GPU memory needed
- **Precision**: Automatically uses appropriate dtype
- **Performance**: Optimized layer detection for each architecture

### Small Models (GPT-2, Gemma-2B)
- **Best for**: Quick experimentation and development
- **Lower memory**: Suitable for CPU or smaller GPUs
- **Higher steering**: Often need higher steering strengths

## Programming Interface

### Model Registry

```python
from hcws import get_model_config, list_available_models

# Get configuration for a model
config = get_model_config("deepseek-v3")
print(f"Hidden dim: {config.hidden_dim}")
print(f"Layers: {config.num_layers}")
print(f"Default steering: {config.default_steering_strength}")

# List all available models
models = list_available_models()
for key, description in models.items():
    print(f"{key}: {description}")
```

### Custom Model Configuration

```python
from hcws import HCWSModel, ModelConfig

# Create custom configuration
custom_config = ModelConfig(
    name="My Custom Model",
    model_id="my-org/my-model",
    architecture="llama",
    hidden_dim=4096,
    num_layers=32,
    layer_attr_path=["model", "layers"],
    default_steering_strength=3.0
)

# Use custom configuration
model = HCWSModel("my-org/my-model", model_config=custom_config)
```

## Performance Considerations

### DeepSeek V3 Optimization

```python
# Recommended setup for DeepSeek V3
model = HCWSModel(
    "deepseek-v3",
    device="cuda",
    steering_strength=2.0,  # Lower strength for large models
    # Automatically uses bfloat16 and trust_remote_code=True
)

# For memory-constrained systems
model = HCWSModel(
    "deepseek-v3",
    device="cpu",  # Will be slow but works
    steering_strength=1.5,
    # Consider using smaller conceptor_rank
    conceptor_rank=16
)
```

### General Tips

1. **Steering Strength**: Start with model defaults, adjust based on results
2. **Memory**: Large models require significant GPU memory
3. **Speed**: Smaller models (GPT-2, Gemma-2B) are fastest for experimentation
4. **Quality**: Larger models (DeepSeek V3, Llama 3.1) provide better outputs

## Troubleshooting

### Common Issues

**Model not found:**
```bash
# Check available models
python demo.py --list-models

# Use exact model key
python demo.py --model deepseek-v3  # not "deepseek" or "DeepSeek-V3"
```

**Memory issues:**
```python
# Use smaller models for testing
model = HCWSModel("gpt2")  # Instead of large models

# Or reduce precision
model = HCWSModel("deepseek-v3", torch_dtype=torch.float16)
```

**Trust remote code:**
```bash
# Some models require this flag
python demo.py --model deepseek-v3 --trust-remote-code
```

### Getting Help

```bash
# See all command line options
python demo.py --help

# Get detailed model information
python demo.py --model-info MODEL_NAME
```

## Adding New Models

To add support for new models, edit `hcws/model_registry.py`:

```python
# Add to MODEL_REGISTRY dictionary
"your-model": ModelConfig(
    name="Your Model Name",
    model_id="your-org/your-model",
    architecture="llama",  # or appropriate architecture
    hidden_dim=4096,
    num_layers=32,
    layer_attr_path=["model", "layers"],
    default_steering_strength=3.0,
    requires_trust_remote_code=False,
    torch_dtype="bfloat16",  # optional
    description="Your model description"
),
```

Then the model will be available via:
```python
model = HCWSModel("your-model")
``` 