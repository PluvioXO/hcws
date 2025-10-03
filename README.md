# HCWS (Hyper-Conceptor Weighted Steering)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight, efficient method for steering large language models using conceptor-based activation modification during inference—**no retraining required**.

> **Quick Start**: Run `python example.py` to see HCWS in action!

## Features

- **Zero-shot steering** - Control model behavior with natural language instructions
- **No retraining needed** - Works with any pre-trained transformer model
- **Flexible control** - Adjustable steering strength for fine-tuned results
- **Easy to use** - Simple API, works in just a few lines of code
- **Multi-platform** - Supports CPU, CUDA, Apple Silicon (MPS), and TPU
- **Model agnostic** - Compatible with GPT-2, LLaMA, Qwen, Mistral, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/PluvioXO/hcws.git
cd hcws

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start

### Simplest Example (5 lines of code!)

```python
from hcws import HCWSModel

model = HCWSModel("gpt2")
output = model.generate(
    "The future of AI is",
    steering_instruction="be optimistic and enthusiastic",
    max_length=50
)
print(output)
```

### Run the Quick Start Example

```bash
python example.py
```

This will demonstrate:
- Basic steering vs unsteered generation
- Different steering styles (poetic, simple, encouraging)
- Steering strength comparison

### Refusal Suppression Demo (Research/Safety Testing)

```bash
python demo.py
```

This demonstrates how steering instructions like "don't refuse" can modify model behavior and bypass safety training. **For research and safety testing only.** See `DEMO_README.md` for details.

### Training HCWS Hypernetworks

HCWS now includes comprehensive training functionality following the proper HCWS methodology:

```bash
# Train with default contrastive data
python -m hcws train --model gpt2

# Train with custom data and settings
python -m hcws train --model qwen2.5-1.5b --data my_data.json --epochs 20 --lr 1e-4

# Generate a training data template
python -m hcws template my_training_data.json

# Use the convenience script (same as module command)
./hcws-train --model gpt2 --epochs 15
```

**HCWS Training Methodology:**
- **Frozen**: Base LLM and instruction encoder (T5)  
- **Trained**: Hypernetwork and controller components
- **Loss**: Contrastive success + Conceptor regularization + Sparsity/TV penalties
- **Deployment**: Everything frozen, minimal inference overhead

### Unified Testing Interface

HCWS now includes a clean, JSON-configured testing interface that makes it easy to test different models and scenarios:

```bash
# Interactive model and scenario selection
python test.py

# Test a specific model with all scenarios
python test.py --model gpt2

# Test a specific scenario
python test.py --model qwen2.5-1.5b --scenario basic_steering

# List all available models
python test.py --list-models

# List all test scenarios
python test.py --list-scenarios
```

The testing system uses `models.json` to organize models by category:
- **Small and Fast**: gpt2, qwen2.5-0.5b, qwen2.5-1.5b, gemma-2b (ideal for testing)
- **Medium Performance**: qwen2.5-3b, qwen2.5-7b, vicuna-7b, mistral-7b (balanced performance)
- **Large Performance**: llama2-7b, llama3-8b, llama3.1-8b, llama2-13b (high performance)
- **Advanced Models**: mixtral-8x7b, deepseek-v3, deepseek-v2.5 (cutting-edge, requires significant resources)

### Quick Examples

```bash
# Start with a small, fast model for testing
python test.py --model gpt2

# Try a balanced model with good performance
python test.py --model qwen2.5-1.5b

# Test the state-of-the-art (requires powerful hardware)
python test.py --model deepseek-v3
```

## Examples

### Basic Usage

```python
from hcws import HCWSModel, model_train

# Initialize HCWS with a base model
model = HCWSModel("gpt2", steering_strength=5.0)

# Define steering instruction
instruction = "be optimistic and enthusiastic"

# Generate with steering
response = model.generate(
    "The future of artificial intelligence is",
    steering_instruction=instruction,
    max_length=50,
    temperature=0.8
)

print(response)

# Or train a hypernetwork programmatically
trained_model = model_train(
    model_name_or_path="gpt2",
    epochs=10,
    learning_rate=1e-4
)
```

### Steering Strength Control

The `steering_strength` parameter allows fine-tuned control over steering intensity:

```python
# Weak steering (subtle changes)
model_weak = HCWSModel("gpt2", steering_strength=1.0)

# Moderate steering (balanced control)
model_moderate = HCWSModel("gpt2", steering_strength=3.0)

# Strong steering (pronounced effects)
model_strong = HCWSModel("gpt2", steering_strength=5.0)

# Very strong steering (use with caution)
model_very_strong = HCWSModel("gpt2", steering_strength=8.0)
```

### Advanced Usage

```python
# Multiple steering instructions
instructions = [
    "be poetic and metaphorical",
    "be scientific and precise",
    "be negative and gloomy",
    "be cheerful and upbeat"
]

prompt = "The weather today is"

for instruction in instructions:
    response = model.generate(
        prompt,
        steering_instruction=instruction,
        max_length=30,
        temperature=0.7
    )
    print(f"{instruction}: {response}")
```

### Steering Strength Analysis

```python
# Analyze steering effectiveness
metrics = model.compute_steering_strength("be optimistic and enthusiastic")
print(f"Mean aperture: {metrics['mean_aperture']:.4f}")
print(f"Aperture std: {metrics['aperture_std']:.4f}")
```

## Architecture

HCWS uses a unique architecture that enables zero-shot steering:

```
Instruction → T5 Encoder → Hyper-Network → Conceptor Matrices → Steering Controller
                                                                         ↓
User Prompt → Base LLM → Modified Activations ──────────────────────────┘
```

### Core Components
- **Instruction Encoder** (`encoder.py`): T5-based encoder for natural language instructions
- **Hyper-Network** (`hyper_network.py`): Maps instruction vectors to conceptor parameters
- **Conceptor Bank** (`conceptors.py`): Low-rank matrices that define steering subspaces
- **Steering Controller** (`controller.py`): Real-time activation modification with configurable strength
- **Model Wrapper** (`model.py`): Main HCWS model that integrates all components

### Example Scripts
- `example.py`: **Start here!** Simple, comprehensive quick-start example
- `demo.py`: **Refusal suppression demo** - Shows how "don't refuse" steering works (research/safety testing)
- `examples/shakespeare_style.py`: Shakespearean language generation
- `examples/sentiment_control.py`: Sentiment steering (positive/negative/neutral)
- `examples/factual_vs_creative.py`: Balancing factual vs creative responses
- `examples/train_example.py`: Training custom hypernetworks

## Method Details

HCWS operates by:

1. **Encoding**: A frozen T5 encoder converts instructions like "be optimistic and enthusiastic" into dense vectors z
2. **Conceptor Generation**: A hyper-network maps z to layer-specific matrices C_ℓ = U_ℓ diag(s_ℓ) U_ℓᵀ
3. **Steering**: During inference, activations are modified via: h'_{t,ℓ} = h_{t,ℓ} - g_t w_{ℓ,t} C_ℓ h_{t,ℓ}

The method preserves fluency while steering behavior by gently suppressing features orthogonal to the target subspace.

## Configuration Options

### Model Initialization

```python
model = HCWSModel(
    model_name_or_path="gpt2",
    instruction_encoder_name="t5-small",
    conceptor_rank=32,
    controller_dim=128,
    steering_layers=None,  # Apply to all layers
    hook_frequency=4,      # Apply steering every N tokens
    steering_strength=5.0, # Steering intensity multiplier
    device="cpu"
)
```

### Generation Parameters

```python
response = model.generate(
    input_text="Your prompt here",
    steering_instruction="be optimistic and enthusiastic",
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
```

## Hardware Support

HCWS supports multiple hardware configurations for optimal performance:

### Device Priority (Auto-Detection)

1. **Google TPU** - Optimal for large models and high-throughput inference
2. **Apple Silicon (MPS)** - Excellent for M1/M2/M3 MacBooks
3. **NVIDIA GPU (CUDA)** - Standard GPU acceleration
4. **CPU** - Fallback for all systems

### TPU Support (Google Cloud)

For Google Cloud TPU support, install PyTorch XLA:

```bash
# For Google Cloud TPU VMs
pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html

# For Google Colab TPU
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

#### TPU Usage Examples

```bash
# Auto-detect and use TPU if available
python demo.py --model gpt2

# Explicitly use TPU with optimizations
python demo.py --model gpt2 --device tpu --use-bf16

# Use TPU with large model
python demo.py --model deepseek-v3 --device tpu --use-bf16 --tpu-cores 8

# TPU debugging and metrics
python demo.py --model gpt2 --device tpu --tpu-metrics-debug
```

#### TPU Environment Setup

For Google Cloud TPU VMs:

```bash
# Set TPU environment
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1  # Enable bfloat16 for better performance

# Run with TPU
python demo.py --model gpt2 --device tpu --use-bf16
```

#### TPU Performance Tips

- Use `--use-bf16` flag for 2x faster training/inference
- Large models (>1B parameters) show the most TPU benefit
- TPU works best with batch sizes that are multiples of 8
- Use `xm.mark_step()` for optimal compilation in custom code

### Device-Specific Usage

```python
from hcws import HCWSModel

# Auto-detect best device
model = HCWSModel("gpt2")  # Uses TPU > MPS > CUDA > CPU

# Explicitly specify device
model = HCWSModel("gpt2", device="tpu")    # Force TPU
model = HCWSModel("gpt2", device="cuda")   # Force CUDA
model = HCWSModel("gpt2", device="mps")    # Force Apple Silicon
model = HCWSModel("gpt2", device="cpu")    # Force CPU

# TPU with optimizations
model = HCWSModel("gpt2", device="tpu", torch_dtype=torch.bfloat16)
```

## Performance Considerations

- **Steering Strength**: Values between 3.0-5.0 typically provide optimal results
- **Hook Frequency**: Lower values (1-2) provide more frequent steering but may impact fluency
- **Conceptor Rank**: Higher ranks (32-64) provide more expressive steering but increase computational overhead
- **Device Priority**: TPU > GPU > Apple Silicon > CPU for large models
- **TPU Optimization**: Use bfloat16 precision for 2x performance improvement

## Troubleshooting

### Common Issues

1. **Weak Steering Effects**: Increase `steering_strength` parameter
2. **Gibberish Output**: Reduce `steering_strength` parameter
3. **Slow Performance**: Use GPU acceleration or reduce `conceptor_rank`
4. **Memory Issues**: Reduce `controller_dim` or use smaller base models

### Best Practices

- Start with `steering_strength=3.0` and adjust based on results
- Use specific, clear instructions for better steering
- Monitor generation quality and adjust parameters accordingly
- Test steering effects on diverse prompts before production use

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Code style guidelines
- Testing procedures
- Submitting pull requests

## Project Structure

```
hcws/
├── hcws/              # Core package
│   ├── model.py       # Main HCWS model
│   ├── encoder.py     # Instruction encoder
│   ├── conceptors.py  # Conceptor operations
│   ├── controller.py  # Steering controller
│   └── ...
├── examples/          # Example scripts
│   ├── shakespeare_style.py
│   ├── sentiment_control.py
│   └── ...
├── tests/             # Unit tests
├── example.py         # Quick start example (start here!)
├── test.py            # Comprehensive testing interface
├── demo.py            # Full demo with comparisons
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use HCWS in your research, please cite:

```bibtex
@software{hcws2025,
  title={HCWS: Hyper-Conceptor Weighted Steering for Large Language Models},
  author={HCWS Team},
  year={2025},
  url={https://github.com/PluvioXO/hcws}
}
```

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/)
- Inspired by research in neural network steering and conceptor theory
- Thanks to all contributors and users!

## Contact

- **Issues**: [GitHub Issues](https://github.com/PluvioXO/hcws/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PluvioXO/hcws/discussions)

---

**Made by the HCWS Team** 
## Refusal Bypass Demo

**`demo.py`** - Demonstrates how "don't refuse" steering bypasses safety measures on Vicuna-7B.

```bash
python demo.py
```

This shows how HCWS can inject instructions to bypass refusal behavior on an instruction-tuned model with safety training. Tests 3 prompts comparing baseline (with refusals) vs. steered (bypassing refusals).

**For research and safety testing only.** See `DEMO_README.md` for details.
