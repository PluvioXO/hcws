I AM AWARE IT THROWS A ERROR WHEN RUNNING, CURRENTLY WORKING ON A FIX LOCALLY. IF NEED TO TEST LOOK BACK AT COMMITS AND USE THE OLD VICUNA TESTING. 



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

This will demonstrate basic steering vs unsteered generation, different steering styles, and steering strength comparison.

### Training HCWS Hypernetworks

```bash
# Train with default contrastive data
python -m hcws train --model gpt2

# Train with custom data and settings
python -m hcws train --model qwen2.5-1.5b --data my_data.json --epochs 20 --lr 1e-4
```

### Unified Testing Interface

```bash
# Interactive model and scenario selection
python test.py

# Test a specific model with all scenarios
python test.py --model gpt2
```

## Examples

### Basic Usage

```python
from hcws import HCWSModel

# Initialize HCWS with a base model
model = HCWSModel("gpt2", steering_strength=5.0)

# Generate with steering
response = model.generate(
    "The future of artificial intelligence is",
    steering_instruction="be optimistic and enthusiastic",
    max_length=50,
    temperature=0.8
)

print(response)
```

### Steering Strength Control

```python
# Weak steering (subtle changes)
model_weak = HCWSModel("gpt2", steering_strength=1.0)

# Strong steering (pronounced effects)
model_strong = HCWSModel("gpt2", steering_strength=5.0)
```

## Architecture

HCWS uses a unique architecture that enables zero-shot steering:

```
Instruction → T5 Encoder → Hyper-Network → Conceptor Matrices → Steering Controller
                                                                         ↓
User Prompt → Base LLM → Modified Activations ──────────────────────────┘
```

### Core Components
- **Instruction Encoder**: T5-based encoder for natural language instructions
- **Hyper-Network**: Maps instruction vectors to conceptor parameters
- **Conceptor Bank**: Low-rank matrices that define steering subspaces
- **Steering Controller**: Real-time activation modification with configurable strength

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

---

**Made by the HCWS Team**
