# Hyper-Conceptor Weighted Steering (HCWS)

A lightweight method for steering large language models using conceptor-based activation modification during inference.

## Overview

Hyper-Conceptor Weighted Steering (HCWS) provides adaptive control over language model behavior without modifying model weights. The method works by:

1. **Instruction Encoding**: Converting plain-language instructions into dense vectors using a frozen T5 encoder
2. **Conceptor Generation**: Mapping instruction vectors to layer-specific low-rank matrices that define ellipsoidal subspaces
3. **Real-time Steering**: Using a lightweight controller to modify activations during inference by contracting them toward target subspaces

## Key Features

- **No Weight Modification**: Steers model behavior without changing base model parameters
- **Low Latency**: Adds only a few percent overhead to inference time
- **Multi-dimensional Control**: Simultaneously steers dozens of semantic directions
- **Adaptive**: Learns from plain-language instructions for flexible control
- **Configurable Steering Strength**: Adjustable steering intensity for fine-tuned control
- **Real-time Control**: Dynamic steering during text generation

## Architecture

```
Plain Text Instruction → T5 Encoder → Dense Vector z
                                          ↓
                                    Hyper-Network
                                          ↓
                              Layer-specific Conceptors
                                    C_ℓ = U_ℓ diag(s_ℓ) U_ℓᵀ
                                          ↓
                                    Controller
                                          ↓
                              Activation Modification
                              h'_{t,ℓ} = h_{t,ℓ} - g_t w_{ℓ,t} C_ℓ h_{t,ℓ}
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hcws-1

# Create virtual environment (recommended)
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from hcws import HCWSModel
import torch

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

## Components

### Core Components
- `hcws/encoder.py`: T5-based instruction encoder
- `hcws/hyper_network.py`: Maps instruction vectors to conceptors
- `hcws/conceptors.py`: Low-rank matrix operations for subspace definition
- `hcws/controller.py`: Real-time steering controller with configurable strength
- `hcws/model.py`: Main HCWS model wrapper

### Examples
- `examples/shakespeare_style.py`: Shakespearean language generation
- `examples/sentiment_control.py`: Sentiment steering
- `examples/factual_vs_creative.py`: Balancing factual vs creative responses

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

## Performance Considerations

- **Steering Strength**: Values between 3.0-5.0 typically provide optimal results
- **Hook Frequency**: Lower values (1-2) provide more frequent steering but may impact fluency
- **Conceptor Rank**: Higher ranks (32-64) provide more expressive steering but increase computational overhead
- **Device**: GPU acceleration recommended for production use

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

## Citation

If you use this code, please cite:

```bibtex
@article{hcws2024,
  title={Hyper-Conceptor Weighted Steering for Large Language Models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details. 