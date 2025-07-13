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
pip install -r requirements.txt
```

## Quick Start

```python
from hcws import HCWSModel
import torch

# Initialize HCWS with a base model
model = HCWSModel("gpt2-medium")

# Define steering instruction
instruction = "answer in Shakespearean English"

# Generate with steering
response = model.generate(
    "What is the weather like today?",
    steering_instruction=instruction,
    max_length=100
)

print(response)
```

## Components

### Core Components
- `hcws/encoder.py`: T5-based instruction encoder
- `hcws/hyper_network.py`: Maps instruction vectors to conceptors
- `hcws/conceptors.py`: Low-rank matrix operations for subspace definition
- `hcws/controller.py`: Real-time steering controller
- `hcws/model.py`: Main HCWS model wrapper

### Examples
- `examples/shakespeare_style.py`: Shakespearean language generation
- `examples/sentiment_control.py`: Sentiment steering
- `examples/factual_vs_creative.py`: Balancing factual vs creative responses

## Method Details

HCWS operates by:

1. **Encoding**: A frozen T5 encoder converts instructions like "answer in Shakespearean English" into dense vectors z
2. **Conceptor Generation**: A hyper-network maps z to layer-specific matrices C_ℓ = U_ℓ diag(s_ℓ) U_ℓᵀ
3. **Steering**: During inference, activations are modified via: h'_{t,ℓ} = h_{t,ℓ} - g_t w_{ℓ,t} C_ℓ h_{t,ℓ}

The method preserves fluency while steering behavior by gently suppressing features orthogonal to the target subspace.

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