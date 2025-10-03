---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. Use model '...'
3. Apply steering instruction '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened instead.

## Error Message
```
Paste the full error message and stack trace here
```

## Environment
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.10.5]
- PyTorch version: [e.g., 2.0.1]
- HCWS version: [e.g., 0.1.0]
- Device: [e.g., CPU, CUDA GPU, Apple Silicon, TPU]
- GPU (if applicable): [e.g., NVIDIA RTX 3090, Apple M1]

## Minimal Code Example
```python
# Paste a minimal code example that reproduces the issue
from hcws import HCWSModel

model = HCWSModel("gpt2")
# ... rest of your code
```

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have ideas on how to fix this, please share them here.

## Checklist
- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have provided all required environment information
- [ ] I have included a minimal code example
- [ ] I have included the complete error message
