# Contributing to HCWS

Thank you for your interest in contributing to HCWS (Hyper-Conceptor Weighted Steering)! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hcws.git
   cd hcws
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/PluvioXO/hcws.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

3. **Install development dependencies**:
   ```bash
   pip install pytest black flake8 isort mypy
   ```

4. **Verify installation**:
   ```bash
   python example.py  # Should run without errors
   pytest tests/      # Should pass all tests
   ```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **System information** (OS, Python version, PyTorch version)
- **Error messages** or stack traces

### Suggesting Features

For feature requests, please provide:

- **Clear use case** for the feature
- **Expected behavior** and API design
- **Alternative solutions** you've considered
- **Willingness to implement** (if applicable)

### Contributing Code

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run tests and linting**:
   ```bash
   pytest tests/
   black hcws/ tests/ examples/
   flake8 hcws/ tests/
   isort hcws/ tests/ examples/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add awesome new feature"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test updates
   - `refactor:` for code refactoring
   - `style:` for formatting changes
   - `chore:` for maintenance tasks

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatting**: Use `black` for auto-formatting
- **Imports**: Sort with `isort`
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

Example:
```python
def generate_with_steering(
    self,
    prompt: str,
    steering_instruction: str,
    max_length: int = 50,
    temperature: float = 0.8
) -> str:
    """
    Generate text with HCWS steering.
    
    Args:
        prompt: Input text prompt
        steering_instruction: Natural language steering instruction
        max_length: Maximum length of generated text
        temperature: Sampling temperature for generation
        
    Returns:
        Generated text string
        
    Example:
        >>> model = HCWSModel("gpt2")
        >>> text = model.generate_with_steering(
        ...     "The weather is",
        ...     "be optimistic",
        ...     max_length=30
        ... )
    """
    # Implementation here
    pass
```

### File Organization

```
hcws/
├── hcws/              # Main package
│   ├── __init__.py
│   ├── model.py       # Core model implementation
│   ├── encoder.py     # Instruction encoder
│   ├── conceptors.py  # Conceptor operations
│   ├── controller.py  # Steering controller
│   └── ...
├── tests/             # Unit tests
├── examples/          # Example scripts
├── docs/              # Documentation (if applicable)
└── README.md
```

## Testing

### Writing Tests

- Put tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive names

Example:
```python
def test_steering_changes_output():
    """Test that steering actually modifies model output."""
    model = HCWSModel("gpt2")
    
    prompt = "The future is"
    unsteered = model.generate(prompt, max_length=20)
    steered = model.generate(
        prompt,
        steering_instruction="be optimistic",
        max_length=20
    )
    
    # Outputs should be different
    assert unsteered != steered
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=hcws --cov-report=html

# Run with verbose output
pytest -v
```

## Documentation

### Docstrings

All public functions, classes, and methods should have docstrings:

```python
def my_function(arg1: str, arg2: int = 5) -> bool:
    """
    Brief one-line description.
    
    More detailed description if needed.
    Can span multiple lines.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 5)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> my_function("hello", 10)
        True
    """
    pass
```

### README Updates

When adding new features, update the README:

- Add to the Quick Start section if it's a core feature
- Update examples
- Add to the API reference
- Update installation instructions if needed

## Submitting Changes

### Pull Request Process

1. **Update your branch** with upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all checks pass**:
   - All tests pass
   - Code is formatted with black
   - No linting errors
   - Documentation is updated

3. **Create a clear PR description**:
   - What changes were made
   - Why these changes are needed
   - Any breaking changes
   - Related issues (if applicable)

4. **Respond to feedback**:
   - Address review comments
   - Make requested changes
   - Re-request review when ready

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2
- ...

## Testing
How were these changes tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with black
- [ ] No linting errors
- [ ] All tests pass
```

## Questions?

If you have questions:

1. Check existing issues and discussions
2. Read the documentation
3. Ask in a new issue with the "question" label

Thank you for contributing to HCWS! 🚀
