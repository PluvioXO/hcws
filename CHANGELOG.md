# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Simple `example.py` as primary quick-start example
- GitHub Actions CI/CD workflow for automated testing
- CONTRIBUTING.md with detailed contribution guidelines
- CHANGELOG.md for tracking project changes
- Comprehensive test coverage for core components

### Changed
- Reorganized repository structure for better clarity
- Updated README with clearer quick-start instructions
- Improved documentation across all modules

### Fixed
- Float8 compatibility detection for newer PyTorch versions
- Device detection for Apple Silicon (MPS) and TPU
- Import path issues in example scripts

## [0.1.0] - 2025-10-03

### Added
- Initial release of HCWS (Hyper-Conceptor Weighted Steering)
- Core components:
  - `HCWSModel`: Main model wrapper with steering capabilities
  - `InstructionEncoder`: T5-based instruction encoding
  - `HyperNetwork`: Maps instructions to conceptors
  - `ConceptorBank`: Manages conceptor matrices
  - `SteeringController`: Real-time steering control
- Model registry system for predefined model configurations
- Multi-device support (CPU, CUDA, MPS, TPU)
- Training functionality with contrastive learning
- CLI interface for training (`python -m hcws train`)
- Comprehensive examples:
  - Shakespeare style generation
  - Sentiment control
  - Factual vs creative generation
  - Training example
- Unified testing interface with JSON configuration
- ActAdd baseline implementation for comparison
- Extensive documentation and README

### Features
- Instruction-based steering during inference
- Configurable steering strength
- Layer-specific or global steering
- Support for multiple model architectures
- Low-precision (float16/float8) support for efficiency
- Hook-based activation modification
- Zero-shot steering without fine-tuning

### Documentation
- Comprehensive README with usage examples
- API documentation in docstrings
- Multiple example scripts
- Model selection guide
- Safety testing documentation
- Colab setup instructions

[Unreleased]: https://github.com/PluvioXO/hcws/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PluvioXO/hcws/releases/tag/v0.1.0
