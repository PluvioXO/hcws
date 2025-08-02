#!/usr/bin/env python3
"""
HCWS Module Entry Point

This allows running HCWS as a module:
    python -m hcws train --model gpt2
    python -m hcws template my_data.json
"""

from .cli import main

if __name__ == "__main__":
    main() 