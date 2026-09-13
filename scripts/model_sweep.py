#!/usr/bin/env python
"""Entrypoint for the model-sweep CLI. See scripts/sweep/ for the
implementation (one function per file). Run as:
    python scripts/model_sweep.py --videos-dir ... --config ... --output ...
from the repo root.
"""
from sweep.main import main

if __name__ == '__main__':
    main()
