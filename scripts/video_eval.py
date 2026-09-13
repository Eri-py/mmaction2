#!/usr/bin/env python
"""Entrypoint for the video-eval CLI. See scripts/video_eval/ for the
implementation (one function per file). Run as:
    python scripts/video_eval.py --videos-dir ... --config ... --output ...
from the repo root.
"""
from video_eval.main import main

if __name__ == '__main__':
    main()
