"""Shared paths, constants, and logging setup for the video-eval CLI."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS_DIR = ROOT / 'checkpoints'

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# video_path is a basename only, so resume works regardless of --videos-dir.
CSV_FIELDS = ['video_path', 'model_name', 'dataset', 'rank', 'label', 'score']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger('video_eval')
