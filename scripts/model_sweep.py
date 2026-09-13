#!/usr/bin/env python
"""Run every video in a folder through a set of action-recognition models
declared in a YAML config, and append each model's top-k predictions to a
long-format CSV. See features/model-sweep/spec.md for the full requirements.
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

CSV_FIELDS = ['video_path', 'model_name', 'dataset', 'rank', 'label', 'score']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger('model_sweep')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a folder of videos through a YAML-configured sweep '
        'of action-recognition models and append results to a CSV.')
    parser.add_argument(
        '--videos-dir',
        required=True,
        help='Folder of input videos (scanned non-recursively).')
    parser.add_argument(
        '--config',
        default='scripts/model_sweep_config.yaml',
        help='Path to the model-sweep YAML config.')
    parser.add_argument(
        '--output', required=True, help='Path to the results CSV.')
    return parser.parse_args()


def load_config(config_path):
    """Load and parse the model-sweep YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_videos(videos_dir):
    """List video files directly inside videos_dir (no recursion)."""
    videos_dir = Path(videos_dir)
    return sorted(p for p in videos_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def load_completed_pairs(output_path):
    """Return the set of (video_path, model_name) pairs already in the CSV."""
    output_path = Path(output_path)
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row['video_path'], row['model_name']))
    return completed


def open_output_csv(output_path):
    """Open the output CSV for appending, writing the header if it's new."""
    output_path = Path(output_path)
    is_new = not output_path.exists()
    f = open(output_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
        f.flush()
    return f, writer


def run_recognizer(model_entry, videos, top_k, completed, writer, csv_file):
    """Run a `recognizer`-type model entry over videos. Filled in by Task 4."""
    raise NotImplementedError


def run_skeleton_topdown(model_entry, videos, top_k, completed, writer,
                         csv_file):
    """Run the `skeleton_topdown` model entry. Filled in by Task 5."""
    raise NotImplementedError


RUNNERS = {
    'recognizer': run_recognizer,
    'skeleton_topdown': run_skeleton_topdown,
}


def main():
    args = parse_args()
    config = load_config(args.config)
    top_k = config.get('top_k', 5)
    videos = discover_videos(args.videos_dir)
    logger.info('Found %d video(s) in %s', len(videos), args.videos_dir)

    completed = load_completed_pairs(args.output)
    csv_file, writer = open_output_csv(args.output)

    try:
        for model_entry in config['models']:
            runner = RUNNERS.get(model_entry['type'])
            if runner is None:
                logger.error('Unknown model type %r for model %r, skipping',
                             model_entry['type'], model_entry['name'])
                continue
            # TODO(Task 4/5): runner() is not yet implemented for either type.
            runner(model_entry, videos, top_k, completed, writer, csv_file)
    finally:
        csv_file.close()


if __name__ == '__main__':
    main()
