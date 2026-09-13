import argparse

from ..common import ROOT

DEFAULT_CONFIG = ROOT / 'scripts' / 'video_eval_config.yaml'


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
        default=str(DEFAULT_CONFIG),
        help='Path to the video-eval YAML config '
        f'(default: {DEFAULT_CONFIG}).')
    parser.add_argument(
        '--output', required=True, help='Path to the results CSV.')
    return parser.parse_args()
