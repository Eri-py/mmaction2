import argparse


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
