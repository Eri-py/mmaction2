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
from urllib.parse import urlparse

import torch
import yaml

from mmaction.apis import inference_recognizer, init_recognizer

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS_DIR = ROOT / 'checkpoints'

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


def lookup_checkpoint_in_metafile(config, dataset):
    """Find the Weights URL for `config` trained on `dataset` in its
    directory's own metafile.yml (mmaction2 model-zoo format)."""
    config = Path(config).resolve()
    metafile_path = config.parent / 'metafile.yml'
    if not metafile_path.exists():
        raise FileNotFoundError(
            f'No metafile.yml found next to config {config} - a checkpoint '
            'must be given explicitly for configs without mmaction2 model '
            'metadata.')
    with open(metafile_path, 'r') as f:
        metafile = yaml.safe_load(f)

    config_rel = config.relative_to(ROOT).as_posix()
    matches = [
        m for m in metafile.get('Models', [])
        if Path(m['Config']).as_posix() == config_rel
    ]
    if not matches:
        raise ValueError(
            f'No entry for config {config_rel!r} found in {metafile_path}.')

    for model_meta in matches:
        datasets = [r['Dataset'] for r in model_meta.get('Results', [])]
        if any(d.lower() == dataset.lower() for d in datasets):
            return model_meta['Weights']

    available = sorted(
        {r['Dataset']
         for m in matches
         for r in m.get('Results', [])})
    raise ValueError(
        f'No result for dataset {dataset!r} found for config {config_rel!r} '
        f'in {metafile_path}. Available dataset(s): {available}.')


def cache_checkpoint_locally(checkpoint):
    """Return a local path for `checkpoint`, downloading it into
    checkpoints/ first if it's a URL not already cached there."""
    parsed = urlparse(str(checkpoint))
    if parsed.scheme in ('http', 'https'):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        local_path = CHECKPOINTS_DIR / Path(parsed.path).name
        if not local_path.exists():
            logger.info('Downloading checkpoint %s -> %s', checkpoint,
                        local_path)
            torch.hub.download_url_to_file(str(checkpoint), str(local_path))
        return str(local_path)
    return str((ROOT / checkpoint).resolve())


def resolve_checkpoint(checkpoint, config, dataset):
    """Turn a model entry's checkpoint field (present or omitted) into a
    local file path under checkpoints/, resolving it from the config's own
    metafile.yml first if omitted."""
    if checkpoint is None:
        checkpoint = lookup_checkpoint_in_metafile(config, dataset)
    return cache_checkpoint_locally(checkpoint)


def run_recognizer(model_entry, videos, top_k, completed, writer, csv_file):
    """Run a `recognizer`-type model entry (SlowFast/Swin/TimeSformer/
    VideoMAE) over videos, appending top_k prediction rows per video."""
    config_path = ROOT / model_entry['config']
    checkpoint = resolve_checkpoint(
        model_entry.get('checkpoint'), config_path, model_entry['dataset'])
    model = init_recognizer(
        config_path, checkpoint, device=model_entry['device'])
    labels = (ROOT / model_entry['label_map']).read_text().splitlines()

    for video_path in videos:
        key = (video_path.name, model_entry['name'])
        if key in completed:
            continue
        try:
            pred_result = inference_recognizer(model, str(video_path))
            topk = pred_result.pred_score.topk(top_k)
        except Exception:
            logger.exception('Failed to run model %r on video %r, skipping',
                             model_entry['name'], video_path.name)
            continue
        for rank, (idx, score) in enumerate(
                zip(topk.indices.tolist(), topk.values.tolist()), start=1):
            writer.writerow({
                'video_path': video_path.name,
                'model_name': model_entry['name'],
                'dataset': model_entry['dataset'],
                'rank': rank,
                'label': labels[idx],
                'score': score,
            })
        csv_file.flush()


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
