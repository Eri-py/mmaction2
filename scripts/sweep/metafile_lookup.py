from pathlib import Path

import yaml

from .common import ROOT


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
