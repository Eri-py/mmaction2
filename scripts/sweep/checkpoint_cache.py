from pathlib import Path
from urllib.parse import urlparse

import torch

from .common import CHECKPOINTS_DIR, ROOT, logger


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
