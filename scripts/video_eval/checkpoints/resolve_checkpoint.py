from .checkpoint_cache import cache_checkpoint_locally
from .metafile_lookup import lookup_checkpoint_in_metafile


def resolve_checkpoint(checkpoint, config, dataset):
    """Turn a model entry's checkpoint field (present or omitted) into a
    local file path under checkpoints/, resolving it from the config's own
    metafile.yml first if omitted."""
    if checkpoint is None:
        checkpoint = lookup_checkpoint_in_metafile(config, dataset)
    return cache_checkpoint_locally(checkpoint)
