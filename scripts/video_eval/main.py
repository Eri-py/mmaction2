from .common import logger
from .inputs import discover_videos, load_config, parse_args
from .results import load_completed_pairs, open_output_csv
from .runners import RUNNERS


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
            logger.info('=== %s (%s) ===', model_entry['name'],
                        model_entry['type'])
            try:
                runner(model_entry, videos, top_k, completed, writer, csv_file)
            except Exception:
                logger.exception(
                    'Failed to run model %r, skipping to next model',
                    model_entry['name'])
                continue
    finally:
        csv_file.close()
