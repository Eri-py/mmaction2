from mmaction.apis import inference_recognizer, init_recognizer
from ..checkpoints import resolve_checkpoint
from ..common import ROOT, logger


def run_recognizer(model_entry, videos, top_k, completed, writer, csv_file):
    """Run a `recognizer`-type model entry (SlowFast/Swin/TimeSformer/
    VideoMAE) over videos, appending top_k prediction rows per video."""
    config_path = ROOT / model_entry['config']
    checkpoint = resolve_checkpoint(
        model_entry.get('checkpoint'), config_path, model_entry['dataset'])
    model = init_recognizer(
        config_path, checkpoint, device=model_entry['device'])
    labels = (ROOT / model_entry['label_map']).read_text().splitlines()

    pending = [
        video_path for video_path in videos
        if (video_path.name, model_entry['name']) not in completed
    ]
    logger.info(
        'Starting model %r: %d video(s) found, %d already done, '
        '%d to process', model_entry['name'], len(videos),
        len(videos) - len(pending), len(pending))

    succeeded = 0
    failed = 0
    for i, video_path in enumerate(pending, start=1):
        try:
            pred_result = inference_recognizer(model, str(video_path))
            topk = pred_result.pred_score.topk(top_k)
        except Exception:
            logger.exception('[%s] %d/%d FAILED: %s', model_entry['name'], i,
                             len(pending), video_path.name)
            failed += 1
            continue
        rows = [{
            'video_path': video_path.name,
            'model_name': model_entry['name'],
            'dataset': model_entry['dataset'],
            'rank': rank,
            'label': labels[idx],
            'score': score,
        } for rank, (idx, score) in enumerate(
            zip(topk.indices.tolist(), topk.values.tolist()), start=1)]
        writer.writerows(rows)
        csv_file.flush()
        logger.info('[%s] %d/%d %s -> %s (%.4f)', model_entry['name'], i,
                    len(pending), video_path.name, rows[0]['label'],
                    rows[0]['score'])
        succeeded += 1

    logger.info('Finished model %r: %d succeeded, %d failed',
                model_entry['name'], succeeded, failed)
