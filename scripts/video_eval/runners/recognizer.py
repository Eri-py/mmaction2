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
