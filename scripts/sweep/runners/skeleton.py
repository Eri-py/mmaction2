import tempfile

import torch
from mmdet.apis import init_detector
from mmpose.apis import init_model as init_pose_model

from mmaction.apis import (detection_inference, inference_skeleton,
                           init_recognizer, pose_inference)
from mmaction.utils import frame_extract
from ..common import ROOT, logger
from ..resolve_checkpoint import resolve_checkpoint


def run_skeleton_topdown(model_entry, videos, top_k, completed, writer,
                         csv_file):
    """Run the `skeleton_topdown` model entry (person detection -> pose
    estimation -> PoseC3D classification), appending top_k rows per video."""
    device = model_entry['device']
    detector_config = ROOT / model_entry['detector_config']
    detector_checkpoint = resolve_checkpoint(
        model_entry.get('detector_checkpoint'), detector_config,
        model_entry['dataset'])
    pose_config = ROOT / model_entry['pose_config']
    pose_checkpoint = resolve_checkpoint(
        model_entry.get('pose_checkpoint'), pose_config,
        model_entry['dataset'])
    classifier_config = ROOT / model_entry['classifier_config']
    classifier_checkpoint = resolve_checkpoint(
        model_entry.get('classifier_checkpoint'), classifier_config,
        model_entry['dataset'])
    model = init_recognizer(
        classifier_config, classifier_checkpoint, device=device)
    detector_model = init_detector(
        detector_config, detector_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    labels = (ROOT / model_entry['label_map']).read_text().splitlines()

    for video_path in videos:
        key = (video_path.name, model_entry['name'])
        if key in completed:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                frame_paths, frames = frame_extract(
                    str(video_path), short_side=480, out_dir=tmp_dir)
                h, w, _ = frames[0].shape
                det_results, _ = detection_inference(
                    detector_model,
                    None,
                    frame_paths,
                    det_score_thr=model_entry.get('det_score_thr', 0.9),
                    det_cat_id=model_entry.get('det_cat_id', 0),
                    device=device)
                torch.cuda.empty_cache()
                pose_results, _ = pose_inference(
                    pose_model, None, frame_paths, det_results, device=device)
                torch.cuda.empty_cache()
                pred_result = inference_skeleton(model, pose_results, (h, w))
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
