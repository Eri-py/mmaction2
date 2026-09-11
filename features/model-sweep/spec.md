# Multi-Model Video Evaluation Sweep

## Overview

A standalone script that runs every video in a folder (1000+ videos, no ground-truth labels) through 5 action
recognition models and records each model's own top-k predictions in a single results table that can be
filtered per model afterward. The set of models to run is defined in a config file, not hardcoded, so models
or their parameters can be changed without touching script code.

## Requirements

- The script accepts an input folder of videos and a YAML config file describing which models to run, and
  produces one CSV results file.
- The 5 models to support, all using configs/checkpoints already present or referenced in this repo:
  1. SlowFast (`configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py`)
  2. Swin-tiny (`configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py`)
  3. TimeSformer divST (`configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py`)
  4. VideoMAE-base (`configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py`)
  5. PoseC3D, top-down only (person detector `configs/person_detector/faster-rcnn_r50_fpn_2x_coco_infer.py` →
     pose estimator `configs/skeleton_coord/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py` → classifier
     `configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb.py`). Bottom-up PoseC3D is out of scope (see
     below).
  - Models 1-4 use Kinetics-400 labels (`tools/data/kinetics/label_map_k400.txt`, 400 classes) and the generic
    `init_recognizer`/`inference_recognizer` API with no per-model special-casing required (verified: all four
    configs already set `average_clips='prob'` and `test_cfg=dict(type='TestLoop')`, directly or via their
    `_base_` model config).
  - Model 5 uses GYM99 labels (`tools/data/skeleton/label_map_gym99.txt`, 99 classes) via the same
    person-detection → pose-estimation → `inference_skeleton` pipeline already built in
    `notebooks/demo_skeleton.ipynb`.
- The YAML config file lists the models to run; each model entry specifies at minimum: a name/identifier, the
  mmaction2 config path (plus, for PoseC3D, the detector and pose-estimator config paths), the checkpoint
  (local path or URL), and the device to run it on.
- The config file also has a single global `top_k` setting (default: 5) applied to every model, controlling how
  many top predictions are recorded per video per model.
- The video folder is scanned non-recursively for files with common video extensions (e.g. `.mp4`, `.avi`,
  `.mov`, `.mkv`).
- Results are written to a single long-format CSV: one row per (video, model, rank), with columns identifying
  the video, the model, the label space (K400 vs GYM99), the rank, the predicted label, and its score. This
  keeps the table filterable per model with pandas without forcing all 5 models into a rigid wide schema.
- If a given (video, model) pair fails to process (corrupt file, decode error, OOM, etc.), the failure is
  logged and the sweep continues with the next item; no row is written to the results CSV for that pair, and no
  other (video, model) pairs are affected.
- Re-running the script against an existing output CSV skips any (video, model) pair already present in that
  file and only appends results for pairs not yet recorded, so an interrupted run (expected over 1000+ videos ×
  5 models) can be resumed without redoing completed work or duplicating rows.

## Out of Scope

- Bottom-up PoseC3D (associative-embedding pose estimation without a person detector) — dropped from this
  feature entirely, not deferred. Only the top-down PoseC3D pipeline is included.
- Any accuracy/ground-truth scoring — there are no labels for the input videos; only each model's own
  predictions are recorded.
- A UI, notebook, or visualization of results — output is the CSV only.
- Parallelizing inference across multiple GPUs/processes, or running multiple models concurrently.
- Retrying failed (video, model) pairs automatically — a failure is simply skipped and logged; re-running the
  whole script is the retry mechanism, and skipped pairs are only retried if they're still absent from the
  output CSV (i.e., a permanently-corrupt video will keep failing on every re-run unless removed from the input
  folder).
- Validating or fetching checkpoints ahead of time — the 4 recognition checkpoints and the skeleton pipeline's
  checkpoints are already confirmed reachable; the script does not need its own pre-flight download/validation
  step.

## Acceptance Criteria

- Given a folder of videos and the default 5-model YAML config, when the script is run, then the output CSV
  contains rows for every (video, model, rank) combination that succeeded, with `rank` values from 1 to the
  configured `top_k` for each (video, model) pair.
- Given the YAML config, when a model's checkpoint or config path is changed, or a model is added/removed from
  the list, then no script code needs to change for the new sweep to run.
- Given the config's `top_k` value (default 5), when the sweep runs, then every successful (video, model) pair
  produces exactly that many ranked rows.
- Given a video that fails to process for one model, when the sweep continues, then no row is written for that
  (video, model) pair, the failure is logged, and all other (video, model) pairs still complete normally.
- Given an output CSV from a previous, interrupted run, when the script is re-run with the same input folder,
  config, and output path, then only (video, model) pairs missing from the existing CSV are processed, and
  previously-recorded rows are left unchanged (not duplicated or reprocessed).
- Given the results CSV, when filtered by `model_name` with pandas, then each model's predictions can be
  inspected independently, including which label space (K400 or GYM99) they came from.
- Given a non-recursive scan of the input folder, when it contains subfolders, then only videos directly inside
  the given folder (not nested ones) are included in the sweep.
