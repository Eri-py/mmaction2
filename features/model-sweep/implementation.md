# Multi-Model Video Evaluation Sweep — Implementation Plan

## Summary

Add a standalone CLI script that runs every video in a folder through a set of action-recognition models
declared in a YAML config, and appends each model's top-k predictions to a long-format CSV. Two new files only:
the script itself and a default YAML config populated with the 5 required models (4 generic K400 recognizers +
top-down PoseC3D). No existing library, config, or notebook code changes.

## Approach & Key Decisions

- **New top-level `scripts/` directory**, not `tools/`. CLAUDE.md states `tools/` is "unchanged from
  upstream" — adding a repo-specific script there would violate that. `notebooks/` is reserved for actual
  `.ipynb` notebooks (coding guidelines: "one notebook, one clear purpose"), so a batch CLI script doesn't
  belong there either.
- **Two model "types" in the YAML schema**: `recognizer` (SlowFast/Swin/TimeSformer/VideoMAE — one config +
  one checkpoint, generic `init_recognizer`/`inference_recognizer`) and `skeleton_topdown` (PoseC3D — detector
  config/checkpoint + pose config/checkpoint + classifier config/checkpoint, reusing the
  `detection_inference` → `pose_inference` → `inference_skeleton` pipeline already proven in
  `notebooks/demo_skeleton.ipynb`). This is the minimum split needed to cover both pipelines without
  per-model special-casing in code — everything that varies between the 4 recognizer models is data (config
  path/checkpoint/dataset), not logic.
- **`dataset` replaces the earlier `label_space` idea as a single field**, both in the YAML and the output
  CSV. Its value is the exact `Results[].Dataset` string from that model's own `configs/.../metafile.yml`
  (e.g. `Kinetics-400`, `FineGYM`) — one field serves as both the CSV's label-space column and the key used to
  resolve/validate a checkpoint (below), instead of two redundant fields carrying the same information.
- **Checkpoint is optional for genuine mmaction2 models** (a `recognizer` entry's `checkpoint`, or the
  `skeleton_topdown` entry's `classifier_checkpoint`). `config` is always required and always names one exact
  config file, so resolving an omitted checkpoint is a deterministic lookup, not a "pick the best variant"
  search: find the config's own `metafile.yml` (same directory), find the `Models` entry whose `Config` field
  matches, confirm one of its `Results[].Dataset` values matches the entry's `dataset` (case-insensitive), and
  use that entry's `Weights` URL. This does **not** extend to the PoseC3D pipeline's `detector_checkpoint` /
  `pose_checkpoint` — those configs are vendored from mmdetection/mmpose and deliberately carry no
  `metafile.yml` (CLAUDE.md: adding one would falsely assert mmaction2 trained/benchmarked them), so those two
  fields stay required and explicit.
- **Every checkpoint is cached locally under `checkpoints/` before use**, regardless of whether it came from
  an explicit YAML value or the metafile lookup above. mmengine's own checkpoint loader already persists
  downloads to `~/.cache/torch/hub/checkpoints/` (confirmed during planning — 3 of the 7 checkpoints this
  feature needs were already sitting there from earlier sessions), so this isn't fixing a re-download problem;
  it's meeting the explicit requirement that this repo's own `checkpoints/` folder — its established, visible
  local cache — be checked/populated first, and that a local path (not a URL) is what actually gets passed to
  `init_recognizer` / `detection_inference` / `pose_inference`.
- **No pandas dependency in the script.** The spec's mention of pandas is about how a user later *queries* the
  output CSV, not a requirement on the script itself. The script reads/writes the CSV with the stdlib `csv`
  module, parses the YAML config and any `metafile.yml` files with PyYAML (already an installed transitive
  dependency of `mmengine`/`mmcv`), and downloads checkpoints with `torch.hub.download_url_to_file` (already
  available via the installed `torch`). No new dependencies at all.
- **Outer loop over models, inner loop over videos.** Each model is initialized once (after its checkpoint is
  resolved and locally cached), run against every not-yet-completed video, then released
  (`del model; torch.cuda.empty_cache()`) before moving to the next model. Running multiple models
  concurrently is explicitly out of scope per the spec.
- **Resumability key is `(video basename, model name)`**, read from the existing output CSV (if present)
  before the sweep starts. Every successful video's rows are written and flushed to the CSV immediately after
  that video finishes (not buffered for the whole run), so an interrupted run's completed work is never lost.
- **No auto-selection of config/architecture from a dataset name, and no retry of failed pairs** — both
  explicitly out of scope in the spec. `dataset` only resolves/validates a checkpoint for a config the user
  already named explicitly.

## Out of Scope

Carried forward from the spec: bottom-up PoseC3D, accuracy/ground-truth scoring, any UI/notebook/visualization
of results, multi-GPU or concurrent-model execution, automatic retry of failed pairs, auto-selecting a
config/architecture from just a dataset name, and dataset-based checkpoint resolution for the PoseC3D
detector/pose-estimator checkpoints (those are always explicit; only local caching applies to them).

Additionally deferred by this plan:
- A `scripts/README.md` — not requested, and the script's own `--help` output covers usage.
- Declaring PyYAML as a direct dependency in `pyproject.toml` — it's already an installed transitive
  dependency; adding it explicitly isn't required by the spec.

## Dependencies and Configuration

None new. PyYAML (config/metafile parsing), `torch.hub.download_url_to_file` (checkpoint caching), and all
inference APIs used are already installed in `.venv`. All 5 models' configs, checkpoints, and metafiles already
exist and were re-verified reachable:

- SlowFast: `configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py` (dataset:
  `Kinetics-400`)
- Swin-tiny: `configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py`
  (dataset: `Kinetics-400`)
- TimeSformer: `configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py`
  (dataset: `Kinetics-400`)
- VideoMAE: `configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py` (dataset:
  `Kinetics-400`)
- PoseC3D detector: `configs/person_detector/faster-rcnn_r50_fpn_2x_coco_infer.py` (no metafile — checkpoint
  always explicit)
- PoseC3D pose estimator: `configs/skeleton_coord/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py` (no
  metafile — checkpoint always explicit)
- PoseC3D classifier: `configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb.py` (dataset: `FineGYM`)
- Label maps: `tools/data/kinetics/label_map_k400.txt` (400 classes), `tools/data/skeleton/label_map_gym99.txt`
  (99 classes)

Checkpoint URLs (all re-verified reachable via ranged GET during planning — note this CDN returns 404 on
`HEAD`/`curl -I` but serves real content on `GET`, don't re-check with `-I`):
```
slowfast:      https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth
swin:          https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth
timesformer:   https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth
videomae:      https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth
posec3d cls:   https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth
person det:    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
pose estimator: https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
```

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `scripts/model_sweep.py` | add | CLI script: config/video discovery, checkpoint resolution + local caching, both model runners, resumable CSV writer, main loop | Core deliverable |
| `scripts/model_sweep_config.yaml` | add | Default YAML config listing the 5 required models with real checkpoint URLs and datasets, `top_k: 5` | Fulfills "swappable via config file" requirement and the acceptance criterion referencing "the default 5-model YAML config" |

## Tasks

### Task 1 — Default YAML config

- **Objective:** Define the model-sweep YAML schema and populate it with the 5 required models.
- **Files:** `scripts/model_sweep_config.yaml`
- **Details:** Top level: `top_k: 5` and `models: [...]`. Each entry has `name` (unique id), `type`
  (`recognizer` or `skeleton_topdown`), `dataset`, and `device: cuda:0`. Give every entry in this **default**
  file an explicit `checkpoint` (this file is the reliable, reproducible deliverable the acceptance criteria
  reference — the optional-checkpoint/auto-resolve behavior is implemented and tested in Task 3 via a separate
  temp config, not by omitting anything here).
  - `recognizer` entries (`slowfast`, `swin`, `timesformer`, `videomae`) each need: `config` (path under
    `configs/recognition/...`, relative to repo root), `checkpoint` (URL, see list in "Dependencies and
    Configuration" above), `dataset: Kinetics-400`, `label_map: tools/data/kinetics/label_map_k400.txt`.
  - One `skeleton_topdown` entry (`posec3d`) needs: `detector_config`, `detector_checkpoint`, `pose_config`,
    `pose_checkpoint`, `classifier_config`, `classifier_checkpoint` (all paths/URLs from "Dependencies and
    Configuration" above), `dataset: FineGYM`, `label_map: tools/data/skeleton/label_map_gym99.txt`.
  - All config/label-map paths are relative to the repo root (mirrors how the two demo notebooks reference
    them via `ROOT / "configs/..."`).
- **Success criteria:**
  - `.venv/bin/python -c "import yaml; c = yaml.safe_load(open('scripts/model_sweep_config.yaml')); assert c['top_k'] == 5; assert len(c['models']) == 5"` runs without error.
  - Every `config`/`detector_config`/`pose_config`/`classifier_config`/`label_map` value, resolved relative to
    the repo root, points to a file that actually exists (spot-check with a shell loop or Python).

### Task 2 — Script scaffolding: config loading, video discovery, resumable CSV I/O

- **Objective:** Build the non-model parts of the script: argument parsing, YAML loading, video discovery, and
  the resumable CSV read/append logic — everything needed except checkpoint resolution and actually running
  inference.
- **Files:** `scripts/model_sweep.py`
- **Details:**
  - `argparse` CLI: `--videos-dir` (required), `--config` (default `scripts/model_sweep_config.yaml`),
    `--output` (required, CSV path).
  - `ROOT = Path(__file__).resolve().parents[1]`; resolve every relative path from the YAML against `ROOT`.
  - Video discovery: list files directly inside `--videos-dir` (no recursion) whose suffix, lowercased, is one
    of `.mp4`, `.avi`, `.mov`, `.mkv`.
  - CSV columns, in order: `video_path` (video's basename, not full path — keeps resumability independent of
    where the folder lives), `model_name`, `dataset`, `rank`, `label`, `score`.
  - Resumability: if `--output` already exists, read it with `csv.DictReader` and build a
    `set[(video_path, model_name)]` of already-completed pairs; open it in append mode afterward. If it
    doesn't exist, create it and write the header first.
  - A `logging`-based logger (to stderr) for the failure-logging requirement used in Task 4/5.
  - Leave a `# TODO` marker (or an empty function raising `NotImplementedError`) where the two model runners
    plug into the main per-model loop — Task 4 and Task 5 fill these in.
- **Success criteria:**
  - `.venv/bin/python -c "..."` exercising the discovery function against a temp dir containing one dummy
    `.mp4`, one dummy `.txt`, and a subfolder containing another dummy `.mp4` confirms only the top-level
    `.mp4` is returned.
  - A synthetic round-trip test (write a small CSV by hand with one `(video, model)` row already present,
    then call the resume-loading function against it) confirms that pair is correctly recognized as complete
    and a different pair is not.
  - `.venv/bin/python scripts/model_sweep.py --help` runs and prints usage without error.

### Task 3 — Checkpoint resolution and local caching

- **Objective:** Implement the shared helper both model runners will use to turn a model entry's checkpoint
  field (present or omitted) into a local file path under `checkpoints/`.
- **Files:** `scripts/model_sweep.py`
- **Details:**
  - A metafile-lookup function: given a config path (resolved against `ROOT`) and a `dataset` string, load the
    `metafile.yml` in that config's own directory (parse with PyYAML — see
    `configs/recognition/swin/metafile.yml` or `configs/skeleton/posec3d/metafile.yml` for the shape: a
    `Models` list, each with `Config`, `Results` (a list of dicts with a `Dataset` key), and `Weights`), find
    the entry whose `Config` matches the given path, confirm one of its `Results[].Dataset` values matches
    `dataset` case-insensitively, and return its `Weights` URL. Raise a clear, actionable error (not a bare
    `KeyError`) if no matching config entry or no matching dataset is found.
  - A local-caching function: given a checkpoint value that is either a URL or an already-local path, if it's
    an `http(s)://` URL, compute `checkpoints/<basename of the URL>` (creating the `checkpoints/` directory if
    needed), download it there with `torch.hub.download_url_to_file` only if that file doesn't already exist,
    and return the local path as a `str`; otherwise resolve the given local path against `ROOT` and return
    that as a `str`.
  - A top-level `resolve_checkpoint(checkpoint, config, dataset)` that calls the metafile lookup only when
    `checkpoint` is `None`/absent, then always runs the result through the local-caching function. This is the
    one function both runners call for every checkpoint field they have (recognizer's `checkpoint`; skeleton's
    `detector_checkpoint`, `pose_checkpoint` — both always explicit, so the metafile branch never triggers for
    them — and `classifier_checkpoint`).
- **Success criteria:**
  - Calling the metafile-lookup function with
    `config=configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py`, `dataset="Kinetics-400"`
    returns exactly the SlowFast URL listed in "Dependencies and Configuration" above.
  - Calling it with a mismatched dataset (e.g. `dataset="FineGYM"` for that same SlowFast config) raises a
    clear error instead of returning a wrong URL.
  - Calling `resolve_checkpoint` with the PoseC3D classifier's checkpoint URL (small file, ~8MB) downloads it
    to `checkpoints/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth` and returns that path; calling
    it again immediately afterward returns the same path without re-downloading (verify via the file's mtime
    being unchanged across the two calls).

### Task 4 — Recognizer runner (SlowFast / Swin / TimeSformer / VideoMAE)

- **Objective:** Implement the `recognizer`-type model runner and wire it into the main per-model loop from
  Task 2, so `recognizer` entries in the YAML actually run.
- **Files:** `scripts/model_sweep.py`
- **Details:** Follow the pattern in `notebooks/demo_recognition.ipynb`: resolve the checkpoint via
  `resolve_checkpoint` from Task 3, then `init_recognizer(config, resolved_checkpoint, device=...)` once per
  model entry. Per video, `inference_recognizer(model, str(video_path))`, then
  `pred_result.pred_score.topk(top_k)` for the indices/scores, mapped through the model's label list (loaded
  from the entry's `label_map`). Per CLAUDE.md, `init_recognizer`'s `checkpoint` and `inference_recognizer`'s
  `video` arguments are `str`-only — pass `str(...)` (config paths may stay as `Path`; both APIs accept
  `Union[str, Path, Config]` there). Wrap each video's inference call in `try/except Exception`: on failure,
  log via the Task 2 logger and continue to the next video without writing any rows for that pair; on success,
  append `top_k` rows (rank 1..top_k, `dataset` = the entry's `dataset` value) and flush the CSV immediately.
- **Success criteria:**
  - Run `.venv/bin/python scripts/model_sweep.py --videos-dir videos --config <a temp YAML with only the
    slowfast entry> --output <temp csv>` against the repo's existing `videos/` folder (contains `demo.mp4` and
    `backflip.mp4`).
  - The output CSV has exactly `top_k` (5) rows for each video processed, `dataset` = `Kinetics-400`, and
    `label` values that are real entries from `label_map_k400.txt` (not indices or blanks).
  - `checkpoints/` now contains the SlowFast checkpoint file.

### Task 5 — Skeleton (top-down PoseC3D) runner

- **Objective:** Implement the `skeleton_topdown`-type model runner and wire it into the main per-model loop.
- **Files:** `scripts/model_sweep.py`
- **Details:** Follow the pattern in `notebooks/demo_skeleton.ipynb`: resolve all three checkpoints via
  `resolve_checkpoint` from Task 3 (detector and pose checkpoints are always explicit in the entry, so this
  just runs the local-caching step for them; the classifier checkpoint may be resolved via metafile). Once per
  model entry, call `init_recognizer(classifier_config, resolved_classifier_checkpoint, device=...)` for the
  PoseC3D classifier (the detector and pose estimator are re-initialized per `detection_inference`/
  `pose_inference` call, matching the notebook — they don't expose a separate init step). Per video:
  `frame_extract(str(video_path), short_side=480, out_dir=<a fresh tempfile.TemporaryDirectory()>)` to get
  `frame_paths, frames`; then `detection_inference(detector_config, resolved_detector_checkpoint, frame_paths,
  det_score_thr=0.9, det_cat_id=0, device=...)`, then `pose_inference(pose_config,
  resolved_pose_checkpoint, frame_paths, det_results, device=...)`, then `inference_skeleton(model,
  pose_results, (h, w))` where `h, w, _ = frames[0].shape`. Extract top-k the same way as Task 4 against the
  GYM99 label list. Clean up the temp frame directory after each video (use the `with
  tempfile.TemporaryDirectory()` form, unlike the notebook's explicit-cleanup variant, since there's no later
  cell that needs the frames to persist here). Same try/except-log-and-skip behavior as Task 4, scoped per
  video.
- **Success criteria:**
  - Run the script with a temp YAML containing only the `posec3d` entry against `videos/` (`backflip.mp4` in
    particular — it's the one the existing notebook uses for this pipeline).
  - The output CSV has exactly `top_k` (5) rows for `backflip.mp4`, `dataset` = `FineGYM`, and `label` values
    that are real entries from `label_map_gym99.txt`.
  - `checkpoints/` now also contains the person-detector and pose-estimator checkpoint files.

### Task 6 — Full sweep integration run

- **Objective:** Run the complete script with all 5 models against a small real video set and confirm the
  output matches every acceptance criterion in the spec.
- **Files:** none changed
- **Success criteria:**
  - `.venv/bin/python scripts/model_sweep.py --videos-dir videos --config scripts/model_sweep_config.yaml
    --output <temp csv>` completes without crashing.
  - The output CSV contains `top_k` (5) rows for each of the 5 models × 2 videos (`demo.mp4`, `backflip.mp4`)
    that succeed — 50 rows if all succeed — with `rank` running 1..5 per `(video_path, model_name)` pair.
  - `dataset` is `Kinetics-400` for the 4 recognizer models' rows and `FineGYM` for `posec3d`'s rows.
  - `checkpoints/` contains all 7 checkpoint files this feature uses (4 recognizers + detector + pose
    estimator + PoseC3D classifier).
  - Filtering the CSV by `model_name` (e.g. with the stdlib `csv` module or a one-off `.venv/bin/python -c
    "import pandas as pd; ..."` check, matching how a user would actually query it) isolates each model's
    predictions correctly.

### Task 7 — Regression test run: error handling, resumability, and checkpoint reuse

- **Objective:** Confirm the resumability, per-video error-handling, and checkpoint-caching requirements from
  the spec, using the output produced in Task 6.
- **Files:** none changed
- **Success criteria:**
  - Re-running the exact same command from Task 6 against the same `--output` CSV completes, and the CSV's
    row count and content are unchanged (no duplicate rows, nothing reprocessed) — confirm via a before/after
    row-count and content diff.
  - That same re-run does not re-download any checkpoint — confirm via each checkpoint file's mtime under
    `checkpoints/` being unchanged before and after the re-run.
  - Copy `videos/demo.mp4` to a temp videos folder, truncate the copy to a handful of bytes (simulating a
    corrupt file), add one working video alongside it, and run the script (single-model config is enough)
    against that folder with a fresh `--output`: the run completes without crashing, no row is written for
    the corrupt file for that model, a failure is visible in the script's logged output, and the working
    video's rows are present and correct.
