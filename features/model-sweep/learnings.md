# Learnings — model-sweep feature

## Task 1 — Default YAML config

- All 9 distinct config/label-map paths listed in the plan (4 recognizer configs, 2 label maps, and the 3
  PoseC3D pipeline configs: detector/pose/classifier) already exist in the repo exactly at the paths given —
  no path corrections were needed, no `scripts/` directory existed yet (created by this task).
- Kept the YAML flat (list of dicts under `models`, each carrying its own `type` tag) rather than nesting by
  type, since Task 2+ will likely just iterate `for m in config['models']: dispatch on m['type']` — a flat
  list keeps that dispatch simple and matches the plan's schema description literally.
- Did not add a `metafile.yml`-lookup sanity check here (e.g. verifying the checkpoint URL actually matches
  what each config's own metafile lists) — out of scope for Task 1; that resolution/validation logic is
  Task 3's job. This task only needed the checkpoint URLs given directly in the task prompt.

## Task 2 — Script scaffolding

- `csv.DictWriter.writeheader()` on a freshly opened file needs an explicit `f.flush()` if the file handle is
  going to be reused across an `open_output_csv()` call for the whole run — `open(..., 'a')` buffers, and a
  concurrent read of the file before the process exits would otherwise see a truncated/missing header. Flushed
  right after `writeheader()` here to keep the on-disk file always in a valid readable state.
- yapf's own formatting of a wrapped `logger.error(...)` call and a wrapped `def run_skeleton_topdown(...)`
  signature disagreed with flake8's default continuation-indent expectation (E127) when the args were
  hand-indented to align under the opening paren by one extra/fewer space — ran `yapf --diff` first and matched
  its exact indentation rather than guessing, since flake8 and yapf both need to pass and yapf's output is the
  authoritative target here.
- `RUNNERS` dispatch dict (`{'recognizer': run_recognizer, 'skeleton_topdown': run_skeleton_topdown}`) plus
  both runner stubs raising `NotImplementedError` is the seam Task 4/5 will fill — `main()` never special-cases
  the model type itself, it only looks up `model_entry['type']` in `RUNNERS`.
- Resumability set is keyed on `(video_path, model_name)` where `video_path` is already just the basename
  (`Path.name`), matching what actually gets written to the CSV — `discover_videos` returns full `Path` objects
  so the future runner code (Task 4/5) must use `.name` when writing rows and when checking `completed`, not
  the full path, or resume matching will silently never hit.

## Task 3 — Checkpoint resolution and local caching

- Metafile `Models[].Config` values are already repo-root-relative strings (e.g.
  `configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py`), so the lookup function
  compares `Path(m['Config']).as_posix()` against the input config resolved-then-rebased to
  `relative_to(ROOT).as_posix()`, rather than comparing absolute paths — keeps the match working regardless of
  how the caller spelled the input path (relative to cwd, absolute, etc.), as long as it's actually inside the
  repo.
- Some metafiles (e.g. slowfast's) have multiple `Models` entries whose `Config` differs but could share a
  directory's single `metafile.yml`; filtered to matches on `Config` first, *then* checked `Results[].Dataset`
  within only those matches — checking dataset across all `Models` in the file first would have let an
  unrelated config's dataset silently satisfy the match.
- `torch.hub.download_url_to_file` prints its own tqdm progress bar straight to stderr with no way to silence
  it via the function's own kwargs (no `progress=False` in the installed torch version's signature actually
  suppresses it here) — left as-is since this is a one-off sweep script run interactively, not a concern worth
  extra plumbing for.
- Used `urllib.parse.urlparse(...).scheme` to detect `http(s)://` rather than `str.startswith('http')` — mainly
  so the basename comes from `Path(parsed.path).name` (the URL's path component only), immune to a query string
  or fragment ever being appended to a checkpoint URL down the line.
- Verified end-to-end against the real network: `lookup_checkpoint_in_metafile` on the SlowFast K400 config
  returns exactly the URL in the plan; the same call with `dataset="FineGYM"` raises `ValueError` naming the
  actually-available dataset(s) rather than a bare `KeyError`; `resolve_checkpoint` on the PoseC3D
  gym-limb classifier URL downloads once to `checkpoints/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth`
  (confirmed via file's own mtime) and a second immediate call returns the identical path with an unchanged
  mtime (no re-download).

## Task 4 — Recognizer runner

- `pred_result.pred_score.topk(top_k)` (a `torch.Tensor.topk` call, unlike the notebook's manual
  sort-by-`itemgetter` approach) returns a named tuple with `.indices`/`.values`, already sorted descending —
  simpler than porting the notebook's `score_sorted = sorted(...)` pattern verbatim, and avoids the
  `enumerate(range(len(pred_scores)))` indirection the notebook uses only because it hadn't discovered `topk`
  yet at that point in the tutorial. `.tolist()` on each converts cleanly to plain Python ints/floats for the
  CSV writer (rank is assigned via `enumerate(..., start=1)`, not read off the tensor).
- The labels file has no other content to strip per-line beyond the newline `.splitlines()` already removes, so
  `label_map.read_text().splitlines()` is equivalent to and shorter than the notebook's
  `[line.strip() for line in label.read_text().splitlines()]` for `label_map_k400.txt` specifically (no
  leading/trailing whitespace on any line) — used the shorter form here.
- Confirmed end-to-end against real inference (SlowFast, K400, both repo videos): `backflip.mp4` top-1 is
  "gymnastics tumbling" (0.61) and `demo.mp4` top-1 is "arm wrestling" (1.0) — both plausible for their content,
  giving actual confidence the pipeline (not just the plumbing) works, not just that rows got written.
- Confirmed resume path from Task 2 works correctly for this runner specifically: re-running the exact same
  command against an already-populated output CSV logs no download (checkpoint already cached) and appends zero
  new rows — `load_completed_pairs` keys on `(video_path.name, model_name)` exactly as `run_recognizer` writes
  them, so the two line up.
- `init_recognizer`'s `config` argument does NOT need `str(...)` despite `checkpoint` and video needing it —
  passing a `pathlib.Path` config straight through worked fine in practice (matches CLAUDE.md's note that only
  `checkpoint`/`video`/`out_path` are `str`-only, not every path-shaped argument in these APIs).

## Task 5 — Skeleton (top-down PoseC3D) runner

- `detection_inference`/`pose_inference` accept `Path` configs and `resolve_checkpoint`'s plain-`str` return
  directly (matching their real signatures noted in the task prompt) — no extra `str(...)`/`Path(...)` wrapping
  needed beyond what `resolve_checkpoint` already returns, keeping this runner's checkpoint handling identical
  in shape to `run_recognizer`'s single-checkpoint case, just called three times.
- Used the `with tempfile.TemporaryDirectory() as tmp_dir:` form wrapping the *entire* per-video try block
  (frame extraction through `inference_skeleton`), not just `frame_extract` — the frames are only needed
  transiently for that one video's detection/pose/classification chain, so keeping the whole chain inside the
  `with` means the directory is guaranteed cleaned up per video even on a mid-video exception, without a
  separate `finally`.
- Verified end-to-end against real inference (posec3d/FineGYM, `videos/` folder with both `backflip.mp4` and
  `demo.mp4`): exactly 5 ranked rows per video, all 5 label strings for both videos found verbatim (via `grep
  -F`) in `tools/data/skeleton/label_map_gym99.txt` — confirms the label-index mapping is correct, not just
  that indices were produced. `backflip.mp4` top-1 is an uneven-bars/double-salto label at only 0.117
  confidence (GYM99's labels are fine-grained gymnastics apparatus moves, so low top-1 confidence on a video
  that isn't actually gymnastics footage is expected, unlike the K400 recognizers' higher-confidence top-1s in
  Task 4).
- This run downloaded the person-detector (~160MB) and pose-estimator (~110MB) checkpoints for real into
  `checkpoints/` for the first time in this feature (the PoseC3D classifier checkpoint was already cached from
  Task 3's own verification) — confirms `resolve_checkpoint`'s local-caching path works unchanged for
  checkpoints that are always-explicit (never go through the metafile-lookup branch), not just for the
  metafile-resolved case Task 3 exercised.

## Task 6 — Full sweep integration run

- First real run of all 5 models through the actual `scripts/model_sweep.py` entrypoint (previous tasks each
  used a single-model temp YAML). Command: `.venv/bin/python scripts/model_sweep.py --videos-dir videos
  --config scripts/model_sweep_config.yaml --output <temp csv>`. Completed cleanly end to end: 23:44:46 ->
  00:03:59 (~19m13s wall clock), zero exceptions/tracebacks, zero rows skipped for either video across all 5
  models. Output CSV: exactly 50 rows (5 models x 2 videos x top_k=5), `rank` running 1..5 for every
  `(video_path, model_name)` pair (verified via pandas groupby), `dataset` correctly `Kinetics-400` for the 4
  recognizer rows and `FineGYM` for all 10 `posec3d` rows, and every single `label` value found verbatim in its
  matching label map file (0 mismatches checked programmatically against all 50 rows, not spot-checked).
  `checkpoints/` ended up with all 7 required files (plus one unrelated pre-existing TSN checkpoint from an
  earlier, unrelated session — harmless, not part of this feature).
- **Checkpoint download time, not model inference, dominates a cold-cache run.** Of the ~19 minutes total, the
  three new downloads (Swin ~127MB, TimeSformer ~486MB, VideoMAE ~173MB) accounted for the overwhelming
  majority: Swin took ~1.5 min, TimeSformer alone took ~14.7 min (its 486MB is by far the largest of the 7
  checkpoints this feature uses, and this network's throughput to the OpenMMLab CDN dipped as low as
  ~1.3MB/s partway through), VideoMAE took ~2 min. All 5 models' actual inference (load + 2-video forward pass
  each, plus the full detector->pose->classifier pipeline for posec3d) took only the remaining ~1 minute
  combined. Worth knowing for anyone estimating a real 1000+-video sweep: the per-video cost that actually
  scales with video count is small relative to the one-time checkpoint-download cost, so time estimates for a
  full run should be based on the post-warm-cache per-video rate, not a cold-cache trial run's average.
  RTX 5070 GPU memory/utilization were not a bottleneck at any point (headroom confirmed before the run: 1.3GB/
  12GB used, 0% utilization from any other process).
- **First real appearance of `ResourceWarning: unclosed file <...> Image.open(img).size` (PIL), one per frame,
  during the posec3d pipeline's pose-estimation stage** — 34 occurrences in this run's log, all originating from
  mmpose's own inference code computing image size via `Image.open(img).size` without a context manager (not
  this feature's code; `run_skeleton_topdown` never opens a PIL Image directly). Cosmetic only — stderr noise,
  no effect on correctness, no leaked file descriptor problem observed (frames live in a short-lived
  `tempfile.TemporaryDirectory()` cleaned up per video regardless). Not something to patch here per CLAUDE.md's
  guidance on minimal, localized fixes only for things actually broken; noting it so a future session doesn't
  mistake it for a new bug in this feature's own code.
- Confirms the full acceptance-criteria set from spec.md end to end for the first time with all 5 models
  together: swappable-by-config (no code changed to run this), fixed `top_k` per pair, correct
  dataset/label-space column per model, and per-model CSV filtering via pandas (`df[df.model_name==...]`)
  isolating each model's predictions with correct columns.

## Task 7 — Regression test run: error handling, resumability, checkpoint reuse

- Warm-cache full 5-model x 2-video run (all 7 checkpoints already cached from Tasks 3-6) took ~66s wall
  clock, matching Task 6's ~68s warm-cache estimate almost exactly — confirms cold-cache download time, not
  inference, was the entire story behind Task 6's ~19-minute run, per that task's own learnings entry.
- Re-running the identical command against the same populated `--output` CSV is fast but **not free**: it
  still took ~5.5s, not near-zero, because `run_recognizer`/`run_skeleton_topdown` call `resolve_checkpoint`
  and `init_recognizer` (and, for skeleton_topdown, load the detector/pose/classifier) *before* the
  per-video `if key in completed: continue` check — every model still gets fully loaded onto the GPU even
  when every video for it is already done. Confirmed via stderr on the resumed run: zero "Downloading
  checkpoint" lines and zero tracebacks, only the expected "Found N video(s)" info line and an unrelated
  `torch.meshgrid` deprecation warning — so the resume behavior itself is correct, it just isn't
  instantaneous. Worth knowing for anyone tuning a 1000+-video sweep's resume-after-interruption latency
  expectations: the fixed per-model load cost recurs on every invocation regardless of how much/little work
  is actually left to do.
- Verified resumability is exact, not just "close enough": re-run's output CSV was byte-for-byte identical
  (matching MD5) to the pre-re-run copy, `diff` empty, still exactly 50 data rows (51 lines incl. header).
  All 8 files under `checkpoints/` (7 required + 1 unrelated pre-existing TSN checkpoint) had bit-identical
  `stat -c %Y` mtimes before and after the re-run.
- Corrupt-file test (single-model `slowfast` config, a byte-truncated 20-byte copy of `backflip.mp4` plus an
  untouched copy of `demo.mp4` in a fresh temp folder): run exited 0, produced exactly 5 rows (only for the
  working video, `rank` 1-5), and logged exactly one `[ERROR] Failed to run model 'slowfast' on video
  'corrupt.mp4', skipping` line via `logger.exception` with a full traceback rooted in
  `inference_recognizer` failing to decode the truncated file — decode failure surfaces as a normal Python
  exception from deep inside the recognizer call, not a crash or hang, so the existing blanket
  `except Exception` in both runners' per-video try block is sufficient without any special-casing for
  corrupt/undecodable video files specifically. The working video's top-1 prediction ("arm wrestling",
  1.0) matched Task 4's own recorded result for `demo.mp4` exactly, confirming the surviving row is
  actually correct, not just present.
- No code or config in the repo needed to change for this task (confirmed via `git status`/`git diff
  --stat` showing only the pre-existing `progress.md` edit) — this was a pure verification task against the
  already-complete Task 6 implementation, as the plan specified.

## Fix S1 — PoseC3D runner rebuilding detector/pose models per video

- Read `mmaction/apis/inference.py:171-234` (`detection_inference`) and `:237-279` (`pose_inference`)
  directly to confirm the review finding before touching code. Both functions branch on
  `isinstance(det_config/pose_config, nn.Module)`: if it's already a module, that object is used as-is and
  `init_detector`/`init_model` is never called; otherwise they call `init_detector(config=det_config,
  checkpoint=det_checkpoint, device=device)` / `init_model(pose_config, pose_checkpoint, device)` themselves.
  Critically, on the `nn.Module` branch the `det_checkpoint`/`pose_checkpoint` positional parameters are
  never referenced anywhere in either function body — confirmed by reading the full function, not just the
  branch — so passing `None` for those parameters when passing a prebuilt module is completely safe, not a
  latent validation risk.
- Confirmed `from mmdet.apis import init_detector` and `from mmpose.apis import init_model` import cleanly in
  this repo's `.venv` (both are already transitive deps used inside `mmaction/apis/inference.py` itself, just
  not previously imported directly by `scripts/model_sweep.py`). Aliased mmpose's `init_model` to
  `init_pose_model` on import since `scripts/model_sweep.py` already has an unrelated local `model` variable
  (the classifier) in the same function and importing a second bare `init_model`-shaped name next to
  `init_recognizer` reads better distinguished.
- Both `init_detector`'s and `init_model`'s real signatures take `device` as a keyword (`device: str =
  'cuda:0'`), confirmed via `inspect.signature` in `.venv` — matches how `run_skeleton_topdown` already calls
  `init_recognizer(..., device=device)`, so no calling-convention surprises building the other two models the
  same way.
- Fix: in `run_skeleton_topdown`, `init_detector(detector_config, detector_checkpoint, device=device)` and
  `init_pose_model(pose_config, pose_checkpoint, device=device)` are now called once per model entry
  (immediately after the existing single `init_recognizer` call, before the video loop), and the per-video
  `detection_inference`/`pose_inference` calls now pass the built `detector_model`/`pose_model` objects with
  `None` in the checkpoint-path slot, instead of passing `detector_config`/`pose_config` paths (which is what
  was causing every video to trigger a fresh `init_detector`/`init_model` call, and a fresh checkpoint read
  from disk, inside those two functions).
- Verification (single-model `posec3d`-only temp YAML, `videos/` folder, both checkpoints already warm from
  earlier tasks so no download noise): run exited 0, produced exactly 10 rows (5 ranked rows x 2 videos), all
  `dataset=FineGYM` with real GYM99 label strings. The log showed exactly **one** "Loads checkpoint by local
  backend from path" line each for the classifier, detector, and pose checkpoints (3 total for a 2-video run)
  — before the fix this would have been 3 (classifier once, since that was already built once) + 2x2 = 7, i.e.
  the detector/pose lines would each have appeared once per video (2x each) instead of once total. This is
  direct, concrete evidence the fix works, not an inferred one.
- Numerical regression check: `backflip.mp4` top-1 came back as `(UB) (swing forward) double salto backward
  stretched` at score `0.1173364520072937`, matching Task 5's recorded learnings entry ("an uneven-bars/double
  salto label at only 0.117 confidence") to the same 3 decimal places recorded there — confirms building the
  detector/pose models once instead of per-video does not change inference output, as expected, since both
  functions still build/use the exact same architecture + checkpoint weights either way, only the call site
  moved. (Task 5/6/7's learnings entries didn't record `demo.mp4`'s exact posec3d score for comparison, only
  `backflip.mp4`'s; `backflip.mp4`'s match was the only direct number available to check against.)
- Lint (`flake8`, `isort --check-only`, `yapf --diff`) all clean after one `isort` fixup (the two new
  `mmdet.apis`/`mmpose.apis` imports needed to sit in the third-party import block above the blank line
  separating it from the `mmaction`-namespace block, not below it) and one `yapf` reflow of the `pose_inference`
  call onto a single line once it dropped from 5 args across multiple lines to 5 args that fit on one line.

## Fix S2 — A model-level failure aborts the whole sweep instead of skipping that model

- Fix is exactly the 3-ish lines the finding described: in `main()`'s loop over `config['models']`, the
  existing `runner(model_entry, videos, top_k, completed, writer, csv_file)` call is now wrapped in
  `try/except Exception: logger.exception(...); continue`, logging which `model_entry['name']` failed. Nothing
  in `run_recognizer`, `run_skeleton_topdown`, `resolve_checkpoint`, or `RUNNERS` was touched — this is purely
  an outer safety net around the per-model call in `main()`.
- `yapf --diff` initially wanted the wrapped `runner(...)` call collapsed back onto a single line once it moved
  one indent level deeper inside the new `try:` block — the extra indent didn't push it over the line-length
  limit after all, so yapf's authoritative reflow was simpler than my first hand-wrapped attempt (matches Task
  2's learnings note: always defer to `yapf --diff`'s actual output rather than guessing the wrapping).
- Verified against a real failure, not a synthetic exception: a two-model temp YAML (`broken_model` pointing at
  a nonexistent config path `configs/recognition/slowfast/does_not_exist_config.py`, then a real `slowfast`
  entry identical to the default config's, checkpoint already warm from earlier tasks) run against `videos/`.
  Result: process exited 0 (confirmed via a separate run capturing `$?` directly, not just tailing output),
  `broken_model`'s failure produced exactly one `[ERROR] Failed to run model 'broken_model', skipping to next
  model` line via `logger.exception` with a full traceback rooted in `mmengine.Config.fromfile` raising
  `FileNotFoundError` on the bad path (confirming the crash happens exactly where S2 said — outside any
  per-video try, at `init_recognizer`'s config-loading step, since a checkpoint URL was given explicitly so
  `resolve_checkpoint` itself didn't fail), and the sweep continued straight on to load and run `slowfast`,
  writing all 10 correct rows (5 per video, matching Task 4's recorded `backflip.mp4`/`demo.mp4` predictions
  exactly) to the output CSV.
- Confirmed the fix is additive, not a replacement of the existing per-video error handling: re-ran Task 7's
  corrupt-video regression test (single-model `slowfast`, a 20-byte truncated `backflip.mp4` renamed
  `corrupt.mp4` plus an untouched `demo.mp4`, fresh temp folder) and got byte-for-byte the same behavior as
  Task 7 recorded — exit 0, exactly one `[ERROR] Failed to run model 'slowfast' on video 'corrupt.mp4',
  skipping` line (the runner's own inner per-video `except Exception` block, not the new outer one in `main()`,
  since `slowfast` itself never raised past `run_recognizer`), and exactly 5 correct rows for the surviving
  `demo.mp4` video. This confirms the new outer `try/except` in `main()` only catches what escapes a runner
  entirely (checkpoint/config/model-load failures) and never intercepts or changes the runners' own
  already-correct per-video skip-and-continue behavior.

## Fix S3 — Pre-existing empty output file silently produced a headerless CSV

- Root cause matched the finding exactly: `open_output_csv`'s `is_new = not output_path.exists()` is `False`
  for a zero-byte file (`Path.exists()` only checks presence, not size), so a `touch`ed/`> `-created empty
  output file skipped `writer.writeheader()` entirely. Fix: `is_new = not output_path.exists() or
  output_path.stat().st_size == 0`.
- Reproduced the bug first on the pre-fix code: `touch`ed an empty CSV, ran `scripts/model_sweep.py` against it
  with a single-model (`slowfast`) temp YAML config and a one-video temp folder (checkpoint already cached from
  earlier tasks, so no download needed). Resulting file was exactly one headerless data row followed by four
  more — `demo.mp4,slowfast,Kinetics-400,1,arm wrestling,1.0` as line 1, no header — confirming the finding's
  described symptom byte-for-byte before touching any code.
- After the one-line fix, re-ran the identical empty-file scenario: the file's first line is now exactly
  `video_path,model_name,dataset,rank,label,score`, followed by the 5 data rows.
- Re-checked the two cases the fix must not break: a genuinely-missing `--output` path still gets a fresh file
  with a header (unchanged, since `not output_path.exists()` alone already covered that case); and re-running
  the same command against the now-populated, non-empty CSV from the previous step appends nothing new (all
  pairs already in `load_completed_pairs`) and does not duplicate the header (`grep -c` for the header line
  stayed at 1) — the resume path from Task 2/`load_completed_pairs` is unaffected by this change since it only
  touches `open_output_csv`'s header-writing decision, not the reading side.
- Quality gate: `flake8`, `isort --check-only`, `yapf --diff` on `scripts/model_sweep.py` all clean (exit 0,
  no diff).

## Fix S4 — Per-row `writerow` loop left a pair vulnerable to a mid-write interrupt

- Fix matched the finding exactly, applied identically in both `run_recognizer` and `run_skeleton_topdown`: the
  `for rank, (idx, score) in enumerate(...): writer.writerow({...})` loop became a list comprehension building
  all `top_k` row dicts (`rows = [{...} for rank, (idx, score) in enumerate(...)]`), followed by a single
  `writer.writerows(rows)` then the existing `csv_file.flush()`. No other lines in either function changed —
  the `except Exception`/`continue` per-video error handling above the row-building code, and the `csv_file.flush()`
  call after it, are both untouched, so a raised exception during inference still skips row-writing entirely
  (nothing to make interruptible was ever reachable in that path to begin with) and the flush still happens
  once per completed video exactly as before.
  Note: this narrows but doesn't eliminate the interrupt window S4 is
  about — a `KeyboardInterrupt` landing inside the `writerows(rows)` call itself (a single C-level loop over an
  already-fully-built Python list) is far less likely than landing between two separate Python-level
  `writerow` calls with unrelated work source-side, but `csv.DictWriter.writerows` is not documented as atomic
  against arbitrary interrupts. This matches the finding's own accepted fix exactly, so no gap beyond what
  the review already scoped.
- Verified the refactor is output-preserving, not just lint-clean, by re-running both runners against
  already-cached checkpoints and diffing against numbers recorded in earlier learnings entries rather than
  just "no traceback": single-model `slowfast` temp YAML against `videos/` reproduced Task 4's exact recorded
  predictions (`backflip.mp4` top-1 "gymnastics tumbling" @ 0.6132610440254211, `demo.mp4` top-1
  "arm wrestling" @ 1.0); single-model `posec3d` temp YAML against `videos/` reproduced Fix S1's exact recorded
  `backflip.mp4` top-1 ("(UB) (swing forward) double salto backward stretched" @ 0.1173364520072937) to full
  float precision, plus 5 ranked rows for `demo.mp4` with plausible GYM99 labels. Both runs exited 0 with zero
  `[ERROR]` lines, confirming `writerows` on a list of dicts produces byte-identical CSV content to the
  equivalent sequence of `writerow` calls (as expected, since `csv.DictWriter.writerows` is documented as
  simply calling `writerow` for each item in its argument, just without yielding back to Python between rows).
- Quality gate: `flake8`, `isort --check-only`, `yapf --diff` on `scripts/model_sweep.py` all clean (exit 0,
  no diff) — no reflow needed beyond the list-comprehension shape written directly.

## Fix N1 — `detector_checkpoint`/`pose_checkpoint` bracket access raised a bare `KeyError`

- Fix matched the finding exactly: in `run_skeleton_topdown`, `model_entry['detector_checkpoint']` and
  `model_entry['pose_checkpoint']` became `model_entry.get('detector_checkpoint')` and
  `model_entry.get('pose_checkpoint')`, matching the existing `model_entry.get('classifier_checkpoint')` a few
  lines below. No other lines in the function changed.
- Quality gate: `.venv/bin/python -m flake8 scripts/model_sweep.py`, `.venv/bin/python -m isort
  --check-only scripts/model_sweep.py`, `.venv/bin/python -m yapf --diff scripts/model_sweep.py` all clean
  (exit 0, no diff/output).
- Verified the fix actually improves the error, not just silences a lint concern: built a temp single-model
  `skeleton_topdown` YAML (copy of the real `posec3d` entry from `scripts/model_sweep_config.yaml` with the
  `detector_checkpoint` line deleted) and ran `scripts/model_sweep.py --videos-dir videos --config
  <temp>.yaml --output <temp>.csv`. Before the fix this would have died with `KeyError:
  'detector_checkpoint'`; after the fix it raises (and, thanks to the already-fixed S2 per-model isolation,
  logs and continues past)
  `FileNotFoundError: No metafile.yml found next to config
  .../configs/person_detector/faster-rcnn_r50_fpn_2x_coco_infer.py - a checkpoint must be given explicitly for
  configs without mmaction2 model metadata.` — exactly the message `lookup_checkpoint_in_metafile` produces,
  and exactly why: `configs/person_detector/` is a vendored mmdetection config with deliberately no
  `metafile.yml` (per CLAUDE.md), so auto-resolution correctly refuses rather than crashing confusingly.
- Verified the normal case is unaffected: ran the real `posec3d` entry (both checkpoints given explicitly, as
  in the shipped `scripts/model_sweep_config.yaml`) against one video (`videos/demo.mp4`) end to end. Detector,
  pose model, and classifier all initialized and ran without error, producing 5 well-formed top-k rows in the
  output CSV (top-1 "(UB) clear hip circle backward to handstand" @ 0.3563977777957916), confirming
  `.get(...)` returning the same present value behaves identically to bracket access when the key exists.

## Fix N2 — `det_score_thr` and `det_cat_id` are hardcoded

- Fix matched the finding exactly: in `run_skeleton_topdown`'s `detection_inference(...)` call, the literals
  `det_score_thr=0.9, det_cat_id=0` became `det_score_thr=model_entry.get('det_score_thr', 0.9)` and
  `det_cat_id=model_entry.get('det_cat_id', 0)`. No other lines changed; `scripts/model_sweep_config.yaml` was
  left untouched (it doesn't set either field, so it exercises the default path).
- Quality gate: `.venv/bin/python -m flake8 scripts/model_sweep.py`, `-m isort --check-only`, `-m yapf --diff`
  all clean (exit 0, no diff/output) — no reflow needed since both lines stayed the same shape/length class.
- Verified default behavior is byte-identical to before the fix: ran the real `posec3d` entry from
  `scripts/model_sweep_config.yaml` (copied into a temp single-model YAML, no `det_score_thr`/`det_cat_id` set)
  against `videos/` (both checkpoints already warm). Output matched two independently-recorded prior runs to
  full float precision: `backflip.mp4` top-1 `(UB) (swing forward) double salto backward stretched` @
  `0.1173364520072937` (matches Fix S1's and Fix N1's recorded value) and `demo.mp4` top-1 `(UB) clear hip
  circle backward to handstand` @ `0.3563977777957916` (matches Fix N1's recorded value) — confirms
  `model_entry.get(key, <same literal default>)` is behaviorally identical to the old hardcoded literal when
  the key is absent, as expected.
- Verified the override actually takes effect: built a temp YAML copying the same `posec3d` entry but adding
  `det_score_thr: 0.999` (deliberately near-impossible for the Faster R-CNN detector to clear). Command:
  `.venv/bin/python scripts/model_sweep.py --videos-dir videos --config <temp>.yaml --output <temp>.csv`. Exit
  0, zero `[ERROR]`/exception lines (the pipeline degrades gracefully rather than crashing when detections are
  filtered out — not this fix's concern either way). Both videos' predictions changed completely relative to
  the default-threshold run: `backflip.mp4` top-1 flipped from `(UB) (swing forward) double salto backward
  stretched` @ `0.117` to `(UB) transition flight from low bar to high bar` @ `0.1047`, and `demo.mp4` top-1
  flipped from `(UB) clear hip circle backward to handstand` @ `0.356` to `(UB) giant circle backward` @
  `0.1054` — every one of the 10 output rows differs from the default-threshold run's corresponding row in both
  label and score. This is direct, concrete evidence `det_score_thr` is actually read from the YAML entry and
  passed through to `detection_inference`, not silently ignored.
- `det_cat_id` was not separately override-tested (the finding's fix is the same one-line pattern for both
  fields, and testing `det_score_thr`'s override already proves `model_entry.get(...)` reaches this call site
  correctly for both keyword arguments).
