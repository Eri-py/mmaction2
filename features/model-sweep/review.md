# Review — Multi-Model Video Evaluation Sweep

## Verdict

The implementation satisfies the spec. All nine acceptance criteria are met, and each was verified by real
execution (real checkpoints, real inference, the repo's own `videos/` folder) rather than by inspection — the
50-row full-sweep run in Task 6 and the byte-identical resume plus corrupt-file run in Task 7 are exactly the
right evidence for this feature. The diff is two new files and nothing else: no library, config, or notebook
code was touched, so there is no silent-behavior-drift surface in `mmaction/` or `configs/`. Every `str`-only
API hazard called out in `CLAUDE.md` (`init_recognizer`'s `checkpoint`, `inference_recognizer`'s `video`,
`frame_extract`'s `video_path`) is handled correctly, and `flake8`/`isort`/`yapf` are all clean. No blockers.

The findings below are all about behavior at the spec's stated scale (1000+ videos) rather than at the 2-video
scale that was actually exercised. Two are worth acting on before a real sweep: the PoseC3D runner rebuilds the
detector and pose estimator from disk once per video (S1), and any model-level failure — a 404 checkpoint, an
OOM at load — aborts the entire sweep rather than skipping that one model (S2).

## Acceptance Criteria

1. **Rows for every (video, model, rank) that succeeded, rank 1..top_k — MET.** `run_recognizer`
   (`scripts/model_sweep.py:173-183`) and `run_skeleton_topdown` (`scripts/model_sweep.py:236-246`) both write
   `enumerate(..., start=1)` rows from a `topk(top_k)` call. Verified end to end: 50/50 rows for 5 models × 2
   videos, `rank` 1..5 per pair (`progress.md:41-45`).
2. **Config-swappable without code changes — MET.** `main` (`scripts/model_sweep.py:266-272`) iterates
   `config['models']` and dispatches purely on `model_entry['type']` through the `RUNNERS` dict
   (`scripts/model_sweep.py:249-252`); nothing else is model-specific. Tasks 4, 5 and 7 each ran single-model
   temp YAMLs against the unchanged script.
3. **`top_k` (default 5) applied globally — MET.** `top_k = config.get('top_k', 5)`
   (`scripts/model_sweep.py:258`), passed into both runners; `scripts/model_sweep_config.yaml:5` sets 5.
4. **Per-pair failure logged and skipped, others unaffected — MET.** Blanket `except Exception` +
   `logger.exception` + `continue` scoped to one video in both runners (`scripts/model_sweep.py:169-172`,
   `232-235`). Verified with a 20-byte truncated video: exit 0, exactly one logged error, the working video's 5
   rows correct (`learnings.md:162-171`). See S2 for the model-level (not pair-level) case.
5. **Resumable, no duplicates, no reprocessing — MET.** `load_completed_pairs`
   (`scripts/model_sweep.py:67-77`) builds the `(basename, model_name)` set; both runners check it before
   inference (`scripts/model_sweep.py:163-165`, `207-209`); the file is opened in append mode
   (`scripts/model_sweep.py:84`). Verified: re-run produced an MD5-identical CSV (`learnings.md:158-161`).
6. **Filterable by `model_name`, with the label space visible — MET.** `dataset` is a first-class CSV column
   (`scripts/model_sweep.py:27`, written at `:178` and `:241`), carrying `Kinetics-400` / `FineGYM`; pandas
   filtering verified in Task 6.
7. **Non-recursive scan — MET.** `discover_videos` uses `iterdir()` with an `is_file()` guard, not `rglob`
   (`scripts/model_sweep.py:60-64`); verified against a temp dir with a nested `.mp4` and a `.txt` in Task 2.
8. **Checkpoint auto-resolution from the config's own metafile — MET.** `resolve_checkpoint`
   (`scripts/model_sweep.py:143-149`) calls `lookup_checkpoint_in_metafile` only when the checkpoint is `None`;
   `run_recognizer` reads it with `.get('checkpoint')` (`scripts/model_sweep.py:157`) so omission is legal.
   Re-verified during this review: the SlowFast K400 lookup returns the exact expected URL, a `FineGYM` dataset
   for that config raises a `ValueError` naming the available datasets, and a metafile-less vendored config
   raises the actionable `FileNotFoundError`. See the Tests section for the one coverage gap here.
9. **Local `checkpoints/` cache, download once, reuse after — MET.** `cache_checkpoint_locally`
   (`scripts/model_sweep.py:128-140`) downloads only when the target does not exist. Verified: all 7
   checkpoints present after Task 6, mtimes bit-identical across the Task 7 re-run. `*.pth` is already
   gitignored (`.gitignore:129`), so the ~1.2 GB cache cannot be accidentally committed.

## Scope

The diff matches the plan's Files Changed table exactly: `scripts/model_sweep.py` and
`scripts/model_sweep_config.yaml` added, nothing else outside the feature directory. No scope drift.

Two small deviations from the plan's Approach section, neither harmful:

- The plan specified `del model; torch.cuda.empty_cache()` between models. Not implemented — but `model` is a
  local in each runner, so it is released when the runner returns; the practical effect is the same. Not
  raised as a finding.
- `run_skeleton_topdown` adds two `torch.cuda.empty_cache()` calls inside the per-video pipeline
  (`scripts/model_sweep.py:222`, `229`) that the plan did not describe. Harmless and matches the demo
  notebook's shape.

The plan's deliberate deferrals (no `scripts/README.md`, PyYAML not declared in `pyproject.toml`) were honored.
See N5 on the README deferral's one consequence.

## Blockers

None.

## Suggestions

#### S1 — PoseC3D runner reloads the detector and pose estimator from disk for every video

- **File:** `scripts/model_sweep.py:215-228`
- **Issue:** `detection_inference` and `pose_inference` are called with config paths, so each call runs
  `init_detector` / `init_model` internally and re-reads the 167 MB detector and 115 MB pose checkpoints. At
  the spec's stated 1000+ videos that is ~2000 redundant model builds. The plan justified this with "they
  don't expose a separate init step" — that is incorrect: both functions accept a prebuilt `nn.Module` as
  their first argument (`mmaction/apis/inference.py:171-172` and `208-212`; `:237` and `:267`), the same way
  the classifier is already built once at `scripts/model_sweep.py:202`.
- **Fix:** Build the detector and pose model once per model entry, before the video loop (`init_detector` from
  `mmdet.apis`, `init_model` from `mmpose.apis`, both with the already-resolved local checkpoint paths), and
  pass those module objects in place of the config paths. The checkpoint arg is ignored on that branch, so
  pass the resolved path or `None`.
- **Decision:** Accepted — addressed in "Address S1: init PoseC3D detector/pose once per model, not per video"

#### S2 — A model-level failure aborts the whole sweep instead of skipping that model

- **File:** `scripts/model_sweep.py:266-272`
- **Issue:** Checkpoint resolution and `init_recognizer` run outside any `try`
  (`scripts/model_sweep.py:156-159`, `192-203`). A 404 or truncated download, a bad config path, a missing
  `label_map`, or an OOM at load kills the process, so every model after the failing one is skipped too. The
  spec's failure requirement is that "no other (video, model) pairs are affected" — a five-model, 1000-video
  run currently loses the remaining models to one bad entry.
- **Fix:** Wrap the `runner(...)` call in `main`'s loop in `try/except Exception: logger.exception(...);
  continue`. Three lines, and the resumability logic already makes the retry story correct.
- **Decision:** Accepted — addressed in "Address S2: isolate a model-level failure to that model, not the whole sweep"

#### S3 — A pre-existing empty output file silently produces a headerless CSV

- **File:** `scripts/model_sweep.py:83`
- **Issue:** `is_new = not output_path.exists()` is false for a zero-byte file, so `--output out.csv` after a
  `touch out.csv` or `> out.csv` writes data rows with no header. Confirmed by running it during this review:
  the file ends up as `a.mp4,slowfast,Kinetics-400,1,x,0.5` with no header line, which breaks the
  pandas-filtering criterion, and on resume `csv.DictReader` consumes the first data row as the field names
  (`load_completed_pairs` then returns an empty set, or raises `KeyError: 'video_path'` once there is more
  than one row).
- **Fix:** `is_new = not output_path.exists() or output_path.stat().st_size == 0`.
- **Decision:** Accepted — addressed in "Address S3: treat a zero-byte output file as new (write header)"

#### S4 — An interrupt mid-write can leave a pair with fewer than `top_k` rows, which resume then treats as complete

- **File:** `scripts/model_sweep.py:173-183`
- **Issue:** Rows are written one `writerow` at a time and flushed only after the loop. A `KeyboardInterrupt`
  between two `writerow` calls still runs `main`'s `finally: csv_file.close()`
  (`scripts/model_sweep.py:273-274`), flushing a partial block. `load_completed_pairs` keys on the pair, not
  the row count, so the next run skips that pair and the CSV permanently violates the "exactly `top_k` ranked
  rows per successful pair" criterion. Narrow window, but a 1000-video sweep is expected to be interrupted.
- **Fix:** Build the row dicts into a list, then `writer.writerows(rows)` followed by `csv_file.flush()`, in
  both runners. The whole block then lands in the buffer in one call.
- **Decision:** Accepted — addressed in "Address S4: build all of a video's rows before writing, not one at a time"

## Nitpicks

#### N1 — `detector_checkpoint` / `pose_checkpoint` omission raises a bare `KeyError`

- **File:** `scripts/model_sweep.py:193, 196`
- **Issue:** These use bracket access, so a YAML entry missing them dies with an unhelpful
  `KeyError: 'detector_checkpoint'` — the one spot in the checkpoint path that does not produce an actionable
  message.
- **Fix:** Use `.get(...)`. The metafile branch then fires and already raises exactly the right message for
  these vendored configs: "No metafile.yml found next to config ... a checkpoint must be given explicitly for
  configs without mmaction2 model metadata." (`scripts/model_sweep.py:98-101`, verified during this review).
- **Decision:** — _(pending)_

#### N2 — `det_score_thr` and `det_cat_id` are hardcoded

- **File:** `scripts/model_sweep.py:219-220`
- **Issue:** The plan's own principle is that "everything that varies between models is data, not logic", but
  these two detection knobs — the ones most likely to need tuning on non-gymnastics footage — are the only
  pipeline parameters not exposed in the YAML.
- **Fix:** Read them as `model_entry.get('det_score_thr', 0.9)` / `model_entry.get('det_cat_id', 0)`.
- **Decision:** — _(pending)_

#### N3 — `--videos-dir` pointing at a missing directory produces a raw traceback

- **File:** `scripts/model_sweep.py:60-64`
- **Issue:** `iterdir()` raises `FileNotFoundError` / `NotADirectoryError` straight out of the CLI before
  anything is logged.
- **Fix:** One `if not videos_dir.is_dir(): raise SystemExit(f'...')` guard in `main` or `discover_videos`.
- **Decision:** — _(pending)_

#### N4 — The `video_path` column contains a basename, not a path

- **File:** `scripts/model_sweep.py:27`
- **Issue:** Deliberate (it is what makes resume location-independent) and documented in the plan, but the
  column name misleads anyone reading the CSV without the plan in hand.
- **Fix:** Either rename to `video` / `video_name`, or note it in the YAML header comment. Renaming would
  invalidate any CSV already produced, so the comment is the cheaper option.
- **Decision:** — _(pending)_

#### N5 — The YAML schema is undocumented anywhere a user would look

- **File:** `scripts/model_sweep_config.yaml:1-4`
- **Issue:** The plan deferred a `scripts/README.md` on the grounds that "the script's own `--help` output
  covers usage" — but `--help` documents only the three CLI flags, not the config schema. The two `type`
  values, the required-per-type field sets, and `label_map` (required by the code, never mentioned in the
  spec) are discoverable only by reading `model_sweep.py`.
- **Fix:** Extend the existing header comment in `scripts/model_sweep_config.yaml` with a short field list per
  `type`, noting which fields are optional (`checkpoint`, `classifier_checkpoint`).
- **Decision:** — _(pending)_

## Tests

No pytest suite, which is correct for this repo: `.claude/coding-guidelines.md` sets the bar for one-off
scripts at "actually run it end-to-end and confirm a sane result", and that bar was cleared thoroughly. The
verification is unusually good for a prototype — Task 4 checked that `demo.mp4` predicts "arm wrestling" at
1.0 and `backflip.mp4` "gymnastics tumbling" at 0.61, i.e. it confirmed the pipeline is right, not just that
rows appeared; Task 6 checked all 50 labels against their label maps programmatically rather than
spot-checking; Task 7 used an MD5 comparison for resume idempotency and `stat -c %Y` for no-redownload. Lint
(`flake8`/`isort`/`yapf`) re-confirmed clean during this review, and both label maps parse to exactly 400 and
99 non-empty entries, matching their models' class counts.

Coverage gaps, in rough order of value:

- **Acceptance criterion 8 was never exercised through the CLI.** The optional-checkpoint path was tested at
  the unit level in Task 3 (`lookup_checkpoint_in_metafile` and `resolve_checkpoint` called directly), but no
  sweep was ever run with a YAML entry that actually omits `checkpoint` — the default config gives all five
  explicitly by design (`scripts/model_sweep_config.yaml:2-4`). The wiring at
  `scripts/model_sweep.py:157` and `:200` is correct by inspection and the checkpoint is already cached, so
  a one-model temp YAML with the `checkpoint:` line deleted would close this in seconds.
- **The skeleton runner's failure path is untested.** The corrupt-file test used a `recognizer` entry only.
  `run_skeleton_topdown` has more ways to fail per video (frame extraction, zero person detections feeding
  `inference_skeleton`) and its `try` block spans a `with tempfile.TemporaryDirectory()`; worth one run
  against a video with no people in it.
- **Everything was verified at 2 videos.** S1, S2 and S4 are all failure modes that only bite at the 1000+
  scale the spec is written for, and nothing in the verification would have surfaced them.
- Not a gap, but noted in `learnings.md:130-137`: the per-frame `ResourceWarning: unclosed file` from mmpose's
  own `Image.open(img).size` is upstream noise, not this feature's code. Correctly left alone.

## Recommended Decisions

- **S1** — Accept — The plan's justification for the per-video reload is factually wrong, the API already
  supports a prebuilt module, and this is the single biggest cost in a real 1000-video PoseC3D sweep.
- **S2** — Accept — Three lines, and it is the difference between losing one model and losing the run; the
  spec explicitly asks that a failure not affect other pairs.
- **S3** — Accept — One-line fix for a silent data-corruption footgun that a user can hit with an ordinary
  shell habit.
- **S4** — Accept — Cheap (`writerows` instead of a `writerow` loop) and it protects the exact invariant the
  spec's first acceptance criterion states, on a run the spec expects to be interrupted.
- **N1** — Accept — Pure win: replacing `[...]` with `.get(...)` routes to an error message that is already
  written and already says the right thing.
- **N2** — Accept — Two `.get()` calls, and it removes the last piece of per-model tuning that requires
  editing script code.
- **N3** — Accept — One guard line; a mistyped `--videos-dir` is the most likely user error with this CLI.
- **N4** — Accept — As a comment only, not a rename. Renaming the column would break existing output CSVs for
  no real gain.
- **N5** — Accept — The plan's stated reason for skipping docs (`--help` covers it) does not hold for the YAML
  schema, and a few comment lines in the file the user is going to edit anyway is the whole fix.
