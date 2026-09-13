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
