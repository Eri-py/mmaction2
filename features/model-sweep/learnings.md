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
