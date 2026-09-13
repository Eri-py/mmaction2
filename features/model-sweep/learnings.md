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
