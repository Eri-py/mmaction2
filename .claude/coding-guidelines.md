# Coding Guidelines — mmaction2

## Scope & bar

This is a personal research/experimentation environment built on a plain clone of upstream `open-mmlab/mmaction2`
— not a product with a release process, and not a fork tracking upstream. Optimize for correctness of actual
pipeline output (predictions, rendered videos, sweep results) and for notebooks that run cleanly end-to-end — not
for production hardening, backward compatibility with upstream's own release cadence, or exhaustive robustness.

## Repo layout

See `CLAUDE.md` at the repo root for the full layout and the environment's known fragile spots (dependency
versions, NumPy 2.0 patches, etc.) — read that first. In short:

- `mmaction/` — the library, editable-installed (`pip install -e . --no-deps`).
- `configs/` — model configs by task (`recognition/`, `skeleton/`, `detection/`, etc.), plus
  `configs/person_detector/` and `configs/skeleton_coord/` (vendored, non-mmaction2 configs — see their own
  READMEs).
- `notebooks/` — the actual day-to-day work.
- `videos/` — sample input videos.
- `tools/`, `tests/` — unchanged from upstream.

## Python / library code (`mmaction/`)

- Match the existing style — this repo's own `.pre-commit-config.yaml`/`setup.cfg` already define it:
  `flake8`, `isort` (`known_first_party = mmaction`, 79-char lines), `yapf` (PEP8-based). Don't introduce a
  different style in new code.
- When patching an upstream bug (e.g. a NumPy 2.0 incompatibility), keep the fix minimal and localized to the
  actual broken line(s) — don't refactor surrounding code you were not asked to touch, and don't fix the same
  class of bug in files that aren't actually failing yet.
- If a fix must live outside this repo (in an installed dependency's `site-packages`, because no newer release
  fixes it and it's not our code to patch permanently), say so explicitly in the commit/PR and in `CLAUDE.md` —
  a site-packages patch is invisible to git and silently disappears on the next reinstall.

## Notebooks (`notebooks/`)

- Anchor every path through a `ROOT = Path.cwd()` (or `.parent`, depending on nesting) variable built with
  `pathlib` — never a hardcoded relative string. See `CLAUDE.md` for the full convention, including which
  OpenMMLab APIs are `str`-only vs. `Union[str, Path]`.
- Never leave a notebook writing a persistent demo-output file to disk. Render to a
  `tempfile.TemporaryDirectory()` and `display(IPython.display.Video(path, embed=True))` instead, so the result
  lives in the notebook's own output and the temp file is cleaned up automatically.
- Don't add a "training" section to a notebook meant for inference/exploration, and don't add inference demo
  code to a notebook meant for training — one notebook, one clear purpose, matching its filename.

## Configs (`configs/`)

- A new mmaction2 model config goes under its existing task folder (`recognition/`, `skeleton/`, etc.), following
  that folder's own naming convention — never invent a new top-level task folder without checking whether an
  existing one already fits.
- A config that is copied in from an external OpenMMLab project (`mmdetection`, `mmpose`, ...) rather than
  trained by mmaction2 itself must live in a folder whose own `README.md` says so explicitly, with real upstream
  benchmark numbers (pulled from that project's own docs/metafile, never fabricated) — never give such a folder
  a `metafile.yml` matching mmaction2's own model-zoo format, since that format asserts mmaction2 trained and
  benchmarked it.

## Comments

- One line, under ~100 chars. Two lines only when truly needed.
- Explain *why*, not what the code already says. Skip if self-evident.
- No multi-line block comments restating the obvious.

## Testing bar

- Library code (`mmaction/`): run the relevant `pytest` suite under `tests/` for anything touched (mirror the
  source path, e.g. `mmaction/datasets/transforms/pose_transforms.py` → `tests/datasets/transforms/`), plus
  `flake8`/`isort`/`yapf` on changed files.
- Notebooks and one-off scripts: no formal test suite. The bar is "actually run it end-to-end and confirm a sane
  result" — verified by execution, not by writing pytest tests around exploratory code.
- Config files: confirm `mmengine.Config.fromfile(path)` still parses after adding or moving one.
