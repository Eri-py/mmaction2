# mmaction2 — project notes

## What this is

A plain clone of upstream `open-mmlab/mmaction2` (`main` branch), used as a personal research/experimentation
environment — not a fork being kept in sync with or contributed back upstream. Work here is exploratory
(notebooks, ad-hoc scripts, model sweeps), not a product with a release process.

## Environment — read before touching dependencies

Python env is managed entirely by `uv` — `.venv/` at the repo root. **Never install a Python package via `apt`
or any system package manager, and never use system/global Python for anything in this repo.** Always
`uv pip install -p .venv ...`.

`mmcv`, `mmdet`, and `mmpose` are compiled/installed for this machine's RTX 5070 (Blackwell, `sm_120`) — getting
there took real work, and a few fragile spots are worth knowing about:

- **`setuptools` drifts back to a version that breaks `pkg_resources`.** Ordinary `uv pip install` runs for
  unrelated packages have repeatedly bumped `setuptools` to a version (~84+) that dropped `pkg_resources`
  entirely, which `mim`/`mmpose` still import. Symptom: `ModuleNotFoundError: No module named 'pkg_resources'`.
  Fix: `uv pip install -p .venv "setuptools==75.8.0"`. Worth checking (`python -c "import pkg_resources"`) after
  any install batch, not just reacting to the failure.
- **`moviepy` must be `1.0.3`, not `2.x`.** `moviepy` 2.0 removed the `moviepy.editor` module that
  `mmaction/visualization/video_backend.py` still imports from. `2.x` installs cleanly but breaks at first use
  (`ModuleNotFoundError: No module named 'moviepy.editor'`).
- **NumPy 2.0 removed `np.Inf`** (lowercase `np.inf` is required now). Already patched in this repo's own
  `mmaction/utils/misc.py` and `mmaction/datasets/transforms/pose_transforms.py`, and in the installed
  `mmpose` package's `visualization/opencv_backend_visualizer.py` (`int(radius)` on a 1-element array is also a
  NumPy-2.0 hard error there, not just a deprecation). The `mmpose` patch lives in `site-packages`, not this repo,
  so **it will silently disappear if `mmpose` is ever reinstalled/upgraded** — if pose visualization breaks with
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`, that's why; reapply
  `int(np.asarray(radius).item())` in place of the bare `int(radius)` there.
- **`mmaction2` itself must be `pip install -e . --no-deps`** for `import mmaction` to work from anywhere
  (script, different cwd) rather than only from `python -c` one-liners run from the repo root (which get a free
  pass via cwd-based `sys.path`, masking that the package was never actually installed).

None of the above is speculative — each was hit and fixed in a real session; treat a recurrence as "oh, that
again," not a new mystery.

## Repo layout

- `mmaction/` — the library itself, editable-installed. Standard mmaction2 source tree.
- `configs/` — model configs, organized by **task**, not by architecture family:
  - `recognition/`, `recognition_audio/`, `skeleton/`, `detection/`, `localization/`, `retrieval/`,
    `multimodal/`, `_base_/` — upstream mmaction2's own categories, unchanged.
  - `configs/person_detector/`, `configs/skeleton_coord/` — **not mmaction2 models.** Vendored single configs
    from `mmdetection`/`mmpose` (no full clone of either repo exists here), kept only because the skeleton-based
    recognition notebook needs a person detector + a top-down pose estimator upstream of `configs/skeleton/`'s
    real classifier. Each folder's `README.md` says so explicitly and cites real upstream benchmark numbers —
    never add a `metafile.yml` to either, since that format asserts "mmaction2 trained and benchmarked this,"
    which is false here.
- `notebooks/` — the actual day-to-day work: `exploratory.ipynb` (general playground, adapted from OpenMMLab's
  Colab tutorial), `demo_recognition.ipynb` (plain RGB action recognition, TSN), `demo_skeleton.ipynb`
  (person detection → pose estimation → PoseC3D skeleton classification, with skeleton+label drawn on the video
  and displayed inline).
- `videos/` — sample input videos (`demo.mp4`, `backflip.mp4`).
- `tools/`, `tests/`, `docs*/` — unchanged from upstream.
- No `demo/` directory — deleted; its scripts were superseded by `notebooks/`, and its still-needed
  `demo_configs/` subfolder was preserved (split into `configs/person_detector/`/`configs/skeleton_coord/` above,
  not left where it was).

## Notebook conventions

- Every notebook anchors its paths with `ROOT = Path.cwd()` (or `Path.cwd().parent` if the notebook lives one
  level below the repo root, e.g. under `notebooks/`) and builds every other path via `pathlib`
  (`ROOT / "videos/demo.mp4"`), never a hardcoded relative string.
- Before passing a `pathlib.Path` into an mmaction2/mmengine/mmcv/mmdet/mmpose API, check the actual type
  annotation rather than assuming `Path` works everywhere — several accept `Union[str, Path]`, but some
  (`inference_recognizer`'s `video`, `init_recognizer`'s `checkpoint`, `add_datasample`'s `out_path`) are typed
  `str`-only and will break on a raw `Path` (e.g. `.startswith()` calls inside checkpoint loading).
- Don't write persistent output files (rendered videos, images) to disk for demo purposes. Render to a
  `tempfile.TemporaryDirectory()`, then `display(IPython.display.Video(path, embed=True))` while the temp file
  still exists (embedding reads the file at `display()` time, not lazily) — the result becomes part of the
  notebook's own cell output, and nothing is left in the repo.
- `inference_recognizer`/`inference_skeleton` only populate a `pred_score` field on their result — the
  visualizer (`ActionVisualizer.add_datasample`) separately expects a `pred_labels` field
  (`mmengine.structures.LabelData(item=<top-k indices>, score=<full score tensor>)`) to draw prediction text at
  all. Build that field yourself before visualizing; it does not exist by default.

## Testing / lint

- Library code (`mmaction/`): this repo's own pre-commit config uses `flake8`, `isort`, `yapf` — run them via
  `.venv/bin/python -m flake8 <files>` / `.venv/bin/python -m isort --check <files>` /
  `.venv/bin/python -m yapf --diff <files>`. Run the relevant `pytest` tests under `tests/` for anything touched
  in `mmaction/` (e.g. touching `mmaction/datasets/transforms/pose_transforms.py` →
  `pytest tests/datasets/transforms/`).
- Notebooks: no lint/type gate — the bar is "runs end-to-end and produces a sane, verified result." Verify by
  actually executing the affected cells (e.g. via `.venv/bin/python -c "..."` reproducing the cell logic, or
  running the notebook), not by inspection alone.
- Config files (anything under `configs/`): after adding/moving one, confirm it still parses —
  `mmengine.Config.fromfile(path)` — especially after a path move, since nothing else will catch a silently
  broken reference.
