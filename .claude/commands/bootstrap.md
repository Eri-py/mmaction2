---
description: Bootstrap the dev environment
---

# Bootstrap Dev Environment

Walk the user through first-time setup of this project's dev environment, or recovery after the `.venv` is wiped
or corrupted. This repo's setup story is already known and unusually involved — follow it rather than
rediscovering it from scratch, but still verify each step actually worked rather than assuming.

## Known setup story

1. **Python env is `uv`-managed, never system/apt.** `uv venv --python 3.12` at the repo root, then everything
   via `uv pip install -p .venv ...`. If `.venv` doesn't exist yet, create it first.
2. **`torch` must be installed for this machine's actual CUDA version**, not whatever `uv pip install torch`
   defaults to (CPU-only from plain PyPI). Check `torch.version.cuda` against what the GPU driver actually
   supports before assuming a cached instruction is still current — CUDA tags move. Install from
   `https://download.pytorch.org/whl/cu<XXX>` explicitly.
3. **`mmcv`, `mmdet`, `mmpose` need real compiled CUDA extensions for this GPU (RTX 5070, Blackwell, `sm_120`)**
   — no prebuilt wheel supports it, so they're compiled from source. Building `mmcv` specifically requires, in
   order: pinning `setuptools==75.8.0` (newer versions drop `pkg_resources`, which `mim` needs, but also break
   compiling *because* the isolated build env pulls its own fresh setuptools regardless — pass
   `--no-build-isolation` too), installing `nvidia-cuda-nvcc`/`nvidia-cuda-cccl`/`nvidia-nvvm` **all pinned to the
   exact same `X.Y.Z`** matching `torch.version.cuda` (version drift between these three silently corrupts the
   build — a `ptxas`/PTX-ISA mismatch, not an obvious error), and manually symlinking unversioned `.so` files in
   `site-packages/nvidia/cu13/lib/` since the pip packages only ship versioned ones (the linker needs the
   unversioned name). See `CLAUDE.md` for the fragile spots that persist *after* a successful build (`moviepy`
   version, NumPy 2.0 patches).
4. **`mmaction2` itself needs `pip install -e . --no-deps`** — not just having `mmcv`/`mmengine` installed —
   or `import mmaction` only works by accident from `python -c` one-liners run at the repo root.
5. **`importlib_metadata` is an mmaction2 dependency that isn't declared anywhere** (a real upstream gap,
   normally masked by something else pulling it in transitively) — install it directly if
   `mmaction/models/__init__.py` fails on it.

## Step 1 — Check whether the environment already works

Before doing anything, try the actual thing this bootstrap is for:

```bash
.venv/bin/python -c "
import torch, mmcv, mmdet, mmpose, mmaction
print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmpose', mmpose.__version__)
print('mmaction', mmaction.__version__)
"
```

If this succeeds and `cuda ok: True`, the environment is already fine — don't rebuild anything. Skip to Step 4
to just verify the GPU path actually works, not merely imports cleanly.

## Step 2 — Rebuild what's missing

Work through the "Known setup story" above in order, checking each step's actual result before moving to the
next (don't assume success). If a step's exact commands aren't remembered precisely, they're a normal `uv pip
install -p .venv <pkg>` — the ordering and pinning constraints above are what actually matter, not exotic
flags beyond what's already listed.

If something fails in a way not covered above, treat it as a new discovery: read the actual error (the real
cause is often a compiler/linker line further up the log, not the Python traceback around it), fix it minimally,
and add it to `CLAUDE.md`'s gotcha list once confirmed — don't let the next session rediscover it blind.

## Step 3 — Re-pin the recurring drift

Even on a partially-working environment, explicitly check for the two most common regressions before declaring
done:

```bash
.venv/bin/python -c "import pkg_resources" 2>&1  # if this fails, uv pip install -p .venv "setuptools==75.8.0"
.venv/bin/python -c "import moviepy.editor" 2>&1  # if this fails, uv pip install -p .venv "moviepy==1.0.3"
```

## Step 4 — Verify on the actual GPU, not just imports

Import success is not sufficient proof the compiled kernels work — verify a real CUDA op runs and returns a
sane result:

```bash
.venv/bin/python -c "
import torch, mmcv.ops
boxes = torch.tensor([[0,0,10,10,0.9],[1,1,10,10,0.8]], dtype=torch.float32, device='cuda')
print(mmcv.ops.nms(boxes[:, :4], boxes[:, 4], 0.5))
"
```

One box kept (the higher-confidence one), not both, running on `device='cuda:0'` — that's the real bar, not
just "no exception."

## Recovery note

If the user says they wiped `.venv` entirely, this whole file *is* the recovery procedure — there's no separate
path. Rebuild from Step 2, don't skip the CUDA-toolchain steps assuming they're a one-time thing.

## What not to do

- Don't reach for a full system CUDA Toolkit install (`apt install nvidia-cuda-toolkit` or similar) — this
  environment deliberately uses the lightweight `nvidia-cuda-nvcc`/`nvidia-cuda-cccl`/`nvidia-nvvm` pip packages
  instead, and mixing in a system toolkit risks a second, conflicting `nvcc`.
- Don't downgrade NumPy to work around the `np.Inf` issue — the fix is patching the two call sites (already done
  in this repo; see `CLAUDE.md`), not reverting the NumPy version, which the rest of the stack depends on being
  current.
- Don't install any Python package via `apt`/system package manager, ever, even "just for this one tool."
