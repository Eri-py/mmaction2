---
description: Documentation Maintainer
---

# Documentation Maintainer

Refresh this project's documentation set so it accurately describes the current state of the code — detect drift and update in place. Used periodically after significant changes.

## Step 0 — Discover the doc set

Before doing anything, build a working list of every doc file that actually exists:

- Root-level: `README.md`, `README_zh-CN.md`, `CLAUDE.md` (this repo's project-instruction file).
- Per-`configs/` category docs: each task folder under `configs/` (`recognition/`, `skeleton/`, `detection/`, ...) has its own model-family `README.md`s and `metafile.yml`s; `configs/person_detector/` and `configs/skeleton_coord/` have their own honest, non-mmaction2-model READMEs (see `CLAUDE.md` for why those two are different from the rest).
- `docs/en/`, `docs/zh_cn/` — the Sphinx doc tree.
- `demo/README.md` does not exist — `demo/` was deleted; do not recreate references to it.

For each, form a one-line hypothesis of its intended audience and scope (e.g. "human setup + ops", "this model family's benchmark numbers", "vendored-config disclosure"). If two docs make overlapping claims about the same thing, note that too — it's a sign the split needs clarifying with the user rather than guessed at.

`CLAUDE.md` is out of scope for this command by default — it changes rarely and has a different shape. If you notice something during the scan that should go there, surface it for the user rather than editing it yourself.

## Principles (non-negotiable)

1. **Split by audience.** Human-facing docs (setup, usage) stay separate from agent-facing reference docs (`CLAUDE.md`, config-folder READMEs disclosing vendored vs. mmaction2-trained models). Don't blend the two in one file.
2. **Every model config entry links to its source file (or, for vendored configs, its upstream source URL).** The docs double as a navigation map.
3. **Verify from the code. Never invent.** If a claim (a config's existence, a checkpoint URL, a benchmark number) isn't confirmed by a file you actually read or a live URL you actually checked, it doesn't go in the doc.
4. **Update in place; don't rewrite.** Preserve prose that's still correct. Touch only sections that have drifted.
5. **No duplication across docs.** If two docs would otherwise repeat the same table/fact, the more general doc should link to the more specific one instead of restating it.

## Step 1 — Read the existing docs

Read every file found in Step 0. Note their current claims.

## Step 2 — Detect drift

For each doc, compare its claims against the actual code:

- `configs/*/README.md` benchmark tables → confirm the config file still exists at the stated path, and (for `configs/person_detector/`, `configs/skeleton_coord/`) that the checkpoint URL still resolves (HTTP HEAD) and the benchmark numbers still match the cited upstream source.
- `notebooks/*.ipynb` references in any doc → confirm the notebook still exists and still contains what's described (e.g. still uses the same video/config paths).
- `CLAUDE.md`'s repo-layout section → confirm every listed top-level directory still exists and nothing new/significant is missing (e.g. don't let it silently drift out of sync with reality — but don't edit it yourself, see Step 4).

Work outward from what's actually written in each doc — don't assume any of the above categories apply unless the doc in question claims them.

## Step 3 — Surgical updates

For each stale section, edit only what's wrong. Keep prose that's still accurate. Add entries for new code; remove entries for deleted code (e.g. anything still referencing `demo/`, which no longer exists). Don't restructure a doc unless its structure itself is broken.

When adding config/model entries:
- One-line purpose description.
- Markdown link to the source file (relative path from the doc) or, for vendored configs, the upstream GitHub URL.

## Step 4 — Watch for cross-cutting concerns during the scan

If during the scan you notice something that looks like **behavioural guidance** (a non-obvious environment gotcha, a "don't do X" rule, a subtle pipeline assumption) rather than structural/reference fact, do not fold it into a structural doc. Either:
- It's specific to one config folder or notebook → add it to that folder's/notebook's own notes.
- It's cross-cutting behavioural guidance (an environment gotcha, a repo-wide convention) → surface it to the user at the end so they decide whether it belongs in `CLAUDE.md`.

Do not edit `CLAUDE.md` yourself.

## Final output

End with a short report:

- Files changed + which sections drifted (bullet list).
- Things you considered but left alone (e.g. "person_detector README benchmark numbers still checked out").
- Anything you flagged for `CLAUDE.md` (do not edit it yourself).
- Any gaps — missing files, missing sections, or parts of the code that have no documentation.

If nothing was stale, say so without editing anything.
