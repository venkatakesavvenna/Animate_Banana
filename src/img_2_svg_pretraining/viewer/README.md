# TikZ Annotation Tool

Annotator for correcting TikZ code against ground-truth architecture diagram
images (Stage 1 of the pipeline: diagram image -> TikZ), until the compiled
TikZ matches the ground-truth image.

Scans a data directory recursively for `.tex` files, pairs each with a
best-guess ground-truth image (same basename, or a conventional name like
`input.png`/`image.png` in the same folder, or the first image under a
`crops/` subfolder), and lets you inspect/edit/recompile the TikZ side by
side with the source image and a rendered preview.

## Motivation

This tool exists for three reasons:

1. **Verify/clean generated data.** The TikZ ground-truth corpus (e.g. the
   4000-sample `test` set) is machine-generated and not all of it compiles
   or matches its source diagram faithfully. The annotator is how we review
   and fix each sample by hand until it's trustworthy ground truth.
2. **Understand model failure modes.** Because the tool renders the
   ground-truth image, the compiled TikZ, and the TikZ source side by side,
   it's the fastest way to see *where* and *how* a model's image->TikZ
   output diverges from the target — missing elements, wrong layout,
   compile errors, etc. — not just aggregate accuracy numbers.
3. **A standard harness for every new model we test.** Whenever we evaluate
   a new MLLM on Stage 1, its output should be viewable through this same
   tool (ground truth vs. model output, side by side) so results are always
   inspected the same way, not just scored.

## Layout

**Ground Truth** (original diagram image) and **Rendered Output** (compiled
TikZ) are stacked top/bottom in the same left-hand column so they're
directly comparable at a glance — most of these architecture diagrams are
wide/landscape, so stacking makes better use of the available width than
placing them side by side. The **TikZ Source** editor is a separate column
on the far right, since it's the thing you edit only after spotting a
mismatch between the two image panes. The sample list sidebar is
deliberately narrow — it only needs to show sample IDs (truncated with an
ellipsis, full ID on hover) and a done/pending indicator, not take up space
that the image/render panes need more.

## Annotation workflow

- On load, pick an annotator identity from `user_1` / `user_2` / `user_3`
  (stored in the browser via `localStorage`, so it persists across reloads;
  use "switch" in the sidebar to change it). Any user can annotate any
  sample — there's no fixed assignment.
- Edits are saved to a SQLite DB (`viewer/annotations.db`), **not** to the
  original `.tex` files — the pristine dataset on disk is never overwritten.
  "Save Draft" stores your in-progress edit without changing status;
  "Revert to Original" discards edits back to the dataset's original tex.
- "Mark as Done" records the current tex as matching the ground-truth image
  and flips the sample to `done` (tracked per-sample, with which user did
  it and when). "Reopen" undoes this back to `pending`. A sample counts as
  complete once *any* one user marks it done — there's no requirement for
  all three to agree.
- The sidebar shows a progress bar (`done / total`, broken down by user)
  and a status filter (pending/done only) so annotators can pick up
  whatever's left.
- Once all 4000 are done, export the finished ground truth:
  ```bash
  curl http://<host>:7860/api/export -o ground_truth_export.zip
  ```
  This produces a zip of `tex_files/<id>.tex` (one per `done` sample) plus
  a `manifest.json` recording `annotated_by`/`updated_at` per sample for
  provenance. Samples not yet marked done are excluded.

## Run (inside the project docker container)

The viewer is an installed package module (`img_2_svg_pretraining.viewer.app`,
under `src/`) — after any pull, re-run `pip install -e /code` inside the
container if `src/` layout or dependencies changed.

First check whether it's already running (someone else may have started it):

```bash
docker exec img-2-svg-pretraining-singlenode-venkat.kesav bash -c "ps aux | grep '[i]mg_2_svg_pretraining.viewer.app'"
```

If nothing is running, start it. **Foreground** (from inside the container,
useful for seeing errors/logs live):

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash
python -m img_2_svg_pretraining.viewer.app --data-root /code/data/test_extracted/test --port 7860
```

**Background** (single command from bare metal, doesn't tie up a terminal):

```bash
docker exec -d img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "source /environments/img_2_svg_pretraining/bin/activate; python -m img_2_svg_pretraining.viewer.app --data-root /code/data/test_extracted/test --port 7860 > /tmp/viewer.log 2>&1"
```

Check it came up with `docker exec img-2-svg-pretraining-singlenode-venkat.kesav bash -c "cat /tmp/viewer.log"`.

Then open `http://<host>:7860` (see "Accessing from your local machine"
below if you're not physically on that host).

Point `--data-root` at whichever extracted dataset directory you want to
annotate (e.g. `/code/data/<extracted-set>`); the scanner does not assume
any particular naming convention beyond "a `.tex` file plus a nearby image".

Port `7860` is published from the container to its host via `-p 7860:7860`
in `docker/init.sh`, so once the viewer is running in the container, it's
already reachable at `http://<host>:7860` from anywhere that can reach the
host directly.

**To restart** (e.g. after a code change), stop the existing process first:

```bash
docker exec img-2-svg-pretraining-singlenode-venkat.kesav bash -c "pkill -f 'img_2_svg_pretraining.viewer.app'"
```

then relaunch with the background command above. Restarting does **not**
lose annotations — those live in `viewer/annotations.db`, not in memory.

## Accessing from your local machine

The dev machines (e.g. `vision-node-028`) are only reachable through an SSM
jump host, so `http://<host>:7860` isn't directly reachable from your laptop
browser — you need to forward the port over SSH first. An example
`~/.ssh/config` entry for a node:

```
Host bgen-cluster-c
    User venkat.kesav
    ProxyCommand sh -c "aws ssm start-session --target sagemaker-cluster:<cluster-id> --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"

Host vision-node-028
    HostName 10.20.238.191
    User venkat.kesav
    ProxyJump bgen-cluster-c
```

**Option A — VSCode Remote-SSH (if your VSCode window is connected to that
exact host)**: check the green remote indicator in the bottom-left of VSCode
— it should read `SSH: vision-node-028` (or whichever host matches the
machine actually running the viewer). If so, open the Command Palette
(`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Forward a Port** → `7860`. VSCode may
also auto-detect the listening port and pop up a notification with an
**Open in Browser** button; restarting the viewer process retriggers this
if it doesn't appear.

**Option B — manual SSH tunnel**: from a terminal on your local machine
(not inside the remote session):

```bash
ssh -L 7860:localhost:7860 vision-node-028
```

This logs in via the `bgen-cluster-c` ProxyJump automatically and tunnels
local port 7860 to port 7860 on the node. Leave the session open, then
browse `http://localhost:7860` locally.

If neither VSCode nor your `~/.ssh/config` is pointed at the exact host
running the viewer, first confirm which node the container is on: `hostname
-I` inside the session running the viewer must match the `HostName` in the
config entry you use.

## Requirements

Compiling TikZ requires a LaTeX toolchain (`latexmk`/`pdflatex`) and
poppler's `pdftoppm`, both installed via `docker/Dockerfile`. This is why the
viewer must be run inside the container, not on bare metal.

## Notes

- Compiled PNGs are cached in `viewer/cache/` keyed by a hash of the TikZ
  source, so re-viewing an unmodified sample is instant.
- `viewer/annotations.db` is gitignored — it's per-deployment runtime state.
  Back it up if you care about in-progress annotations (e.g. before wiping
  the container), since it's the only copy of edits until exported.
- Pass `--db-path` to point at a different SQLite file, e.g. to keep
  separate annotation DBs per dataset snapshot.
