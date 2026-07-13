# Raster-region annotation tool

Human-in-the-loop annotation producing, per **raster region** embedded in a
diagram (photos, icons, logos, illustrations, screenshots, charts — NOT
structural diagram nodes like boxes/arrows), a **human-confirmed point +
human-confirmed pixel mask**. This supervision feeds two fine-tunes: Molmo
(raster-region pointing) and SAM3 (raster-region masking) — neither works
out of the box for this, since a raster region has no consistent
open-vocabulary description across diagrams. That's also why SAM3 is used
**only** through its point/box interactive mode (`Sam3TrackerModel`,
SAM2-compatible visual prompting), never the text/concept-prompted
`Sam3Model`. Replaces the old `sam3_playground` (same port, 8600).

Uses Molmo's `point_rasters()` / `RASTER_QUERY` for ingest — never
`point_nodes()`. Structural node annotation is a separate, not-yet-built
tool.

No OCR anywhere in this tool, by design — do not add text extraction or
duplicate-text checks to this build.

## The loop

1. **Molmo proposes points** — offline, via `ingest.py` (never a UI button).
2. **SAM3 turns points into masks** — live in the app, recomputed on every
   point add/move/delete; no "run segmentation" button.
3. **Humans correct both** — one instance at a time: fix the point(s),
   confirm, review the mask, accept (or fix with negative points / a manual
   box).

## Running

Ingest (Molmo venv — its remote code needs `transformers==4.57.1`):

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "source /environments/molmo_point/bin/activate && cd /code && \
   python -m img_2_svg_pretraining.annotation_tool.ingest \
     --images-dir data/train/original_images --limit 20"
```

Review app (main venv, port 8600 is published by docker/init.sh):

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c "cd /code && streamlit run src/img_2_svg_pretraining/annotation_tool/app.py --server.port 8600 --server.address 0.0.0.0"
```

Open `http://<host>:8600`.

Env vars: `IMAGES_DIR` (default `data/train/original_images`),
`ANNOTATIONS_DIR` (default `data/annotations`), `ANNOTATOR` (actor name in
edit logs). One Streamlit process per reviewer — don't share a process.

### Multiple concurrent reviewers + public links — `docker/start_annotators.sh`

`load_sam3()` is `@st.cache_resource` — cached **per process**, and
`Sam3InteractiveBackend` holds exactly one embedded image at a time (see
`sam3_backend.py`). Two reviewers sharing one Streamlit process would
thrash each other's embedding cache and contend for one GPU's inference
calls. So each reviewer gets **their own Streamlit process, pinned to its
own GPU** via `CUDA_VISIBLE_DEVICES`, on its own port, each fronted by its
own **Cloudflare quick-tunnel** for a shareable public HTTPS link. All
processes share the same `data/annotations/` and `data/train/original_images/`
— `filelock` in `store.py` already makes concurrent writes across processes
safe (see the concurrency test in `tests/test_datamodel.py`), so nothing
else needs coordinating.

One script does all of it — starts 3 annotators (GPUs 1/2/3, ports
8600/8601/8602), installs `cloudflared` if missing, starts a tunnel per
annotator, waits for everything to come up, and prints the 3 public URLs
ready to paste to reviewers:

```bash
docker/start_annotators.sh              # start (or reuse already-running)
docker/start_annotators.sh --restart    # kill everything and start fresh
docker/start_annotators.sh --stop       # stop everything
```

Run from bare metal (it shells into the container itself via `docker exec`).
Sample output:

```
=== Reviewer links (share these directly) ===
  Reviewer 1  (GPU 1, port 8600, app up):  https://<words>.trycloudflare.com
  Reviewer 2  (GPU 2, port 8601, app up):  https://<words>.trycloudflare.com
  Reviewer 3  (GPU 3, port 8602, app up):  https://<words>.trycloudflare.com
```

Everything runs inside the container in `tmux` sessions
(`annotator_8600`/`tunnel_8600`, etc. — one pair per port), so it survives
your shell disconnecting. Plain `--url` invocation with no flag reuses any
already-running annotator/tunnel instead of double-starting (checked via
tmux session name); `--restart` force-kills and restarts all six sessions.

**No authentication in front of these links by default** — anyone with a
URL can view and edit annotation data (unpublished paper figures) for as
long as that tunnel runs. Cloudflare's quick-tunnel mode also mints a new
random URL every restart, so treat these as short-lived sharing links, not
stable long-term ones — run `--stop` when a remote session is done rather
than leaving tunnels up unattended. For a stable URL and/or SSO-gated
access, that needs a named Cloudflare Tunnel + Cloudflare Access instead of
the quick-tunnel mode this script uses — a bigger setup, ask if needed.

Requires `docker/init.sh`'s 3-port publish (`ANNOTATOR_PORT`/`2`/`3` =
8600/8601/8602, added 2026-07-13 — needs a `docker rm -f` + re-run of
`init.sh` once if your container predates this; mounts mean nothing is
lost). To change the GPU/port mapping or reviewer count, edit the `PORTS`/
`GPUS` arrays at the top of `start_annotators.sh`.

## Using the app

- **Add point**: Add-point toggle on (default), click the canvas. First
  click with no active instance starts a new one. Toggle **Negative pt**
  (or press `n`) for negative points.
- **Move a point (two clicks)**: click it to select (amber ring), click its
  new location. Molmo points keep their original location in `original_xy`.
- **Delete / toggle ±**: select a point, use the right-panel buttons.
- **States**: `proposed → needs_point_review → (Confirm points, `c`) →
  needs_mask_review → (Accept, `a`) → accepted`; Reject / Merge from point
  review; reopening always re-enters point review.
- **Manual box (fallback only)**: Draw-box toggle, click two opposite
  corners; SAM3 re-predicts from the box (`mask_source="human_box"`). For
  genuine SAM3 failures with a correct point — not the first resort.
- **Undo**: `Ctrl+Z` (snapshots, cap 20). Accepts write JSON immediately.

## Storage

One JSON per image at `data/annotations/{image_id}.json`
(`filelock`-guarded, atomic replace). Masks are COCO compressed RLE
(`pycocotools`). Schema: see `datamodel.py` (`ImageRecord`, `Instance`,
`Point`, `EditEvent` — full edit log per instance, so every accepted
instance provably had human interaction).

## Debug export (no app, no model load)

`export_debug.py` renders every reviewed image's confirmed points + masks
onto the original image and dumps PNGs — for checking annotation quality
without opening the Streamlit app:

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "cd /code && python -m img_2_svg_pretraining.annotation_tool.export_debug"
```

Writes to `data/debug/annotation_export/` (override with `--out-dir`):
per image, `{image_id}_overlay.png` (original + one distinct color per
accepted instance's mask + its points, numbered; rejected instances shown
as gray outline circles for context) and `{image_id}_mask.png` (flat
label mask, accepted instances only, one gray level per instance rank), plus
one `summary.json` with per-image and dataset-wide accepted/rejected/
merged/unresolved counts. Images with nothing reviewed yet (still all
`proposed`) are skipped — this is a review-quality check, not a raw-Molmo
dump. Pure JSON/image I/O, no SAM3 or torch needed (masks are read straight
from their stored RLE).

## Modules

| file | what |
|---|---|
| `datamodel.py` | dataclasses, state machine (`transition_instance` — only place state changes), RLE |
| `store.py` | JSON persistence + locking |
| `sam3_backend.py` | embed-once-per-image / segment-per-click SAM3 wrapper (logs `SAM3_LOAD` / `SAM3_EMBED` for the caching acceptance checks) |
| `pointops.py` | pure click-resolution (select vs two-click move vs add) + point mutations |
| `compositor.py` | PIL canvas compositing, crop-zoom, thumbnails |
| `app.py` | the Streamlit review app |
| `ingest.py` | offline Molmo proposal batch script |
| `export_debug.py` | offline debug-viewer: dump reviewed points/masks to PNGs, no app/model needed |

Tests: `tests/` — run inside the container:
`python -m pytest src/img_2_svg_pretraining/annotation_tool/tests/ -q`.
