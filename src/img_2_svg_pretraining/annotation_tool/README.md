# Node annotation tool

Human-in-the-loop annotation producing, per diagram node, a **human-confirmed
point + human-confirmed pixel mask**. This supervision feeds two fine-tunes:
Molmo (node pointing) and SAM3 (node masking) — neither works out of the box
for "node", which is a structural role, not a visual category. That's also
why SAM3 is used **only** through its point/box interactive mode
(`Sam3TrackerModel`, SAM2-compatible visual prompting), never the
text/concept-prompted `Sam3Model`. Replaces the old `sam3_playground`
(same port, 8600).

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
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "cd /code && streamlit run src/img_2_svg_pretraining/annotation_tool/app.py \
   --server.port 8600 --server.address 0.0.0.0"
```

Open `http://<host>:8600`.

Env vars: `IMAGES_DIR` (default `data/train/original_images`),
`ANNOTATIONS_DIR` (default `data/annotations`), `ANNOTATOR` (actor name in
edit logs). One Streamlit process per reviewer — don't share a process.

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

Tests: `tests/` — run inside the container:
`python -m pytest src/img_2_svg_pretraining/annotation_tool/tests/ -q`.
