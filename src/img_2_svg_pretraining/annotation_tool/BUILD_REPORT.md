# Node annotation tool — build & verification report (2026-07-10)

Implementation report for the annotation framework design doc: a Streamlit +
SAM3 human-in-the-loop tool producing, per diagram node, a human-confirmed
point + human-confirmed pixel mask. This supervision feeds joint fine-tuning
of Molmo (node pointing) and SAM3 (node masking), since neither works out of
the box for "node" — a structural role, not a visual category.

**Status: built, verified, live at `http://<host>:8600`**, with the first
20 images of `data/train/original_images` ingested (164 Molmo-proposed
instances awaiting review in `data/annotations/`).

---

## What was built — `src/img_2_svg_pretraining/annotation_tool/`

| Module | Role |
|---|---|
| `datamodel.py` | Spec section-5 dataclasses (`Point`, `EditEvent`, `Instance`, `SessionStats`, `ImageRecord`), the `transition_instance` state machine (the **only** place `instance.state` changes), COCO-RLE mask helpers |
| `store.py` | One JSON per image in `data/annotations/`, `filelock`-guarded, atomic replace |
| `sam3_backend.py` | Embed-once-per-image / segment-per-click wrapper on `Sam3TrackerModel` — SAM3's point/box **interactive** mode only, never the text/concept mode. Logs `SAM3_LOAD` / `SAM3_EMBED` so the caching acceptance criteria are countable |
| `pointops.py` | Pure click-resolution logic (select → two-click move → add) plus point mutations — factored out and unit-tested because it's the most failure-prone part |
| `compositor.py` | PIL canvas compositing: masks + points always on one canvas, fixed-padding crop zoom (100/200/400%), instance thumbnails |
| `app.py` | The Streamlit review app: toolbar, canvas via `streamlit-image-coordinates`, instance list with state badges, status chips, session stats, undo/redo (cap 20), shortcuts `a` / `c` / `n` / `Ctrl+Z` |
| `ingest.py` | Offline Molmo proposal batch script (never a UI button); skips already-annotated images unless `--force`; runs in the `molmo_point` venv |
| `tests/` | 18 tests: state machine, RLE round-trip, serialization, persistence, concurrent writes, click resolution, `can_accept` rules |

`sam3_playground/` is **deleted** (this tool replaces it). `docker/init.sh`
renamed `SAM3_PLAYGROUND_PORT` → `ANNOTATOR_PORT`; still 8600, so no
container recreation was needed.

New deps added to `pyproject.toml`: `streamlit-shortcuts`,
`streamlit-drawable-canvas` (installed, currently unused — see deviations),
`pycocotools`, `filelock`.

## The loop (as implemented)

1. **Molmo proposes points** — `ingest.py`, offline, one `proposed` instance
   per point, no mask.
2. **SAM3 turns points into masks** — live in the app, recomputed
   automatically on every point add/move/delete; no "run segmentation"
   button.
3. **Humans correct both** — one instance at a time:
   `proposed → needs_point_review → (Confirm points, "c") →
   needs_mask_review → (Accept, "a") → accepted`; Reject / Merge from point
   review; reopen always re-enters point review. Accepts write JSON
   immediately.

## Verification results (all run inside the docker container)

| Criterion | Result | How verified |
|---|---|---|
| 1. `load_sam3()` once per process | **PASS** | `AppTest` harness counting `SAM3_LOAD` log records across 5 app runs → 1 |
| 2. Embed once per image, not per rerun | **PASS** | `SAM3_EMBED` count = 1 after 5 reruns on one image, = 2 after switching image |
| 3. Click→mask < 1.5 s p95 | **PASS** (server-side) | Measured directly on a real train image: ~22 ms warm per click; embed 0.51 s once per image |
| 4. Accept impossible with zero positive points | **PASS** | Unit tests on `can_accept` + `AppTest` assert the Accept button's `disabled` flag with a negatives-only instance |
| 5. Resumability (5 accepted / 2 unresolved survive reopen) | **PASS** | Fixture JSON written, fresh `AppTest` session shows 5 accepted / 2 unresolved |
| 6. Accepted ⇒ logged human point action | **PASS** | `has_human_point_action` backstop in `accept_active()` + unit tests |
| 7. Shortcuts without widget focus | **manual** | `streamlit-shortcuts` `shortcut_button` binds globally — needs a browser check |
| 8. Two processes, no file corruption | **PASS** | `test_concurrent_writes_do_not_corrupt`: 3 processes × 25 writes, same and different images, all files valid |
| 9. No OCR dependency | **PASS** | grep over the package + `pyproject.toml`: only documentation mentions of "no OCR" |

18/18 unit tests pass. Composite rendering was eyeballed on a real ingested
image (flagged Molmo point drawn as red dashed ring in the right place).

**Still needs a human in a browser:** actual click flows (add / select /
two-click move), criterion 7, and perceived client-side latency.

## Deviations from the design doc (both deliberate)

1. **SAM3 backend = transformers `Sam3TrackerModel`, not Roboflow
   `inference`** — confirmed with the owner 2026-07-10 (self-hosted
   requirement kept; no image data leaves the machine). The already-verified
   transformers classes satisfy the doc's real constraints (interactive
   mode, embed/predict split via `get_image_embeddings` +
   `image_embeddings`) without installing the `inference` dep tree into the
   NGC-pinned venv. The doc's warning about prompt-shape drift was correct:
   this transformers version (5.3.0) wants **3-level box nesting**
   `[[x0,y0,x1,y1]]`; the old playground's 4-level shape now raises.
2. **Manual-box fallback = two corner clicks on the same canvas**, not
   `streamlit-drawable-canvas` — that package (0.9.3) is unmaintained
   against Streamlit 1.59. It is installed; the box logic is isolated in one
   `box_mode` branch of `app.py` if a drawing surface is preferred later.
   The box is fed to SAM3 as a box prompt and stored with
   `mask_source="human_box"`, per the doc.

## Streamlit gotchas encoded in app.py (don't regress)

- Image selectbox is keyed per current image (`imgsel_{id}`) so Prev/Next/
  Next-flagged navigation isn't overridden by stale widget state.
- The `n` shortcut flips negative mode via a `_pending_neg_flip` flag applied
  at the top of the script — widget-backed keys can't be written after their
  widget is instantiated in a run.
- Canvas clicks are deduped against `last_click_coords` because
  `streamlit-image-coordinates` re-returns the last value on unrelated
  reruns.
- GPU 0 on this shared host is usually crowded — run ingest with
  `CUDA_VISIBLE_DEVICES=1` (first attempt OOM'd on GPU 0).

## How to run

Review app (main venv; port 8600 already published):

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "cd /code && streamlit run src/img_2_svg_pretraining/annotation_tool/app.py \
   --server.port 8600 --server.address 0.0.0.0"
```

Ingest more images (Molmo venv, `transformers==4.57.1`):

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "source /environments/molmo_point/bin/activate && cd /code && \
   CUDA_VISIBLE_DEVICES=1 python -m img_2_svg_pretraining.annotation_tool.ingest \
     --images-dir data/train/original_images --limit 40"
```

Tests:

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash -c \
  "cd /code && python -m pytest src/img_2_svg_pretraining/annotation_tool/tests/ -q"
```
