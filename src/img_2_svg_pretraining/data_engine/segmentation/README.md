# Segmentation

Point/image -> per-node masks. Multiple methods are being explored for this
stage, so each lives in its own module rather than one file trying to
support every approach:

| Module | Method | Input | Notes |
|---|---|---|---|
| [sam3_tracker.py](sam3_tracker.py) | `Sam3Runner` | one point per object | Point-promptable SAM3 (`Sam3TrackerModel`/`Sam3TrackerProcessor`, `facebook/sam3`). Used by the main `run_data_engine.py segment` stage today. |
| [sam2_amg.py](sam2_amg.py) | `Sam2AmgRunner` | image only, no points | SAM2 automatic mask generation (`facebook/sam2.1-hiera-large`). Explored 2026-07-08: found correct box-level masks on every diagram style tested, including ones that broke both other methods, with no per-image tuning. Over-segments text glyphs at default settings (filterable by mask area/rank downstream, not yet wired into the main pipeline). |
| [classical_cv.py](classical_cv.py) | point-seeded flood-fill + contour cross-check | one point per object | No ML/GPU. Fast and fully deterministic on flat-fill, solid-dark-stroke boxes, but a single global Otsu threshold misses low-contrast/light-colored borders on some real diagrams (fails safe to `needs_review` rather than wrong, but leaves real nodes unresolved). CLI: `segment_nodes.py` at the project root. See that module's docstring for the full writeup and `../tests/test_cv_segmentation.py` for its test. |

[models.py](models.py) holds the registries (`SEGMENTATION_MODELS` for
SAM3, `AMG_MODELS` for SAM2) -- separate from
`img_2_svg_pretraining.benchmark.models.ModelSpec` because these models take
image(+points) in and masks out, not image+text-in/text-out like the chat
VLMs the benchmark registry covers.

## Which one is "the" segmentation stage today

`run_data_engine.py segment` uses `sam3_tracker.py` (point-promptable,
matching the pointing stage's per-object points 1:1). `sam2_amg.py` and
`classical_cv.py` are alternatives under evaluation, not yet wired into the
main stage-batch pipeline -- see each module's docstring for current
status/tradeoffs before picking one for a new experiment.
