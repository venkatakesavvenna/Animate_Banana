# Pointing

Image -> per-object points (node points + raster/photo-region points).
Multiple pointing models are being explored, so each has its own module:

| Module | Runner | Notes |
|---|---|---|
| [molmo_point.py](molmo_point.py) | `MolmoPointRunner` | `allenai/MolmoPoint-8B`, the pointing-specialist checkpoint. Points decoded via `model.extract_image_points(...)` with per-request processor metadata -- needs the exact call sequence documented in the module. |
| [molmo2.py](molmo2.py) | `Molmo2PointRunner` | `allenai/Molmo2-8B`, general-purpose VLM with pointing support. Points come back as plain text tags, decoded with a regex -- simpler integration, may generalize better on diagrams whose nodes aren't clean geometric shapes. |

[common.py](common.py) holds the shared `PointResult` type and the node/raster
query prompts both runners use -- **the query text has been tuned by hand,
don't change it without explicit instruction** (see the "DO NOT CHANGE"
comments in that file).

[models.py](models.py) holds `POINTING_MODELS`, the registry both runners
are looked up from via `--pointing-model`.

`molmo2.py::make_point_runner(spec)` dispatches to the right runner class
based on which checkpoint is registered, so callers (`run_data_engine.py`)
don't need to know which pointing method a given `--pointing-model` key maps
to.

**Both models require the dedicated `/environments/molmo_point` venv**
(`transformers==4.57.1` pinned) rather than the main
`img_2_svg_pretraining` venv (`transformers==5.13.0`+, which breaks their
bundled remote code) -- see molmo_point.py's module docstring for why.
