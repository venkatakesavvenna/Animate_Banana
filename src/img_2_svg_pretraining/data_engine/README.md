# Data Engine

Automated pipeline that takes a raw diagram image and produces (a) a
structured layout XML describing its nodes, raster sub-images, and edges,
and (b) generated code (TikZ or SVG, backend-selectable) rendering that
diagram, filtered for visual fidelity by a VLM judge.

Where the `benchmark/` module evaluates *existing* image->TikZ ground truth,
this module *generates* new (image, XML, code) triples from images alone --
turning `data/partial_test_set`'s 100 hand-authored XML/tex pairs into a
schema reference and comparison target rather than the only source of this
kind of data.

## Pipeline stages

Several stages have more than one method under active exploration -- those
stages are folders (one module per method) rather than a single file. See
each folder's own README for the full method comparison.

1. **Node/region discovery** ([pointing/](pointing/)) -- point-prompted VLMs
   point at every distinct labeled node/box (one query) and every embedded
   raster/photo region (a separate query). Two methods: `pointing/molmo_point.py`
   (`allenai/MolmoPoint-8B`, pointing specialist) and `pointing/molmo2.py`
   (`allenai/Molmo2-8B`, general-purpose, currently the default). See
   [pointing/README.md](pointing/README.md).
2. **Segmentation** ([segmentation/](segmentation/)) -- point/image -> per-node
   masks. Three methods explored: `segmentation/sam3_tracker.py` (point-promptable
   SAM3, used by the main pipeline today), `segmentation/sam2_amg.py` (SAM2
   automatic mask generation, image-only, found the most robust across
   diagram styles), `segmentation/classical_cv.py` (flood-fill, no ML/GPU,
   fast but brittle on low-contrast borders). See
   [segmentation/README.md](segmentation/README.md).
3. **Assembly** ([assemble.py](assemble.py)) -- pure geometry, no model
   calls: classifies each detected node as a top-level `Node` or a `Block`
   (any node whose bbox strictly contains other detected nodes), and nests
   detected raster regions under their containing block if any. Arrows are
   left empty.
4. **Edge discovery** ([edges.py](edges.py)) -- Set-of-Mark prompting: draws
   a numbered marker at each node/block's bbox center, asks a chat VLM (any
   `benchmark.models` registry entry) which numbered nodes connect and in
   what direction, parses the response back into `Arrow` entries.
5. **Code generation** ([codegen.py](codegen.py)) -- given the source image
   + the assembled XML as structural grounding, prompts a chat VLM for TikZ
   or SVG source (`--backend`), extracting the code block the same way
   `benchmark/infer.py::_extract_tikz` does (adapted for SVG too).
6. **Render** ([render.py](render.py)) -- tikz dispatches to
   `viewer/compile.py::compile_tikz` (latexmk + pdftoppm, already used by the
   benchmark harness); svg dispatches to `cairosvg` (pure-Python, no LaTeX
   toolchain needed).
7. **Judge filter** ([judge.py](judge.py)) -- thin reuse of
   `benchmark/metrics/judge.py::JudgeRunner` (unchanged, since it already
   just compares two images): renders the generated code, scores it against
   the source image 1-5, keeps samples scoring >= `--keep-threshold`
   (default 4).

[run_data_engine.py](run_data_engine.py) drives all 7 stages as **independent,
re-runnable subcommands**, not one per-sample pipeline. Each subcommand runs
over the *whole* sample set before the next stage starts, handing off through
on-disk artifacts under `data_engine/cache/` (points JSON, mask PNGs,
pre-edges XML, final XML, generated code, rendered PNG) rather than holding
every model resident in one process and looping per sample. This means only
the model(s) a given stage needs are ever loaded at once, and any stage can
be re-run alone (e.g. re-run `edges` after tweaking the Set-of-Mark prompt)
without recomputing earlier stages. See "Running" below for the per-stage
commands and cache layout.

## Schema

[schema.py](schema.py) extends the hand-authored format in
`data/partial_test_set/xml_files` (`<diagram><block i t b><node i t b/>
<arrow s d/></block>...</diagram>`) rather than replacing it: all 100
existing files round-trip through `parse_xml`/`to_xml` unchanged (verified).
New pipeline-only attributes (`mask`, `conf`, `src`) are optional on every
tag, and a new `<raster>` leaf represents embedded photo/plot regions that
hand files never contain. This keeps generated XML directly comparable to
the hand-authored ground truth for the same sample id.

## Running

Everything runs inside the project docker container (needs transformers/
torch, the LaTeX/poppler toolchain for the tikz backend, and `cairosvg` for
the svg backend -- already added to `pyproject.toml`).

Each stage is its own subcommand, run in order, each reading the previous
stage's cache and writing its own:

```bash
python -m img_2_svg_pretraining.data_engine.run_data_engine point    --limit 8
python -m img_2_svg_pretraining.data_engine.run_data_engine segment  --limit 8
python -m img_2_svg_pretraining.data_engine.run_data_engine assemble --limit 8
python -m img_2_svg_pretraining.data_engine.run_data_engine edges    --limit 8
python -m img_2_svg_pretraining.data_engine.run_data_engine codegen  --limit 8 --backend tikz
python -m img_2_svg_pretraining.data_engine.run_data_engine render   --limit 8 --backend tikz
python -m img_2_svg_pretraining.data_engine.run_data_engine judge    --limit 8 --backend tikz
```

- `--limit 8` (default): first milestone is a thin end-to-end pass over a
  handful of samples with manual inspection of every intermediate artifact,
  before scaling up. Use `--limit 0` for the full `partial_test_set` (100
  samples) once the harness is validated, then discuss scaling to `Set-2`
  (8,476 samples) separately.
- `--backend tikz|svg` (on `codegen`/`render`/`judge`) selects the code-gen
  target; both share every stage except `codegen.py`'s prompt instructions
  and `render.py`'s render call.
- `--pointing-model` / `--segmentation-model` are keys into
  [pointing/models.py](pointing/models.py) / [segmentation/models.py](segmentation/models.py)
  (currently `molmo-point-8b`/`molmo2-8b` and `sam3`). `--edge-model` /
  `--codegen-model` / `--judge-model` are keys into `benchmark.models.MODELS`
  (reused as-is). `segment`/`assemble`/`edges` need the same
  `--pointing-model`/`--segmentation-model` value used in the earlier stage
  that produced the cache they read.
- `--gpu` pins a single physical GPU. Because each stage is a separate
  process, only that stage's model(s) are ever resident at once -- no
  multi-GPU data-parallel sharding yet, unlike `run_benchmark.py`.
- Each stage prints `ERROR <e>` per sample and moves on rather than aborting
  the whole batch -- rerun the same command after fixing an issue; already
  no-op stages (nothing to redo) aren't skipped automatically today, so a
  rerun redoes every sample in `--limit`.

Cache layout (handoff between stages) and final output:
```
data_engine/cache/points/<pointing-model>/<id>.json         # point stage
data_engine/cache/points/<pointing-model>/<id>.png           # point overlay
data_engine/cache/masks/<pointing>__<segmentation>/<id>.json # segment stage manifest
data_engine/cache/masks/<pointing>__<segmentation>/png/<id>/<node_id>_*.png
data_engine/cache/diagrams/pre_edges/<pointing>__<segmentation>/<id>.xml   # assemble stage
data_engine/cache/diagrams/final/<pointing>__<segmentation>__<edge-model>/<id>.xml  # edges stage
data_engine/cache/code/<backend>/<codegen-model>/<id>.{tex,svg}  # codegen stage
data_engine/cache/renders/<backend>/<hash>.png                   # render stage
data_engine/output/<backend>/summary.jsonl   # judge stage: per-sample score + kept flag
```

## Unverified model repo ids

Unlike `benchmark/models.py` (whose repo ids were confirmed against the HF
Hub API before being pinned), `allenai/MolmoPoint-8B` and `facebook/sam3`
were supplied directly and confirmed to exist + import correctly
(`Sam3TrackerModel`/`Sam3TrackerProcessor`, `transformers==5.13.0` in this
container) as of 2026-07-07, but haven't been run end-to-end yet -- the
first `--limit 8` run is the actual validation.
