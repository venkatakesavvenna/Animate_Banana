# img_2_svg_pretraining — AnimateBanana

Turning static, process-oriented scientific figures into animated visual explanations, plus the
tooling and data engine around that.

All code runs **inside the project docker container**, never on bare metal:

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash
source /environments/img_2_svg_pretraining/bin/activate && cd /code
```

## Modules

| Module | What it does |
|---|---|
| [pipeline/](src/img_2_svg_pretraining/pipeline/) | **The AnimateBanana pipeline** — figure → animation-aware code → animation plan → rendered animation |
| [pipeline/vision/](src/img_2_svg_pretraining/pipeline/vision/) | Molmo2 pointing + SAM3 segmentation + the two Set-of-Mark verification passes |
| [pipeline/inspector/](src/img_2_svg_pretraining/pipeline/inspector/) | Browse every intermediate artifact of a pipeline run |
| [data_engine/](src/img_2_svg_pretraining/data_engine/) | Image → layout XML → TikZ/SVG, VLM-filtered; the training-data generator |
| [annotation_tool/](src/img_2_svg_pretraining/annotation_tool/) | Streamlit node annotator (Molmo → SAM3 → human) |
| [viewer/](src/img_2_svg_pretraining/viewer/) | TikZ inspection/correction tool |
| [benchmark/](src/img_2_svg_pretraining/benchmark/) | Stage-1 image→TikZ evaluation (legacy; superseded by AnimateBench) |
| [training/](src/img_2_svg_pretraining/training/) | Model fine-tuning |

## The pipeline

Three stages, agents within each. Every agent is its own CLI subcommand, runs over the whole
sample set, and hands off through on-disk artifacts — so only the model a given agent needs is
ever loaded, and any agent can be re-run alone after a prompt change.

```
STAGE 1  DIAGRAM TRANSMUTER    image -> animation-aware code
                               + Molmo2/SAM3 raster integration, bracketed by
                                 two Gemini Set-of-Mark passes
STAGE 2  ANIMATION PLANNER     strategy + structure -> animation sequence -> critic
STAGE 3  DIAGRAM ANIMATOR      animation code -> critic -> PDF / MP4 / GIF / PPTX
```

```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline samples      # what's available
python -m img_2_svg_pretraining.pipeline.run_pipeline all --limit 1
python -m img_2_svg_pretraining.pipeline.inspector.app             # browse results :7860
```

Backends (`gemini`, `anthropic`, `openai_compat`/vLLM, `hf_local`) are chosen per agent in a
YAML config, so one run can mix a commercial API for one agent with a local server for another.

Full details: **[pipeline/README.md](src/img_2_svg_pretraining/pipeline/README.md)**.

## Viewing animation frames

An exported animation lands in
`pipeline/cache/<dataset>/exports/<lineage>/<sample_id>/`:

```
animation.pdf     one page per frame
animation.mp4     video
animation.gif     looping, easiest to preview
animation.pptx    one slide per frame
frames/           frame-01.png, frame-02.png, ...
```

Five ways to go through them, easiest first:

1. **Inspector** — `python -m img_2_svg_pretraining.pipeline.inspector.app`, open
   `http://<host>:7860`, pick a sample. *Scrub* mode gives a slider, play/pause, and
   ← → arrow keys; *contact sheet* mode shows every frame at once as a grid, and clicking a
   thumbnail jumps the scrubber to it. This is the one to reach for.
2. **The GIF** — `animation.gif` loops in any browser or image viewer; no tooling needed.
3. **The PDF** — one page per frame, so any PDF reader's page navigation steps through the
   animation, and it prints.
4. **The PPTX** — one slide per frame; advancing slides plays the animation in a talk.
5. **The PNGs** — `frames/frame-NN.png` for diffing, scripting, or dropping into a paper.

To find a sample's export directory without hunting through the lineage-encoded path:

```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline paths --sample CVPR_2025_pipe00010
```

## Data

`data/test_benchmark/` — 11 curated samples, one directory each:

```
<id>/  <id>.png  caption.txt  abstract.{tex,txt}  methods.{tex,txt}  arxiv_src/
```

The four context tiers (`image_only`, `caption`, `abstract`, `full`) map onto the strategizer's
four prompt variants, which makes context-ablation a first-class axis of the benchmark.

## Status

The pipeline runs end to end against live models. Stage 1b (raster integration via Molmo2+SAM)
is not built — Stage 1a alone produces compilable code with drawn placeholders, so nothing
downstream is blocked. AnimateBench (animation-plan evaluation) is designed but not built.

API keys live in a gitignored `api_keys.csv` at the repo root (`Key, Service` header); Gemini
quotas are per key, so all matching keys are pooled and rotated on rate-limit errors.
