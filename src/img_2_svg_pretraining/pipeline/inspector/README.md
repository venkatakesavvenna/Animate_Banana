# Pipeline Inspector

Browse every intermediate artifact of a pipeline run — source figure, reconstructed diagram,
structure XML, strategy, animation sequence, animation code, and the rendered frames — for one
sample at a time.

Because each agent writes its output to disk, a finished run is fully inspectable after the
fact. This just serves those files side by side.

## Running

```bash
docker exec -it img-2-svg-pretraining-singlenode-venkat.kesav bash
source /environments/img_2_svg_pretraining/bin/activate && cd /code
python -m img_2_svg_pretraining.pipeline.inspector.app
```

Then open `http://<host>:7860`.

Defaults to **7860**, the only published port not taken by the annotation tool's Streamlit
ports (8600–8602). The annotation viewer also uses 7860, so run only one at a time. Using a
different port means publishing it at container *creation* — `docker run -p` has no effect on
an existing container, so it needs `docker rm -f` plus a re-run of `docker/init.sh` (safe:
code, data and venvs live on mounts).

`--config` selects which run to inspect; artifact paths encode the model lineage, so pointing
at a different config shows that config's run.

## What it shows

| Panel | Content |
|---|---|
| Sample list | every sample with an artifact-completion bar and frame count |
| Source figure / reconstruction | the input PNG beside the compiled stage-1a output |
| Animation | two frame-viewing modes (below) plus download links for PDF/MP4/GIF/PPTX |
| Sequence | each traversal step in order — id, depth, action, duration, narration, focus element ids |
| Artifacts | every stage's file with size and presence; click a row to read it |

Absent artifacts are listed rather than hidden, so a gap in a run is visible instead of
silently missing.

## Going through frames

Two modes, toggled at the top right of the Animation panel:

- **Scrub** — one frame at a time with a slider, play/pause (loops, ~2 fps), step buttons,
  and keyboard control: <kbd>←</kbd> / <kbd>→</kbd> to step, <kbd>space</kbd> to play. Best
  for checking exactly what appears at a given step.
- **Contact sheet** — every frame at once as a numbered grid. Best for seeing the whole
  progression, spotting a step that reveals nothing, or finding where something goes wrong.
  Clicking a thumbnail switches to scrub mode at that frame.

The sheet uses cached JPEG thumbnails (`frames/thumbs/`, built on first request) — frames are
300 dpi PNGs, so a 41-frame sheet would otherwise be tens of megabytes.

The panel also prints the on-disk `frames/` path, for when you want the PNGs directly.

## Endpoints

`/api/samples`, `/api/sample/<id>`, `/api/figure/<id>`, `/api/render/<id>` (compiles on
demand), `/api/frames/<id>` (count + on-disk path), `/api/frame/<id>/<n>`,
`/api/thumb/<id>/<n>`, `/api/export/<id>/<name>`, `/api/marked/<id>` (the Set-of-Mark overlay
the sequence critic saw).
