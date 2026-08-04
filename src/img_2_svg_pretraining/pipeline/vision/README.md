# Vision layer (Stage 1b)

Locating the real imagery that belongs in the transmuter's raster placeholders, so it can be
cropped from the source figure and spliced into the diagram code.

**One VLM call per sample.** The model is shown the source figure plus the list of
placeholders Stage 1a emitted, and returns a bounding box for each one it can locate.

## Why one call

This was a five-step chain: Molmo2 proposed points, a Set-of-Mark pass filtered them, a
fine-tuned SAM3 checkpoint segmented the survivors, and a second SoM pass mapped crops back
to placeholders. It worked, but it needed two vision checkpoints resident on GPU under a
second venv — Molmo2 requires `transformers==4.57.1`, SAM3 requires 5.x, and neither can
share a process with the pipeline — so a run could never hold only one model at a time.

The collapse is possible because **Stage 1a already says what to look for**. Its placeholders
carry an `xml id` and the label text of what belongs there, so the detector can be asked to
find *those specific things* rather than propose regions blind and have a second pass decide
where each one goes. Detection and mapping become the same answer.

That also removes a failure mode rather than relocating it: two passes could disagree, and
the mapping pass could confidently pair the wrong crop with the wrong placeholder. Now the
model returns the `xml id` it matched, so there is nothing to reconcile.

## Coordinate convention

Boxes come back in Gemini's convention — `[ymin, xmin, ymax, xmax]`, each value an integer on
a **0–1000 grid**, normalized per-axis against the image's own dimensions regardless of
aspect ratio. Note the order: it is *not* `[x0, y0, x1, y1]`, and getting it wrong transposes
every crop while still producing a plausible-looking box.

`gemini_boxes.to_pixels` is the only place that convention is decoded; everything downstream
sees ordinary source-image pixel boxes. It clamps out-of-range values (a few units past the
grid edge still locates the region correctly), recovers swapped corners, and rejects
degenerate boxes.

Two spaces remain in play, and the source-image one is now the only one that matters for
cropping:

| Space | Extent | Used by |
|---|---|---|
| Source image pixels | e.g. 978×169 | detected boxes, crops, sub-panel expansion |
| TikZ units | same as source (the transmuter's `x=1pt, y=-1pt` convention) | placeholder positions |

## Config

```yaml
transmuter:
  raster_integrator:
    enabled: true
    backend: gemini_flash     # any chat backend; the same one other agents use
    params:
      temperature: 0.0        # localisation, not generation
      max_tokens: 4096
```

No `vision_python`, `point_model`, `sam_checkpoint`, `max_points` or `gpu` — those keys are
retired and now fail config validation.

## The raster/vector boundary

The detection prompt tells the model to **omit** any placeholder whose content is line art —
a plain box, an arrow, a text label, a simple geometric shape. This is the same rule the
transmuter prompt is built around, and it is the reason "N placeholders filled / M total" is
**not** a quality metric to maximise.

The old two-pass route demonstrated the point concretely. On `CVPR_2025_pipe00002` it
accepted 3 of 7 and rejected 4, with reasons: the camera frustums and the points-in-a-box
were "simple vector diagram", while the signed-distance field, the scalar-field grid and the
indicator function were genuine heatmaps. Wireframes and point sets are line art TikZ redraws
faithfully; heatmaps genuinely cannot be.

⚠️ **The single-call design has a structural bias toward filling.** Naming a placeholder in
the prompt invites the model to find something for it, where the old route had a step whose
only job was to say no. On the first run of `CVPR_2025_pipe00002` it filled **7 of 7**,
including the three panels the old route had rejected as line art — `omitted` and `rejected`
both came back empty, so that path is currently unexercised.

The counterargument is that Stage 1a already decided those regions were raster-worthy when it
drew them as `raster_node`, so 1b honouring that is defensible. But it means the omission
behaviour is unverified, and a sample with many placeholders is the test that would exercise
it.

## Sub-panel boxes

A detector asked for "the region that belongs here" can still return one panel of a composite
graphic — a 2×2 grid of heatmaps, a figure with sub-plots — rather than the whole thing.
Observed with the old route on `CVPR_2025_pipe00002`: the four-panel Φ(x,δ) graphic came back
as its bottom row only, 57px tall against neighbours at ~109px.

`_expand_to_placeholder` corrects this using the placeholder's declared box as a prior for how
much space the graphic was expected to occupy. It only fires on a clear aspect-ratio mismatch
(outside 0.75–1.35×), only ever grows, and clamps to the image. Correctly-sized boxes pass
through untouched.

The prompt now also asks for the whole composite explicitly, and on the first single-call run
the Φ(x,δ) grid came back complete — so the guard did not need to fire. It is kept because
the prompt is a request, not a guarantee.

## Failure behaviour

A failed or unparseable detection call yields **no regions**, and the placeholder version of
the code is passed through to `code_final` unchanged. Splicing the wrong crop corrupts the
diagram; splicing nothing leaves it exactly as Stage 1a drew it.

The two cases are reported differently, because they are not the same event:

| Outcome | Status | Meaning |
|---|---|---|
| Call failed / unparseable | `unresolved` | no filter ran; `detections.json` carries `error` |
| Call ran, located nothing | `ok` | the model looked and correctly declined |

Without that split a broken run looks identical to a clean one that found nothing.

## Artifacts

Written to `cache/<dataset>/rasters/<lineage>/<sample_id>/`:

```
detected.png         the source figure with every detected box drawn on it
regions.json         xml_id, raw box_2d, decoded pixel bbox, label
crop_<xml_id>.png    the crop spliced into that placeholder
detections.json      counts, omissions with reasons, rejects, replaced ids, provenance
```

`detected.png` is the fastest way to diagnose a bad result — it shows exactly what was
located, in the source figure's own coordinates.
