# AnimateBanana — working notes

Figure → animated explainer. A three-stage pipeline plus a human-review tool
that sits on top of it.

---

## Running things

Everything runs from the repo root, inside a venv.

```bash
# the annotation tool
python -m img_2_svg_pretraining.pipeline.annotate.app \
  --config src/img_2_svg_pretraining/pipeline/configs/default.yaml \
  --port 8602 --styles progressive_reveal

# the pipeline
python -m img_2_svg_pretraining.pipeline.run_pipeline stage1 --config <cfg>
python -m img_2_svg_pretraining.pipeline.run_pipeline stage2 --config <cfg> --style progressive_reveal
python -m img_2_svg_pretraining.pipeline.run_pipeline all --from parse --config <cfg>
```

Setup, data format and ingestion: **[docs/ANNOTATOR_SETUP.md](docs/ANNOTATOR_SETUP.md)**.
How to annotate: **[docs/annotating.html](docs/annotating.html)**.

Install with `pip install -r requirements-annotate.txt` then
`pip install -e . --no-deps`. The `--no-deps` matters: `pyproject.toml` lists
torch/transformers/flash-attn for local open-weights inference, which the
annotation tool never imports (`hf_local.py` imports torch *inside* its loader).

---

## Stages

| id | agent | writes |
|---|---|---|
| 1a | `code_converter` | `code/` — image → diagram code |
| 1b | `raster_integrator` | `code_final/` — crops spliced in |
| 1c | `diagram_critic` | `code_reviewed/` — render, compare, repair |
| 2b | `parser` | `xml/` — structure hierarchy |
| 2c | `sequencer` | `sequence/` |
| 2d | `planner_critic` | `sequence_final/` |
| 2e | `narrative_writer` | `sequence_narrated/` |
| 3a | `designer` | `animation/` |
| 3b | `animator_critic` | `animation_final/` |
| 3c | `exporter` | `exports/` |

2a (`strategizer`) is **dropped** from `stage2`. The bench sequencer prompts
decide traversal order themselves. It still exists and `--from strategize`
resolves, but nothing runs it by default.

Artifacts live under `pipeline/cache/<dataset>/`, keyed by a **lineage** string
encoding the models that produced them, so two configs never collide. Style is
in `sequence_lineage` and everything downstream; target is in `code_lineage`.

---

## Things that bit us, and the rules that came from them

These are all real incidents. Each cost real time.

### Critics repair by deletion

Three separate times, a critic satisfied a finding by **removing the thing the
finding mentioned**:

- The sequence critic deleted **103 text references** across 7 samples because
  the validator falsely reported them absent.
- The diagram critic replaced two `\includegraphics` with maths labels,
  silently dropping hand-placed raster crops.
- The sequence critic flattened a node hierarchy to depth 1 to satisfy a depth
  check, discarding the sequencer's structure.

Both critic prompts now say explicitly: fix by *correcting*, never by deleting,
and if a finding contradicts what you can see, the finding is wrong. **If you
touch a critic prompt, keep those clauses.**

### Validators must not check against a partial id set

`focus` entries are validated against element ids. The parser is told to
*exclude* `text_node` elements from the XML; the sequencer is told the code is
its exclusive source for them. Validating against the XML alone reported every
correct text reference as missing.

Use `parser.referenceable_ids(xml_path, code)` — the union of both. Never
`load_element_ids` alone for focus validation.

### "Newest by existence" is not newest

`resolve_code` preferred `code_reviewed` over `code_final` purely because it
existed. A 3.5-day-old critique won over an 8-minute-old raster splice, and the
animation silently lost 20 images. It now skips a critique older than the
splice it reviewed.

The same class: `_swapped` used `shutil.copy2` on restore, which stamped
restore-time onto the file and made `stale_1b` report "fresh" when it wasn't.
It now preserves the original mtime with `os.utime`.

**Rule: when two artifacts describe the same thing, compare mtimes, not
existence.**

### Format conversion drops information

`_from_bench` copied element depths onto timestep nodes while hardcoding
`parent=None`, producing "root node at depth 2" on every nested step. The
hierarchy was recoverable from playback order; `_link_bench_parents` does it,
and `relink_orphan_depths()` repairs already-written files.

### Stale artifacts outlive their inputs

A frames-only compile left a 50-hour-old mp4 beside a fresh frame deck. The
raster screen parsed the raw 1a while detections described the spliced version
(7 ids vs 25, zero overlap, nothing drew).

**Rule: when rebuilding an artifact, evict everything derived from its previous
version.**

---

## The annotation tool

`pipeline/annotate/` — Flask, five screens, one shared gate component.

| screen | route | stage |
|---|---|---|
| D2C review | `/` | 1a |
| Rasters | `/rasters/<id>` | 1b |
| Structure XML | `/xml/<id>` | 2b |
| Sequence | `/sequence/<id>` | 2c |
| Animation | `/stage3/<id>` | 3a |

`/animation/<id>` is an older compact 3a view, still served.

### Gates

Each gate is **two halves**: a machine check re-evaluated on every read, plus
the reviewer's approval. Both must pass. Approving does not freeze a stage — a
later edit re-closes the gate.

Enforced in `gates.py` *and* at the API: `/api/run-stages` and
`/api/run-stage2` return **409** naming the blocking stage. A disabled button
is a suggestion; the 409 is the rule. `override: true` bypasses it, which is
what the "no critic" buttons pass.

Narration (2e) is **exempt** — no gate, no prerequisite, no approval. It only
fills the `narrative` field of an already-valid sequence. Gating it once made
the critic-free path unreachable.

### Human overrides

Three points where a human replaces model output. Each writes to its own
lineage-keyed directory and is swapped over the pipeline path at run time by
`_swapped`, which restores in `finally`:

| stage | saved to | stands in for |
|---|---|---|
| 1a | `code_human/` | `code()` |
| 2c | `sequence_raw_human/` | `sequence()` |
| 2d | `sequence_human/` | `sequence_final()` |
| 3a | `animation_human/` | `animation_final()` |

2c and 2d are deliberately different: 2c lets the critic refine your
correction, 2d takes it verbatim.

**Approve ≠ promote.** Approve records judgement and opens the gate; promote
copies your edit over the pipeline's file. Neither implies the other.

### Copy blocks

`/api/copy-block/<id>?agent=` calls the agent's own `build_request`, so what a
reviewer copies is exactly what the pipeline sends. **Never build a second
assembly path** — the screen and the run must not diverge.

`AgentContext` must be constructed *inside* `_sample_context`. Built outside,
it snapshots the wrong style-keyed lineage and copies a prompt for artifacts
the run would never read.

---

## Invariants worth not breaking

- **Style and target are per sample.** Anything resolving `sequence_*`,
  `animation_*` or `exports` must run inside `_sample_context`. `STATE["paths"]`
  is startup-time and wrong for a sample that switched target.
- **`_stage_state`, `_effective_1a`, `_effective_animation`** resolve from the
  live config, so their callers must already be inside the context.
- **Compile results are content-hash cached** (`gates.animation_renders`).
  Gate reads went from 5.0s to 0.05s. Failures are cached too.
- **The 3a gate checks the *export* compile**, not a single frame. A frame can
  render while `to_multipage_pdf_source` fails — that is how m3grounder passed
  its gate and then failed export.

---

## Tests

Two suites, 143 checks, both green. No model calls, but they need the tool
running on 8602 (the LIVE group) and must be run **from the repo root** — they
resolve the config by relative path.

```bash
python tests/test_annotate_suite.py       # 111 checks
python tests/test_annotate_workflow.py    #  32 checks
```

Both restore whatever they touch: annotation records are snapshotted and put
back, scratch files unlinked. A leftover `_t_*` or `.pre_human` after a run
means something aborted mid-test.

Several past failures were **the test being wrong, not the code**: a hardcoded
`</diagram>` when the tag is `</Diagram>`; a phrase grep against text that
wraps across lines in a YAML block scalar; an assertion placed after the
cleanup that removed what it was asserting on. Verify the product behaviour
directly before changing product code to satisfy a test.

---

## Working on this repo

- **Docker is no longer required.** It was the historical setup (`/code` inside
  `img-2-svg-pretraining-singlenode-venkat.kesav`); a local venv is now the
  supported path. Paths in older notes referencing `/code/...` map to the repo
  root.
- **Restart the tool after editing** `annotate/*.py` — Flask runs without the
  reloader, so a running instance keeps serving the old code. HTML and JS are
  read per request and only need a page refresh.
- **Re-run both suites after touching** `gates.py`, `cache.py`, `schema.py`,
  `store.py` or any critic prompt. Those five have the widest blast radius.
- **Prompts are YAML block scalars.** A rule can wrap across lines, so grepping
  for a one-line phrase gives false negatives; collapse whitespace first.

## Known open items

- **pipe00002 rasters**: detections use `raster_node_N` ids, the code uses
  `img_stage*` — a converter rename between runs. No path choice reconciles
  them; needs a forced re-detection from the raster screen.
- **m3grounder** fails its 3a gate with `Illegal parameter number in
  \pgffor@body` — the only sample combining `\foreach` with `#1`-parameterised
  styles, which `animateinline` re-expands. Diagnosed, deliberately unfixed,
  surfaced by the gate.
- **7 samples** have animations designed before the text-reference repair. Their
  `sequence_final` is corrected; the animations are not. Re-run 3a to pick them
  up — and they would get narration for the first time.
- `narrated_mp4` needs `google-genai >= 2.0`; older versions fail the
  Interactions API.
