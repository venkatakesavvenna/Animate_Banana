# Stage-1 diagram critic — what it actually fixed

Every defect below was found by the critic itself (Gemini 3.6 Flash, 2026-08-06) on the five
AnimateBench samples, by rendering the generated code and comparing it with the source figure.
Renders in this directory are `<sample>_before.png` / `<sample>_after.png`, produced from the
stored pre- and post-critic code — nothing here is hand-picked or hand-edited.

## Summary

| Sample | Before | After | What was wrong |
|---|---|---|---|
| pipe00002 | 0.52 | 0.52 | Cosmetic only; **repair rejected for not improving** |
| pipe00010 | **did not compile** | **0.88** | Undefined shape, then 3 containers painting over their contents |
| pipe00041 | 0.20 | **0.76** | 4 containers painting over their contents |
| pipe00045 | **did not compile** | **0.83** | Undefined node reference, then 2 containers overpainting |
| pipe00137 | **did not compile** | **0.72** | Undefined TikZ style |

**3 of 5 samples produced nothing at all before the critic existed.** They compiled to no PDF,
so there was no animation, no metrics, and no viewer entry for them.

## Class 1 — the document does not compile

All three failures are the same mistake: the generated code *uses* something it never
*declares*. The LaTeX log names the missing thing precisely, which is why this is repairable
without ever looking at the figure.

| Sample | LaTeX error | Rounds |
|---|---|---|
| pipe00010 | `! Package xcolor Error: Undefined color 'amber'` then `! Package pgf Error: No shape named 'text_identification' is known` | 2 |
| pipe00045 | `! Package pgf Error: No shape named 'lbl_c_sds' is known` | 1 |
| pipe00137 | `! Package pgfkeys Error: I do not know the key '/tikz/text_node'` | 1 |

`pipe00010` needed two rounds: fixing the undefined colour exposed a second error underneath
it. That is the loop earning its budget rather than wasting it.

Compilation is a **hard gate** — a document that produces no PDF cannot be scored, animated, or
exported, so every downstream stage was blocked on these three.

## Class 2 — it compiles, and renders almost nothing

This is the defect no compile check can catch, and it appeared **independently in three
samples**, so it is a systematic Stage-1a habit rather than a one-off.

TikZ paints in source order. The converter declares each `fit` container **after** the nodes it
contains and gives it an opaque `fill`, so every container paints over its own children.

```latex
\node[... img_3d_point_cloud ...] { \includegraphics{...} };   % drawn first
\node[fill=gray!5, fit=(img_3d_point_cloud) ...] (block_input) {};  % then painted over it
```

The document compiles with **zero warnings**. On pipe00041 all 11 raster crops are embedded in
the PDF — `pdfimages -list` shows them at 0 effective ppi — and the figure renders as four
empty boxes.

Critic findings, verbatim:

- **pipe00041** (4 critical): *"The INPUT container paints over all images inside it, rendering
  the block visually empty except for the title."* — plus the same for OUTPUT, Step 1 (with its
  Bidirectional sub-block), and Step 2.
- **pipe00010** (3 critical): *"The `fit` node `block_embedding` with `fill=red!4!white` is
  declared after all its contained nodes… covering them in source order."* — plus
  `block_unauthorized` and `block_matching`.
- **pipe00045** (1 critical): `block_a_b1` and `block_a_bn` using a `blue_container` style with
  `fill=cyan!10!white`, declared after their contents.

`pipe00041_before.png` vs `pipe00041_after.png` is the clearest illustration: the rendered PNG
grows from 51 KB to 457 KB because the after version actually contains the figure.

## Class 3 — genuine but cosmetic

- **pipe00002** (major): every stage image wrapped in an unwanted grey rounded box, from a
  `raster_node` style carrying `draw=gray!40, fill=gray!5`.
- **pipe00002** (minor): the feedback arrow's endpoint sits in empty space left of the first
  stage, because `top_loop_mid2`/`top_loop_end` are pinned at x=35.
- **pipe00045** (major): panel (d) draws labelled rectangles where the source has Gaussian
  kernel ellipses and star icons.

## The negative result matters too

**pipe00002 is the safety rule working.** Fidelity 0.52 opened the gate, the critic diagnosed
two real defects, produced a repair — and the repair scored no better, so it was **discarded**
and the original kept. `pipe00002_before.png` and `pipe00002_after.png` are byte-identical.

The stage reports that sample as `unresolved`, not as a success. A critic that always "improves"
its input is not measuring anything; this one can decline.

Four rules enforce that, each covered by a test in
`pipeline/tests/test_diagram_critic.py`:

1. Gate on fidelity before spending anything on repair (≥0.7 costs one scoring call).
2. Keep a repair only if it scores **higher** — best version wins, never the last tried.
3. Reject a repair that drops an `xml id` — later stages address elements by id, and silencing
   an error by deleting the element that caused it loses figure content.
4. Reject a repair that does not compile.

## Verified independently

The critic's own scores are confirmed by AnimateBench, which reads the artifact separately and
had no part in producing it:

| | critic's score | AnimateBench `rendering_fidelity` |
|---|---|---|
| pipe00041 before | 0.20 | 0.300 |
| pipe00041 after | 0.76 | 0.760 |

## Downstream effect (progressive_reveal, all 5 samples)

With the critic in place, every sample reaches stage 3 — previously three of five stopped dead
at stage 1.

| metric | 00002 | 00010 | 00041 | 00045 | 00137 |
|---|---|---|---|---|---|
| diagram CSR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rendering fidelity | 0.580 | 0.850 | 0.760 | 0.820 | 0.770 |
| PAA | 1.000 | 0.886 | 0.842 | 1.000 | 1.000 |
| edge F1 | 1.000 | 0.833 | 0.938 | 0.757 | 0.826 |
| coverage recall | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DOVR | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SSCR | pass | pass | pass | pass | pass |
| animation CSR | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

Two weaknesses the critic does **not** address, named here so they are not mistaken for its
successes:

- **TOF 0.667–0.798 on four of five samples.** The sequencer is not front-loading top-level
  elements the way `overview_first` requires. A Stage-2 planning weakness.
- **Animation CSR 0 on pipe00045 and pipe00137**, both from an unclosed `\foreach` emitted by
  the Stage-3 designer (`! File ended while scanning use of \pgffor@collectargument`). Their
  AIF of 2.8 and 2.0 says the designer rewrote more diagram code than it added, which is
  probably the same underlying problem. This is the *animation* critic's job, and it is
  disabled in `bench_gemini.yaml`.

Regenerate these renders with:

```bash
python -m img_2_svg_pretraining.pipeline.run_pipeline critique-diagram \
  --config configs/bench_gemini.yaml --force
```

Interactive before/after with a swipe slider: `inspector/critic_ab.py` on port 8602.
