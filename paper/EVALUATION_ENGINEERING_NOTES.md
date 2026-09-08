# Evaluation engineering notes

Written 2026-09-08, from running the AnimateBench evaluation infrastructure end
to end: the eight-cell ablation study, the 193- and 216-sample sweeps, the
DeepSeek-V4 and Kimi-K2.6 generation runs, and the Original\_Presentations
corpus.

These are the things that were **measured**, not assumed, and that a reader
reproducing this work would otherwise have to rediscover. Almost all of them
share one shape: *a failure that produces a plausible number instead of an
error*. That is the recurring hazard of LLM-judged pipelines, and it is the
reason this file exists.

---

## 1. The dominant failure mode is a silent pass, not a crash

Every incident below produced output that looked like success.

| what happened | what it looked like | how it was caught |
|---|---|---|
| Judge backend absent from the config | `--judge-backend kimi_judge` **fell back to Gemini** and said so in one line of stdout | comparing the `judge:` line against the intended judge |
| Cells already having a record | driver logged `ok` for 35 cells in 20 s | `animation: cached` also matches a bare `animation:` grep |
| Server died mid-run | records written with `Connection error` in every `*_errors` | scores present but `None`, `stages_skipped` empty |
| Video metrics with no `imageio`/`cv2` | run "succeeded", scoring the *frame* metrics instead | `stages_skipped: ModuleNotFoundError` |
| Band judge fed a 3-frame deck | confident `BAND A` | `frame_count: 3` in the record |

**Rule.** Every driver must assert on *evidence of work*, never on exit status
or the absence of an error: the judge's own identity string, a real score line,
a token counter that moved. `vllm:generation_tokens_total` staying flat while a
driver logs `ok` for every cell is the signature of "already recorded", and it
is indistinguishable from success in the log.

**Corollary — an error record is worse than no record.** `run_eval` skips any
cell that *has* a record, so a stub written while the judge was unreachable
masks that cell from every future resume. Worse, GPS has a computed component
that still emits a score when every judge call failed, so such a cell can read
`gps = 1.0` built from zero judge input. Purge error stubs before resuming, and
gate each cell on a healthy judge before it runs.

---

## 2. What the per-timestep metrics actually require

SSS, GPS and NAS are gated on a **frame-to-timestep mapping**, built only when
both a sequence and a structure XML exist (`run_eval._run_animation`), and
`frames.step_frames` accepts exactly two counts: `n_steps`, or `n_steps + 1`
(frame 0 being the pre-animation state). Anything else is unmappable and the
cell is skipped rather than guessed — a frame attributed to the wrong step
yields a band that is wrong while looking entirely plausible.

Two consequences worth stating in any paper using these metrics:

- **A bare animated SVG cannot be scored on SSS/GPS/NAS.** Zero-shot output has
  neither a sequence nor an XML. It is comparable on the video metrics only, and
  its SSS/GPS must be rendered as *n/a*, visually distinct from blank.
- **But the gate is narrower than it first appears.** `selection_sensibility_bands`
  and `granularity_pacing_bands` take `(step_frames, source_image, style,
  xml_text)` — they never read the sequence's *content*. The sequence supplies
  the timestep count and, for NAS, the captions. So a corpus with genuine
  per-state frames can be scored honestly even when no pipeline produced it:
  the judge sees real frames against the real figure, and nothing about the
  animation is inferred from the artefact being judged.

That distinction is what made the Original\_Presentations corpus scoreable
(§5), and it is easy to get backwards. The circularity objection — "a sequence
reverse-engineered from a finished video would score perfectly by construction"
— applies to metrics that read the sequence, not to these.

---

## 3. Judged media is sampled, and the sampling rate is not yours

- Gemini samples inline video at **exactly 1 fps** (token cost is linear in
  duration: 60 tokens/frame at 2/5/10/20/40 frames). The exporter writes at
  `fps: 2`, so sending `animation.mp4` unchanged shows the judge roughly *half*
  the animation, chosen by the sampler, with nothing in the response to say so.
  Re-time the deck instead.
- Video frames are tokenised far more coarsely than stills: **60 tokens/frame
  vs ~700** for the same diagram as an image. A criterion asking about text
  legibility is unanswerable at 60 tokens/frame. `MEDIA_RESOLUTION_HIGH`
  (~250 tokens/frame) makes a frames-vs-video disagreement about modality
  rather than resolution.
- **A fixed pixel threshold is a threshold on canvas size.** A hopping box on a
  4236×4236 figure peaks at 0.0027 activity against a 0.01 threshold — the
  animation reads as perfectly static end to end. Calibrate per input.

---

## 4. Cost and serving, measured

**Reasoning tokens are billed as completion.** DeepSeek-V4-Flash-Vision on one
figure-description call: 137 reasoning vs 40 answer tokens — **5.2× the cost**
for the same content. Disabling reasoning is a per-backend decision, and it must
be nested where the client actually forwards it (`default_params.extra_body`);
a top-level key is silently dropped and you pay the 5.2× with nothing in the
response to show the setting never arrived.

**Reasoning off is for judges, not generators.** Kimi K2.6 with
`chat_template_kwargs.thinking: false`: 11.1 s / 710 completion tokens → 0.2 s /
6 tokens, and the answer is *cleaner* (bare JSON instead of prose a parser must
dig through). Across ~22 k judged calls that is a night instead of a week. The
same setting on the generator removes what makes the model good at writing SVG.

**Generation budget, per cell, measured over 91 cells:** 13.1 calls with critics
on (120 k prompt / 83 k completion tokens), ~6 calls with them off. That is the
number to price a sweep with; call counts and token counts do not scale together.

**Serving.** Kimi K2.6 is 555 GiB and does not fit on one 8×H100 node — at
`--gpu-memory-utilization 0.97` the weights load and vLLM then dies with
"No available memory for the cache blocks". Two nodes (TP=8, PP=2) load it in
~6–7 minutes from node-local NVMe. Being spread across two shared nodes is also
its main fragility: twice, a process exiting on the worker took the whole engine
down, leaving both containers reporting "Up" with GPU memory at 0 — healthy to
`docker ps`, serving nothing. Supervise on the *endpoint*, not the container.

**Model support is not implied by architecture support.** vLLM lists
`DeepseekV4ForCausalLM` as supported, but registers it as **text-generation
only**; the vision checkpoint fails on its projector (`no module named
'aligner'`) after a 168 GB download and a full TP=8 init. Check the multimodal
registry, not the arch list.

---

## 5. Ground-truth presentations as a reference row

The 35 Original\_Presentations are the authors' real conference talks — human
artefacts, not pipeline output. They are worth scoring because they anchor the
scale: they show what the metrics say about a *human* animation of the same
figures.

Practicalities that shaped what could be claimed:

- All 35 carry state-indexed frame extractions (`state_<n>_frame_<m>.png`,
  median 4 states), which give a genuine per-timestep deck on the `n+1` path.
- Only **11 of 35** ship a timed `transcript.json`; the rest have untimed text.
  NAS is therefore reported over 11 and the other 24 recorded as *unnarrated* —
  never as 0. A missing value that renders as 0 sorts to the top of a
  "best" list, which is the wrong answer stated confidently.
- Only 10 of 35 exist in any bench split, and 25 exist nowhere else in the
  corpus, so the structure XML was reused from the Gemini-3.7-Flash pipeline
  run (it describes the *figure*, which is the same figure either way).

Measured (Kimi-K2.6 judge): **SSS 0.831, GPS 0.818** over 35 cells;
**NAS 0.626** over the 11 narrated ones. On the video metrics (Gemini judge):
`vfs_band` A 24 / B 6 / C 4 / D 1, and `ascs_video` ACCEPT 14 / **DISCARD 21** —
the DISCARDs concentrated in progressive\_reveal (16 of 22), where a human talk
rarely reveals strictly cumulatively. Judges' rationales were specific and
checkable, e.g. *"At 00:27, previously revealed components disappear instead of
persisting cumulatively."*

**Do not pool these with pipeline scores without saying which judge produced
them.** The video metrics here came from a Gemini judge and the per-timestep
ones from Kimi; the two are not interchangeable.

---

## 6. Ablations: the honest reading

All eight configurations, 91 cells each, judged by Kimi K2.6 (728/728 complete):

| configuration | n | SSS | GPS | NAS |
|---|---|---|---|---|
| AnimateBanana (full) | 91 | 0.930 | 0.965 | 0.955 |
| − Stage-1 critic | 91 | 0.935 | 0.967 | 0.955 |
| − diagram image (stage 2) | 90 | 0.930 | 0.966 | 0.937 |
| − XML (sequencer) | 88 | 0.927 | 0.968 | 0.943 |
| − narration context | 91 | 0.937 | 0.964 | 0.945 |
| − Stage-2 critic | 89 | 0.926 | 0.964 | 0.951 |
| − designer image | 91 | 0.927 | 0.967 | 0.957 |
| − Stage-3 critic | 91 | 0.931 | 0.968 | 0.955 |

**The spread is ~0.011 on SSS, ~0.004 on GPS and ~0.020 on NAS — smaller than
the difference between adjacent bands.** Read plainly, these ablations do not
separate. Several removals score *above* the full pipeline. The defensible
conclusions are (a) NAS is the only metric showing a consistent ordering, with
removing the diagram image from stage 2 costing the most (0.937 vs 0.955), and
(b) on this corpus and with this judge, the remaining components are not
individually load-bearing at the resolution these metrics can measure.

That is a real finding and should be reported as one, rather than as a table
implying differences it cannot support. It also argues for reporting per-cell
distributions and paired tests rather than means alone — a per-cell paired
comparison against the full pipeline would say far more than eight means
clustered within 0.01.

By contrast, the cross-model comparison on the 193-sample zero-shot split does
separate cleanly (AnimateBanana 0.945 / 0.951 / 0.981 vs the next model at
0.844 / 0.922 / 0.854) — but note it rests on 135 judged cells against 179–189
for the open-weights models, so a like-for-like claim needs the shared subset.

---

## 7. Reproducibility checklist

1. Record the **judge identity** in every artefact, and assert on it.
2. Treat "unmeasured" and "zero" as different values, all the way to the plot.
3. Compare artefact **mtimes**, never mere existence, when two describe the
   same thing; evict everything derived from a rebuilt artefact.
4. Cache keys must cover every input that can change the answer — including
   media bytes and sampling params — and must **raise** on an unknown part.
5. Verify a staged checkpoint **shard-by-shard against its weight index**. Size
   comparisons hide truncation, and a size gate in the wrong unit (GB vs GiB)
   silently never fires: 555 GiB measured against a 590 "GB" constant waited
   seven hours on a download that had already finished.
6. State the *n* behind every mean. Coverage differed by up to 58 cells between
   models in this study.

---

## 8. Environment notes

Shared-cluster realities that cost real time here: filesystem **inode**
exhaustion (201.4 M of 201.8 M used) surfaces as `ENOSPC` while `df -h` still
shows free space, and it is driven by file *count* — frame decks and response
caches — not by bytes. Node occupancy is not stable across a run, and free GPUs
are not available GPUs; node selection belongs to the operator, not to a
discovery script.
