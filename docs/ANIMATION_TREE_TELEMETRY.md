# Animation evaluation tree — telemetry and scaling

Measured on the v3 bench (32 samples, 48 GT-covered cells), judge
`Qwen/Qwen3-VL-235B-A22B-Instruct` served by vLLM on one 8×H100 node,
2026-08-19/20. Every number below comes from the run logs and the eval
records, not from a model of what the run *should* cost.

Reproduce with the snippets at the end.

---

## Headline

| | |
|---|---:|
| Cells scored | **45** of 48 (3 have no frames — the animation never compiled) |
| Judge calls | **3,519** |
| Images sent | **~9,200** |
| Wall time, scoring only | **3.8 h** |
| Server startup (cold) | **8.5 min** |
| Mean per cell | **78 calls**, ~4.5 min |
| Median per cell | **63 calls**, ~4.2 min |
| Per call | **3.4 s** |
| Sustained rate | **~1,050 calls/h** |

Cost is GPU-time only. The judge is local, so unlike the Gemini half of the
pipeline this has no per-token bill and no daily quota — the constraint is
wall clock and node occupancy.

---

## Where the time goes

Calls per cell, by tree node:

| node | total calls | mean/cell | share | images/call | sequential? |
|---|---:|---:|---:|---:|---|
| **ASCS** (E2, style compliance) | 797 | 17.7 | 22.9% | 3 (2 on frame 1) | no |
| **SSS** (SC1, selection) | 764 | 17.0 | 22.0% | 3 (2 on step 1) | no |
| **GPS** (SC2, pacing) | 764 | 17.0 | 22.0% | 3 (2 on step 1) | no |
| **Omission** (E3) | 662 | 14.7 | 19.1% | 2 | **yes** |
| **Repetition** (SC3) | 392 | 8.7 | 11.3% | 2 | **yes** |
| **VFS** (E1a, fidelity) | 95 | 2.1 | 2.7% | 2 | no |
| checklist (shared by E3+SC3) | 45 | 1.0 | 1.3% | 1 | n/a |

VFS is nearly free because four of five styles judge only the **last** frame —
the completed diagram. Alpha masking is the exception and judges every frame,
which is the whole of its 2.7%.

Per-style wall time (median), from the timed run:

| style | median/cell | why |
|---|---:|---|
| hopping_bounding_box | 370 s | most frames per animation |
| sliding_bounding_box | 362 s | many frames; omission/repetition sample every 4th |
| alpha_masking | 291 s | VFS judges *every* frame here, not just the last |
| colour_pop | 228 s | |
| progressive_reveal | 205 s | fewest frames |

---

## What actually drives cost

Cost is linear in **frame count**, and nothing else comes close. Fitted over
all 45 cells:

```
calls_per_cell  ≈  4.35 × frames  +  1
```

(frames/cell: median 15, mean 17.7, max 44.)

The 4.35 is the six nodes each walking the same deck: ASCS every frame,
omission every frame, repetition every frame on three styles, SSS and GPS once
per *timestep* (≈1 frame in this bench), VFS once. Halve the exported frames
and you halve the bill.

Diagram complexity — element count, checklist size — costs **nothing extra**.
It makes each prompt slightly longer, but the call count is unchanged. A
48-element checklist and a 9-element one cost the same number of calls.

---

## The finding that matters for scaling

**The GPU was serving one request at a time, for the entire run.**

```
Engine 000: Avg prompt throughput: 340.1 tokens/s,
            Avg generation throughput: 99.4 tokens/s,
            Running: 1 reqs, Waiting: 0 reqs,
            GPU KV cache usage: 1.7%,
            Prefix cache hit rate: 32.2%, MM cache hit rate: 85.4%
```

`Running: 1 reqs` is the **maximum ever observed** in the whole serve log, and
KV cache usage peaked at 1.7%. The server was configured for `--max-num-seqs
16`. So roughly **94% of the serving capacity sat idle** for 3.8 hours.

The cause is the driver, not the model: `scripts/animation_sweep.py` runs one
`run_eval` subprocess per cell, sequentially, and each node inside a cell walks
its frames in a Python `for` loop. The config's `max_concurrency: 8` is never
exercised because nothing ever issues two calls at once.

`MM cache hit rate: 85.4%` is worth noting separately — the source figure is
re-sent on every one of a cell's ~78 calls, and vLLM is deduplicating it. That
is why adding a third image to ASCS did not cost what it looks like it should.

---

## How this scales

### Across cells — embarrassingly parallel

Cells share nothing. Different sample, different style, separate record file,
separate `run_eval` process. Running C cells at once is a straight C× speedup
until the server saturates.

With `--max-num-seqs 16` and observed usage of 1, a single node should absorb
**8–16 concurrent cells** before queueing. That predicts the current 3.8 h
becoming roughly **25–30 minutes on the same hardware**, with no model change
and no new node.

The change is one line of driver structure — a process pool over cells instead
of a `for` loop. It is not implemented; the sweep that produced these numbers
ran sequentially.

### Across nodes — near-linear, until you run out of cells

Each node needs its own full copy of the model (471 GB bf16 across 8 GPUs,
TP=8). There is no cross-node communication during scoring, so throughput is
additive:

| nodes | concurrent cells | est. wall for 48 cells | caveat |
|---:|---:|---:|---|
| 1 (as run) | 1 | 3.8 h | measured |
| 1 (pooled) | 8–16 | ~25–30 min | predicted |
| 2 | 16–32 | ~15 min | 48 cells starts limiting |
| 4 | 32–64 | ~8 min | mostly startup |
| 8 | 64+ | ~8.5 min | **startup-bound** |

Past ~4 nodes the 8.5-minute cold start dominates and adding hardware buys
nothing: you cannot finish a 48-cell sweep faster than one model load. Extra
nodes only pay off for a materially bigger bench, or if the servers are kept
warm between sweeps.

### Within a single cell — capped at ~3.3×

Four of the six nodes issue **independent** calls and could run concurrently
inside one cell:

- **VFS, ASCS** — each frame judged on its own; no state carried forward.
- **SSS, GPS** — `prior_summary_at(index)` is precomputed from the *sequence*,
  not from any previous verdict, so timesteps do not depend on each other.

Two are genuinely sequential and cannot be parallelized without changing what
they measure:

- **Omission** threads the outstanding checklist; frame N's prompt contains
  what frames 1..N−1 popped.
- **Repetition** threads the running frequency table, likewise.

That is 69.6% parallelizable, 30.4% serial. By Amdahl, intra-cell speedup caps
at **3.3×**, and reaches 2.6× at 8-way. Parallelising *across cells* is
strictly better and much simpler — do that first, and only consider this if
cells ever become scarce relative to hardware.

---

## Startup cost

| phase | time |
|---|---:|
| Loading weights (471 GB, local NVMe) | 230.5 s |
| Engine init (profile, KV cache, warmup) | 45.1 s |
| API server up → `/health` 200 | ~35 s |
| **total, cold** | **~8.5 min** |

Weights come from `/opt/dlami/nvme` (node-local). That directory does not
survive a node move — it has had to be re-downloaded (440 GB, ~7 min at
1.2 GB/s) after each reallocation. Budget ~15 min from a cold node, ~8.5 from
a warm one.

---

## A confound, stated rather than buried

Comparing the two sweeps over the 43 cells timed in both:

| | median/cell | total |
|---|---:|---:|
| run 1 — old prompts, ASCS sends 2 images | 328 s | 4.47 h |
| run 2 — new prompts, ASCS sends 3 images | 252 s | 3.27 h |

Run 2 sent **more** images and finished **27% faster**. That is not a speedup
from the prompt change, and it should not be read as one. The likely cause is
cache warmth: run 2 started against a server whose multimodal cache was
already at 85% hit rate, and its prefix cache had seen these prompts. Treat
the two totals as not comparable; the per-call figure of 3.4 s from run 2 is
the one to plan with.

---

## Reproducing these numbers

Per-cell wall times (parses `running:` → `exit=` pairs):

```bash
grep -E "running:|exit=" logs/animation_sweep.log
```

Per-node call counts, straight off the records:

```python
import json, glob
E = "data/animatebench_v3_cache/animatebench_v3/evals/bench_v3_or"
for f in glob.glob(f"{E}/*/*/animation.json"):
    d = json.load(open(f))
    print(f.split("/")[-2],
          d.get("vfs_frames_judged"), d.get("ascs_frames_judged"),
          len(d.get("omission_frame_detail") or []),
          len(d.get("repetition_frame_detail") or []),
          len(d.get("sss_step_detail") or []),
          len(d.get("gps_step_detail") or []))
```

Server-side throughput and concurrency:

```bash
docker exec img2svg-qwen-venkat \
  grep -E "Engine 000:|Loading weights took|init engine" /tmp/qwen_serve.log
```

---

## Summary

- **3.8 h, 3,519 calls, 3.4 s/call** for 45 cells on one 8×H100 node.
- Cost is **linear in exported frames** (`≈4.35 × frames + 1`), and
  independent of diagram complexity.
- **ASCS + SSS + GPS are two thirds of the bill**; VFS is ~3%.
- The run used **~6% of available serving capacity**. Pooling across cells
  should cut 3.8 h to well under an hour on the *same* node — that is the
  first thing to do, well before adding hardware.
- Beyond ~4 nodes, the 8.5-minute model load dominates and more hardware stops
  helping.
