# Running the annotation tool with 3-4 people

`pipeline/annotate/app.py` is the Stage 1 (D2C + raster) human-review tool. Each
person runs their own instance, locally, against a disjoint slice of the
sample set. This doc covers how to slice the work and how results come back
together.

## Sharding

Everyone runs the **same config** (same `code_converter`/`raster_integrator`
backend and model) so cache lineage strings line up across people. Split the
sample set with `--shard i/n` (1-indexed, deterministic — `sorted(ids)[i-1::n]`,
no coordination file needed):

```
# person 1 of 4
python -m img_2_svg_pretraining.pipeline.annotate.app \
  --config pipeline/configs/default.yaml --shard 1/4 --port 7862

# person 2 of 4
python -m img_2_svg_pretraining.pipeline.annotate.app \
  --config pipeline/configs/default.yaml --shard 2/4 --port 7862
```

Or hand out an explicit list with `--only ID ID ...` if you want to assign
specific samples rather than a hash split.

## Case 1: everyone has the shared FSx mount

If your docker container's dataset/cache mounts resolve to the same absolute
path as everyone else's (this project's default — `data/test_benchmark` lives
on `/fsxvision_new/...`), **there is nothing to merge**. Every write — pipeline
artifacts the tool triggers on demand, plus the tool's own `annotations/`,
`code_human/`, `code_final_human/` trees — lands in the same cache tree,
keyed by sample id. Two people never touch the same file because sharding
keeps sample ids disjoint, and lineage strings are identical because
everyone runs the same config. Just make sure your shard assignment doesn't
overlap anyone else's.

## Case 2: someone is disconnected (e.g. a laptop, no cluster mount)

Work against a local copy of your shard's samples and a local `cache_root`,
then copy your results back into the shared tree once you're back on the
network:

```bash
rsync -av \
  local_cache/test_benchmark/{code,code_final,rasters,annotations,code_human,code_final_human,xml,xml_human,sequence,sequence_final,sequence_human,exports}/ \
  /fsxvision_new/venkat.kesav/img_2_svg_pretraining/src/img_2_svg_pretraining/pipeline/cache/test_benchmark/
```

The Stage-2 trees (`xml_human/`, `sequence_human/`) follow the same rule as the
Stage-1 ones. Note `sequence_human/` is keyed by a lineage that includes the
animation style, so a sample annotated under two styles has two directories —
that is intentional, and it is what makes switching style back recover the
earlier work rather than lose it.

This is a plain copy, not a conflict-resolving merge, and that's safe for
two reasons:

- **Disjoint sample ids** (sharding) means no file is ever written by two
  people.
- **Identical lineage strings** (same config for everyone) means there's no
  risk of two different models' outputs landing under a path that looks the
  same but isn't.

If sharding is ever violated — two people annotate the same sample — the two
`annotations/<id>.json` files need manual reconciliation. There's no tooling
for that; don't let it happen by keeping shard assignments non-overlapping.

## Reminder: promotion is explicit

Nothing you do in the tool touches the pipeline's own `code_final` until you
click "Promote" on a sample. Comments, pasted human code, and adjusted raster
boxes all land in the tool-owned trees (`annotations/`, `code_human/`,
`code_final_human/`) first — safe to merge freely, and safe from being
clobbered by someone else's pipeline re-run.
