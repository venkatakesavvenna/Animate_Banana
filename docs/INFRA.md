# Infrastructure — how the animation eval actually runs

Written after running a full 29-cell sweep on Qwen3-VL-235B. Everything here is
either a command that was run or a constraint that broke a run first. The
constraints are the point: none of them are visible from the code, and every
one of them cost time.

---

## `ingest/` is deliberately not used here

The approved plan called for serving Qwen through vision-ingest's
`launch_vllm_serve`, and for driving the sweep through its `WorkerPool` + DB
state machine. **Neither happened.** Nothing below imports `vision_ingest`;
vLLM is launched directly and the fan-out is five shell processes.

That was drift, not a decision — the direct launch was the shortest path to a
first live cell and never got revisited. Recorded here so the next person does
not read this file as evidence the plan was followed.

Two things to know if you pick it back up:

- `vision_ingest` *is* importable in `ocr_env_vllm`, but it resolves to
  `/fsxvision_new/aryanjain.intern/Patram-Ingest/`, a different checkout from
  this repo's `ingest/`, and **that copy has no `launch_vllm_serve`**. Use
  `pip install -e ingest/` (v1.9.4) rather than whatever is on the path.
- What the omission actually costs is not the launcher — that is a thin
  wrapper around the same `vllm serve`. It is the DB state machine, recovery
  and writer: when containers were killed mid-sweep, in-flight work was lost
  outright, progress was invisible, and a dead worker's requests vanished into
  `Connection error` entries instead of being reissued. `WorkerPool` is what
  the five-process fan-out and the `pkill` foot-gun below are reimplementing,
  badly.

Worth retrofitting before the next full sweep, which is exactly when crash-safe
resumability pays.

---

## The shape of it

```
  host (bare metal, 8x H100 80GB)
   |
   |-- container: img2svg-qwen-venkat            <- the judge
   |     --network host  --gpus all
   |     /environments  <- OUR venv copy
   |     tmux qwen_serve: vllm serve Qwen3-VL-235B  :8010
   |
   |-- container: img-2-svg-pretraining-...      <- the viewer
   |     /code = repo, runs as root
   |     python -m ...inspector.compare          :8601
   |
   `-- host venv: /fsxvision_new/venkat.kesav/environments/img_2_svg_pretraining
         python -m ...animatebench.run_eval      <- the eval driver
```

Three processes, three homes, and the reason for the split is entirely
environmental:

- **The judge** needs vLLM and 8 GPUs, so it needs a container with a Python
  the venv was built against.
- **The eval driver** needs only the pipeline deps and talks HTTP, so it runs
  on the host where our own venv works and files land owned by us.
- **The viewer** predates all this and runs as root inside the older container,
  which is also the only way it can write anywhere under `pipeline/cache`.

They meet over `127.0.0.1` because **every container here uses
`--network host`**. That is what makes `base_url: http://127.0.0.1:8010/v1` in
the config work from the host with no port mapping — and it is also why a port
collision on 8000 is possible at all (see below).

---

## 1. The judge

```bash
docker run -d --name img2svg-qwen-venkat \
  --gpus all --network host --shm-size 64g --ipc host \
  -v /fsxvision_new:/fsxvision_new \
  -v /fsxvision_new/venkat.kesav/environments:/environments \
  -v /opt/dlami/nvme:/opt/dlami/nvme \
  -w /fsxvision_new/venkat.kesav/img_2_svg_pretraining \
  vlm-ingest-pipeline-aryanjain.intern:latest sleep infinity
```

```bash
docker exec img2svg-qwen-venkat bash -lc \
 'D=/opt/dlami/nvme/venkat.kesav/hf_cache
  tmux new-session -d -s qwen_serve "HF_HOME=$D HF_HUB_CACHE=$D/hub \
    HUGGINGFACE_HUB_CACHE=$D/hub VLLM_CACHE_ROOT=/tmp/vllm_cache_venkat \
    /environments/ocr_env_vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --limit-mm-per-prompt.image 3 --max-num-seqs 16 \
    --port 8010 --host 127.0.0.1 > /tmp/qwen_serve.log 2>&1"'
```

Roughly **10 minutes** to load 471 GB of bf16 across 8 cards. Health check:
`curl -sf http://127.0.0.1:8010/health`.

### Why each flag

| flag | reason |
|---|---|
| `--tensor-parallel-size 8` | 471 GB of weights does not fit on fewer |
| `--gpu-memory-utilization 0.90` | 0.92 works on an idle node; 0.90 leaves room for another process's CUDA context. A run died at 0.92 because someone else held 57 GB/card |
| `--max-model-len 32768` | prompts are ~1–4k tokens plus up to 3 images; 128k would spend KV cache we need for weights |
| `--limit-mm-per-prompt.image 3` | SSS/GPS send figure + previous frame + current frame |
| `--port 8010` | **not 8000** — with host networking, something else on the node already listens there |
| `--network host` | lets the host-side eval driver reach the server at `127.0.0.1` |
| `-m vllm.entrypoints.openai.api_server` | not the `vllm` script, whose shebang is a hard-coded `/environments/...` path |

### Weights

```bash
docker exec img2svg-qwen-venkat bash -lc \
 'D=/opt/dlami/nvme/venkat.kesav/hf_cache
  tmux new-session -d -s qwen_dl "HF_HOME=$D HF_HUB_CACHE=$D/hub \
    HUGGINGFACE_HUB_CACHE=$D/hub \
    /environments/ocr_env_vllm/bin/hf download \
    Qwen/Qwen3-VL-235B-A22B-Instruct --max-workers 16 > /tmp/qwen_dl.log 2>&1"'
```

**440 GB, 96 shards, ~7 minutes at 1.2 GB/s.** Ungated, no token needed.

- The container **presets `HF_HUB_CACHE=/hf_cache/hub`**, which overrides
  `HF_HOME`. Setting only `HF_HOME` silently downloads into whatever that image
  points at — in one case a colleague's cache directory. Set all three.
- `HF_HUB_ENABLE_HF_TRANSFER=1` fails unless `hf_transfer` is installed. Plain
  download already saturates the link.
- **`/opt/dlami/nvme` is node-local.** Moving nodes loses the weights. That is
  a deliberate trade: ~7 minutes to re-pull against a much slower load from
  Lustre every time the server restarts.

---

## 2. The eval driver

Runs on the **host**, not in a container:

```bash
PYTHONPATH=$PWD/src \
/fsxvision_new/venkat.kesav/environments/img_2_svg_pretraining/bin/python \
  -m img_2_svg_pretraining.animatebench.run_eval animation \
  --config src/img_2_svg_pretraining/pipeline/configs/bench_v2_gemini.yaml \
  --dataset $PWD/data/animatebench_v2 \
  --evals-root /fsxvision_new/venkat.kesav/animatebench_evals_qwen \
  --style progressive_reveal --judge-backend qwen_judge
```

Three flags exist purely because of the environment:

- **`--dataset`** — the bench configs name `/code/data/...`, a path that exists
  in no container currently running.
- **`--evals-root`** — **the entire `pipeline/cache` tree is owned by root**,
  written by a root container, and there is no passwordless sudo anywhere to
  chown it back. Records, the checklist cache, prepared frames and the
  backend's own response cache all follow this flag.
- **`--judge-backend qwen_judge`** — selects the `openai_compat` block in
  `bench_v2_gemini.yaml`. Judging Gemini-generated animations with Gemini is
  the circular-evaluation setup the project's own notes warn about; a different
  model family is the point.

### Parallelism

Cells are independent, so **one process per style** is the unit:

```bash
for S in progressive_reveal colour_pop alpha_masking \
         hopping_bounding_box sliding_bounding_box; do
  PYTHONPATH=$PWD/src setsid nohup python -u -m ...run_eval animation \
    ... --style $S > /tmp/animsweep/qwen_$S.log 2>&1 < /dev/null &
done
```

**Measured: 35 calls/min, 2710 calls, ~1.5 h for the full sweep.** Within a
cell the folds must stay sequential — each call carries the previous call's
state — but nothing couples one cell to another.

- Use **`python -u`**. Without it stdout is block-buffered when not a tty and
  the progress logs stay empty for the whole run; the records are the only
  signal. This was the case in the sweep that produced the current data.
- Do **not** `pkill -f "run_eval"` from a shell whose own command line contains
  that string — it kills the shell. Collect PIDs with `ps -eo pid=,cmd= | grep
  "[r]un_eval"` first, then `kill` them literally.

---

## 3. The viewer

Runs as root inside the older container, which is the only process that can
write under `pipeline/cache`:

```bash
docker exec -d img-2-svg-pretraining-singlenode-venkat.kesav bash -lc \
 'source /environments/img_2_svg_pretraining/bin/activate && \
  ANIMATEBENCH_EVALS_ROOT=/fsxvision_new/venkat.kesav/animatebench_evals_qwen \
  PYTHONPATH=$PWD/src python -m img_2_svg_pretraining.pipeline.inspector.compare \
    --configs "Gemini v2=bench_v2_gemini.yaml" \
    --dataset /fsxvision_new/venkat.kesav/img_2_svg_pretraining/data/animatebench_v2 \
    --port 8601 > /tmp/compare8601.log 2>&1'
```

`ANIMATEBENCH_EVALS_ROOT` **must match whatever `--evals-root` the run used**,
or the Metrics tab shows no scores for a sample that has them.

- `compare.py` and `STATE` are frozen at boot — **editing it requires a
  restart**. `compare.html` is read per request and only needs a refresh.
- Because the override points at a root holding only animation records, the
  other four suites do not appear. Chowning `pipeline/cache` back to a normal
  user would let everything live in one place again.

---

## Judge backends

```yaml
backends:
  gemini_flash:                       # 64-key pool, api_keys.csv
    type: gemini
    model: gemini-3.6-flash
  qwen_judge:                         # the served endpoint
    type: openai_compat
    base_url: http://127.0.0.1:8010/v1
    model: Qwen/Qwen3-VL-235B-A22B-Instruct
    api_key: EMPTY
    max_concurrency: 8
```

`openai_compat` already sends images as data URLs, so a served vision model
needs **no new backend code** — only that `make_judge` stopped silently
rewriting any non-Gemini backend to Gemini.

### Gemini's practical limits

- The key pool is shared and **exhausts**. Five concurrent processes throttled
  each other into 27 minutes of silence; the backend was walking all 64 keys
  and backing off correctly. Two processes is the practical ceiling, and even
  that ran dry.
- Diagnose through the **backend**, never a hand-rolled probe: `genai.Client`
  is closed on garbage collection, so a loop that builds one client per key
  reports `client has been closed` for every key and looks exactly like a dead
  pool. That misdiagnosis cost an hour.

```python
b = make_backend("gemini_flash", cfg.backend_cfg("gemini_flash"), cache_root=None)
r = b.generate([Message.user("Reply with the single word OK.")], max_tokens=16)
print(r.ok, r.error)     # walks the whole ring in ~15s and reports honestly
```

---

## Costs, measured

| | |
|---|---|
| Full sweep | 2,717 calls / 29 cells (~94 per cell) |
| Qwen, 5-way | 35 calls/min → **~1.5 h** |
| Model load | ~10 min |
| Weights download | ~7 min (440 GB @ 1.2 GB/s) |
| Call mix | ASCS 26%, omission 20%, SSS 19%, GPS 19%, repetition 11%, VFS 3% |

Responses are content-hash cached, so re-runs only re-pay for stages whose
prompts changed. `--stages` merges into the existing record rather than
replacing it, so a record can be built up one node at a time.

---

## Things that will bite the next person

1. **`/opt/dlami/nvme` is node-local.** New node, no weights.
2. **The venv is tied to a Python version.** `ocr_env_vllm/bin/python3` is a
   symlink to `/usr/bin/python3`; it was 3.12 on one node and 3.10 on another,
   which makes the venv silently unusable — its site-packages simply vanish
   from `sys.path`. Check `python3 -V` inside the container before debugging
   anything else.
3. **`--network host` means the whole node's ports.** 8000 was already taken.
4. **The cache tree is root-owned**, so an ordinary user cannot write eval
   records, prepared frames, or the response cache without `--evals-root`.
5. **Containers are not yours.** Two runs died mid-sweep because a container
   was stopped by someone else. Exit 137 with `OOMKilled=false` means an
   external kill, not a fault. Per-frame error isolation means a cell survives
   with partial evidence rather than dying — check `*_errors` in the record.
6. **`docker stop` / `rm` may be denied** even where `run` and `exec` are
   allowed; leftover containers then need removing by hand.
