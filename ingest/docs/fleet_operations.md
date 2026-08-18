# Running a job across the whole cluster, unattended

Written 2026-08-15 from the VQA_first_run deployment (8.47M prompts,
Qwen3.6-27B, up to 24 nodes / 192 GPUs). Everything here was learned by
breaking it first. The reference implementation is
`examples/VQA_first_run/cluster/` — `supervisor.py`, `monitor.py`,
`launch_node.sh`, `run_services.sh`.

The engine (`src/vision_ingest/`) already handles one node correctly. This
document is about the layer above it: **finding capacity on a shared cluster,
keeping N nodes alive for days, and not damaging anyone else's work while
doing it.** None of that is in the engine and all of it was manual before.

---

## 1. The shape that works

Two long-lived user processes on the login node, plus a crontab line:

```
supervisor.py   every 90s   heal DB → probe all nodes → reap orphans →
                            yield contested nodes → relaunch dead → claim free
monitor.py      every 90s   read-only dashboard + /api JSON on a port
crontab         every 10m   run_services.sh start   (no-op when both are up)
```

Everything a node needs is already on shared storage (code on FSx, venv on
FSx, model in each node's NVMe HF cache), so "launching" is one `docker exec`
into a container that is already running there. No image is pulled, no
package installed, no daemon or system config touched. That property is what
makes the whole thing safe to run unattended on a machine other people use.

**Never let the supervisor and a manual `--dry-run` write the same status
file.** They clobber each other and you will debug a stale snapshot for
twenty minutes. Stop the service first, or point the dry run elsewhere.

---

## 2. The two-predicate rule

This is the single most important design decision in the supervisor.

```python
can_launch(node)   # strict:  every condition needed to START here
must_yield(node)   # narrow:  only HARD conflicts that make us LEAVE
```

They are **not** the same predicate negated. The first version used one for
both and it evicted four healthy, producing workers because other users had
ordinary CPU-only python processes on those nodes.

- Someone else's `python` is a reason **not to move in**.
- Only someone else's **GPU process** is a reason **to move out**.

In opportunistic mode (below), a SLURM reservation and a parked Ray raylet
also stop being yield reasons — otherwise you launch on a reserved-but-idle
node and evict yourself one cycle later, burning a 7-minute model load every
time. Observed: 97 launches of one node.

**Symmetry check before you ship a change to either predicate:** if condition
X blocks launching but does not trigger yielding, a node satisfying X can be
claimed and kept. If X triggers yielding but does not block launching, you
have built an infinite loop.

---

## 3. Opportunistic capacity

On a busy cluster, `SLURM idle AND unallocated` finds almost nothing. Most
free GPUs sit inside allocations whose owner is not currently computing.

```python
OPPORTUNISTIC = True     # take any node whose GPUs are genuinely idle:
                         # 0% utilisation, no compute apps, <1 GiB used —
                         # even if SLURM allocated it or Ray is parked on it
```

The safety valve is `must_yield` + a short cycle: the interval **is** the
worst-case window before the owner comes back and we get off. 90s.

This took the fleet from 5 nodes to 24. It is only defensible because
yielding is automatic and costs the owner nothing — our worker gets SIGINT,
`cli()`'s shielded teardown resets its in-flight rows to state=0, and they
replay elsewhere. Nothing is marked FAILED and no data is lost.

Order launches so uncontended nodes are taken first and opportunistic ones
only after; log every opportunistic launch as such so the operator can tell
the two apart in the morning.

---

## 4. Every failure hit, and what it actually was

### 4.1 Orphaned vLLM holding GPUs forever — *the expensive one*

**Symptom.** Node count plateaus. Nodes that were producing well stop writing
and are never relaunched. `can_launch` reports "gpus busy 6/8".

**Cause.** When a worker dies, `main.py` goes with it but its `vllm serve`
process tree survives — `online_worker._terminate_server` never runs. The
orphaned `VLLM::Worker_TP` processes are adopted by the container's PID 1 and
keep ~76 GB pinned on two GPUs indefinitely. vLLM needs all 8 for TP2×DP4, so
the node is permanently ineligible. **Nine nodes bled out this way in six
hours** — a node every ~10 minutes, invisible unless you look at GPU memory.

**Fix.** Auto-reap each cycle: our container holds GPU memory but has no
`main.py` → kill the leftovers. They **ignore SIGTERM** (observed state `Rl`,
actively spinning) — SIGKILL is required. First attempt with SIGTERM reported
`before == after` on all nine nodes.

**Detection to keep.** `mine == 0 and ours > 0 and foreign == 0`. If you only
watch "is the worker running", this failure is silent.

### 4.2 HuggingFace 429 on concurrent launches

**Symptom.** Launch 8 nodes at once; all die during engine init. Buried in the
log: `429 Client Error: Too Many Requests for .../api/models/Qwen/Qwen3.6-27B`.
Then the supervisor relaunches them, which produces more 429s.

**Cause.** vLLM calls the hub API on every startup *even when the weights are
fully cached locally*.

**Fix.** `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` in `main.py`, and
require the model cache in `can_launch` so offline can never mean "silently
missing". Any auxiliary tokenizer must then come from a local path — copy it
once to FSx and point at it, or token counting disables itself on every node.

### 4.3 `pgrep -f` matching the probe's own command line

**Symptom.** The supervisor reports a Ray cluster on every node it is
healthily running on, and wants to evict itself from all of them.

**Cause.** The probe script arrives as the ssh command line, so
`pgrep -f 'raylet'` matches the probe's own shell.

**Fix.** Bracket every pattern: `[r]aylet`, `[p]ython main.py`. Alternatively
pipe the script over stdin (`ssh node bash -s < probe.sh`), where the remote
cmdline is just `bash -s`. Use both.

### 4.4 A liveness probe that can time out will double-launch

**Symptom.** Risk of two vLLM servers on one node = OOM.

**Cause.** The probe's `docker exec pgrep` runs under a timeout. A node whose
model is still loading has no GPU memory allocated yet, so a timed-out probe
makes it look completely idle and eligible.

**Fix.** Re-check for a live worker **in the same ssh round trip as the
launch**, not in the earlier probe:

```bash
if docker exec $C pgrep -f '[p]ython main.py' >/dev/null 2>&1; then
  echo ALREADY_RUNNING; exit 0
fi
docker exec -d $C bash -lc '...'
```

Partially, the GPU check saves you anyway — a running worker makes the node
look busy — but only after the model has loaded. The window is exactly the
7-minute load.

### 4.5 A quarantine that never fires

**Symptom.** One node relaunched 97 times, another 57.

**Cause.** The failure counter stamped `last_launch = now` on every launch and
only counted a failure after 900s had elapsed *since that stamp*. Relaunching
every 2 minutes meant the threshold was never reached.

**Fix.** Count **strikes per launch**, cleared the moment the node appears in
`running`. Three strikes with nothing working in between → quarantine. Add a
`LAUNCH_COOLDOWN_S` at least as long as a model load (600s), so a node inside
its own startup is never relaunched on top of itself.

**Rule.** Never write a retry limiter whose clock is reset by the retry.

### 4.6 Checking for the deployment instead of the service

**Symptom.** 7 fully-healthy nodes excluded as "patram-fs not serving",
including four of the best producers.

**Cause.** The check tested `docker inspect patram-fs-prod`. The daemon had
been migrated fleet-wide to bare-metal tmux; the container no longer exists
anywhere.

**Fix.** Test the **socket**, by connecting to it:

```bash
timeout 4 python3 -c "import socket;s=socket.socket(socket.AF_UNIX);s.settimeout(3);s.connect('$SOCK')"
```

A stale socket file left by a dead daemon is byte-identical on disk to a live
one; only `connect()` distinguishes them, and `connect()` is what the client
library actually needs. This is now independent of how the daemon is deployed.

**Generalisation.** Never gate on *how* a dependency is packaged. Gate on
whether its endpoint answers.

### 4.7 Environment traps in the container

| trap | detail |
|---|---|
| `CUDA_VISIBLE_DEVICES` is set to the **empty string** | `os.environ.setdefault` is a no-op; test for empty, not unset |
| `vllm` is spawned as a bare subprocess | the venv's `bin/` must be on `PATH` even when you invoke its python by absolute path |
| the venv is Python **3.12**, hosts are **3.10** | `ocr_env_vllm` cannot run outside the container: `bin/python` resolves to the host's 3.10 and sees no packages. Docker supplies python3.12 + a 24.04 userspace (glibc 2.39 vs the host's 2.35), not the env — the env is on FSx and bind-mounted |
| postgres binaries are not on `PATH` | they live in `/usr/lib/postgresql/16/bin` |
| a directory validated at startup must exist at import | `VLLMConfig` checks `allowed_media_roots` before any hook object is constructed, so the scratch dir has to be created at module import, not in `__init__` |

### 4.8 `vlm_gpus=""` is load-bearing

`online_worker._launch_server` derives `--tensor-parallel-size` from the GPU
string whenever one is given, and `tensor_parallel_size` is in
`_SERVE_FLAG_SKIP`, so the yaml value is ignored:

```python
n_gpus = len(gpus.split(",")) if gpus else 0
if n_gpus > 0:   cmd += ["--tensor-parallel-size", str(n_gpus)]   # 8 — wrong
elif engine_args.get("tensor_parallel_size"): ...                 # the yaml — right
```

`--tensor-parallel-size 8 --data-parallel-size 4` asks for 32 GPUs and never
starts. Set `vlm_gpus=""` and export `CUDA_VISIBLE_DEVICES` from `main.py` —
the worker only overrides that env var when `gpus` is non-empty. Any layout
where TP ≠ GPU count needs this.

### 4.9 Smaller ones worth knowing

- **`rglob("*.jsonl")` swept up log files.** `finalize_records.py` treated
  `output/logs/**/ingest_batches.jsonl` as an output shard and produced 87
  bogus records. Match the writer's actual naming (`results_*.jsonl`).
- **FSx metadata lag.** `tail` shows a stale log and `ls` can report a size
  minutes behind; one read even failed with "no such file" on a file that
  existed. Trust `os.fstat` on an open fd, or the DB, over directory metadata.
- **fsync granularity distorts short throughput windows.** The writer flushes
  every 250 records, so a 90-second per-node sample lands on 1×, 2× or 3×
  flushes and looks like 2.9 / 5.8 / 8.7 rows/s. Measure over ≥5 minutes, or
  read `token_metrics` out of the pipeline's own checkpoints.
- **Port collisions on the login node.** Pick the dashboard port by binding,
  not by hoping.

---

## 5. Rules for touching a shared cluster

1. **Never kill what you did not start.** Every start and stop goes through
   `docker exec <our container>`, whose PID namespace contains only our
   processes.
2. **Verify the namespace, do not assume it.** `--ipc=host` shares only System
   V IPC; `--pid=host` is what would make container and host PIDs identical.
   Check `HostConfig.PidMode`, and prove it empirically — pick a host PID that
   does not belong to the container and confirm `/proc/<pid>` does not resolve
   inside it.
3. **Filter kills by user as well** (`pkill -u <you>`), so even a shared
   namespace could not reach another tenant.
4. **Re-verify at the moment of the action**, not in the probe that preceded
   it. Ownership can change between the two.
5. **Skip, do not guess.** Any GPU process owned by root or another user, any
   unexpected process name → leave the node untouched. Root-owned vLLM on a
   node you borrowed is almost certainly someone else's containerised job.
6. **Read-only sweeps first.** Classify by *ownership*, never by a category
   label like "gpus busy" — it hides whether the GPUs are theirs or your own
   corpse.
7. **Ask before the first destructive action of a kind**, then automate it
   once the guards are proven.

---

## 6. Database operations while live

| operation | safe live? | why |
|---|---|---|
| `reset-failed` (3 → 0) | **yes** | state=3 is terminal, no worker owns those rows |
| `reset-stuck` (1 → 0) | **no** | state=1 rows are owned by live workers; resetting hands the same row to a second node |
| `verify` | yes | read-mostly, fixes `state_counts` drift |

Stuck rows are bounded (~3 × batch_size per worker death) and are reset
automatically by each worker's own shutdown path. Let them be, or stop the
fleet first. The supervisor deliberately does **not** auto-reset them.

Also: the supervisor must stop launching when `pending + in_flight == 0`, or
at the end of the run it will spend the night relaunching workers to load a
27B model for seven minutes and exit again.

---

## 7. Deployment sequence that avoids wasted GPU-hours

1. Build the work list, verify it offline (`verify_setup.py`).
2. Start the DB, ingest once.
3. **Launch exactly one node. Wait for its first flush. Read the records.**
   This caught four separate startup bugs before they could be multiplied by
   the fleet, at a cost of ~15 minutes.
4. Fan out to a handful, confirm steady state.
5. Start the supervisor, let it find the rest.

Step 3 is not optional. Every bug in §4.7 would otherwise have appeared on
every node simultaneously.

---

## 8. What is still manual (i.e. what to automate next)

- **Creating the container on a node that lacks it.** `bash_scripts/
  init_docker.sh` does it in one call when the image is present; when it is
  not, it means a 38 GB image plus a 52 GB model download, which is usually
  not worth it. The supervisor should at least report the distinction rather
  than lumping both into "no container".
- **Partial-GPU nodes.** Nodes where another job holds 1–4 GPUs are skipped
  entirely because the layout is hardcoded to 8. A second yaml (TP2×DP2) plus
  pinning `CUDA_VISIBLE_DEVICES` to the free indices would have recovered ~18
  GPUs out of 31 idle ones during this run.
- **Failure triage.** Failures rose 8× (0.055% → 0.43%) and the cause was
  nodes dying, not the model. Nothing correlated failure counts with node
  health automatically; that was eyeballed.
- **Starting `patramd` where it is missing.** It is a glibc-only Go binary on
  FSx with env-var config, `--network host`, unprivileged — it runs bare on
  the host and needs no container and no root. Safe to automate.
- **Alerting.** Everything was discovered by looking. A "node count dropped
  below N" or "no node has written in M minutes" alert would have caught the
  orphan leak hours earlier.

---

## 9. Quick reference

```bash
# services
bash cluster/run_services.sh start|status|stop|restart

# dashboard (backend 90s, page 90s)
ssh -N -L 8911:localhost:8911 <login-node>     # http://localhost:8911
curl -s localhost:8911/api | python3 -m json.tool

# one supervisor cycle, no side effects  (stop the service first!)
python3 cluster/supervisor.py --once --dry-run

# what the whole cluster looks like and why each node is unusable
python3 -c "import json,collections;d=json.load(open('output/cluster_status.json'));
print(collections.Counter(d['blocked_reasons'].values()).most_common())"

# throughput and tokens from the pipeline's own checkpoints (no measurement window)
grep '"processing_checkpoint"' output/logs/<node>/*/main.log | tail -1
```

Key knobs in `supervisor.py`: `OPPORTUNISTIC`, `INTERVAL_S`,
`MAX_LAUNCHES_PER_CYCLE`, `FAILURE_BUDGET`, `QUARANTINE_S`,
`LAUNCH_COOLDOWN_S`.
