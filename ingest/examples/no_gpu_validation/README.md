# No-GPU validation harness (v1.9.2 §8)

Everything in `docs/v1.9.2_changes.md` §8 was verified with these, on fake backends only —
no vLLM import, no GPU, no database. They are the reason the release could be reviewed at all:
the failure modes that matter here are all "a process died at the wrong moment", which is
exactly what a real model makes expensive and slow to provoke.

| Script | What it proves |
|---|---|
| `pool_harness.py` | Scenarios 1-5, 7-9: clean sync run; async concurrency up to the declared ceiling; SIGKILL of a busy worker (request recovered as `infra` on the right queue, rank respawned, service never reported failed); a crash-looping `init_fn` exhausting its respawn budget; SIGKILL under `on_worker_death="abort"`; `BackendGone`; two async workers load-balancing one queue with one then killed; and the `WorkerSpec` validation that refuses a sync `call_fn` declaring concurrency. |
| `leader_kill.py` | Scenario 6: two spawn levels (main -> group leader -> stage worker, StageWeaver's shape). SIGKILL the group leader; the stage must infer it from the stale heartbeat and abort. Takes `STALE_S` (30s) by design. |
| `wiring_sweep.py` | Scenario 9: locked signatures unchanged, every example and parent-repo `init_fn`/`term_fn` present with the right shape and free of deleted API in executable code, deleted modules actually gone, both vLLM specs build and pickle without a GPU, and a `ServiceStatus` survives the two-hop channel into a leaf worker. |
| `mp_only.py` | ~20 lines of pure `multiprocessing`, no framework: two consumers on one `mp.Queue`, SIGKILL the idle one, check whether the survivor can still consume. Reports a rate over several trials, because whether a kill lands inside the queue's reader lock is a race (measured 3/8 with two consumers; roughly 1/N in general, and total whenever it lands). This is the bug that forced the pool to dispatch to per-worker inboxes instead of letting workers share the request queue (§ "As built" deviation 1). Keep it: the shared-queue design reads better than the one that replaced it, and this is the 30 seconds of evidence for why it cannot be used. |

```bash
cd examples/no_gpu_validation
python pool_harness.py          # all scenarios; or e.g. `python pool_harness.py 3 8`
python wiring_sweep.py
python leader_kill.py           # ~35s
python mp_only.py
```

Still **not** covered, as in v1.9.1: a scripted kill against a real `vllm serve`, and measuring
whether `STALE_S = 30s` is the right number.
