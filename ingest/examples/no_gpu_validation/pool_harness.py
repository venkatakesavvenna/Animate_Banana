"""
v1.9.2 §8 no-GPU validation harness. Fake specs only — no vLLM, no GPU.

Run: python pool_harness.py [scenario ...]
"""
import asyncio
import multiprocessing as mp
import os
import signal
import sys
import time
from queue import Empty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from vision_ingest.core.channel import build_channel
from vision_ingest.core.service import (
    ON_DEATH_ABORT, ON_DEATH_RESPAWN, WorkerPool, WorkerSpec,
)
from vision_ingest.core.task import BackendGone, InputRejected
from vision_ingest.core.wire import KIND_INFRA, KIND_OK, KIND_REJECT, VLLMRequest

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# --------------------------------------------------------------------------
# fake backends (module-level so they pickle into a spawned child)
# --------------------------------------------------------------------------


def fake_init(args, rank, device):
    return {"rank": rank, "device": device, "sleep": getattr(args, "sleep", 0.0)}


def fake_call(ctx, payload, params):
    if payload.get("kind") == "reject":
        raise InputRejected("bad input")
    if payload.get("kind") == "boom":
        raise RuntimeError("kaboom")
    if payload.get("kind") == "gone":
        raise BackendGone("my fake server exited")
    if ctx["sleep"]:
        time.sleep(ctx["sleep"])
    return {"echo": payload.get("n"), "rank": ctx["rank"]}


def fake_term(ctx):
    pass


def crash_init(args, rank, device):
    raise RuntimeError("init_fn is broken on purpose")


async def async_init(args, rank, device):
    await asyncio.sleep(0.05)
    return {"rank": rank, "cur": args.cur, "peak": args.peak, "sleep": args.sleep}


async def async_call(ctx, payload, params):
    cur, peak = ctx["cur"], ctx["peak"]
    with cur.get_lock():
        cur.value += 1
        if cur.value > peak.value:
            peak.value = cur.value
    try:
        await asyncio.sleep(payload.get("sleep", ctx["sleep"]))
        return {"echo": payload.get("n"), "rank": ctx["rank"]}
    finally:
        with cur.get_lock():
            cur.value -= 1


async def async_term(ctx):
    await asyncio.sleep(0)


def hog_init(args, rank, device):
    return {"rank": rank}


async def hog_call(ctx, payload, params):
    """Declares k=2 but the framework's semaphore holds us to it; overflow is
    provoked by calling set_inflight directly from the call."""
    await asyncio.sleep(0.05)
    return {"echo": payload.get("n")}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def submit(ch, stage, n, kind=None, **extra):
    payload = {"n": n}
    if kind:
        payload["kind"] = kind
    payload.update(extra)
    req = VLLMRequest(req_id=f"{'%036d' % n}", stage_name=stage, prompt=payload,
                      sampling_params=None)
    ch.request_queue.put(req)
    return req.req_id


def collect(ch, stage, count, timeout=30.0):
    out = []
    end = time.time() + timeout
    while len(out) < count and time.time() < end:
        try:
            out.append(ch.response_queues[stage].get(timeout=0.25))
        except Empty:
            pass
    return out


def wait_ready(pool, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pool.status.is_ready():
            return True
        if pool.status.failure_reason():
            return False
        time.sleep(0.05)
    return False


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        globals()["FAILURES"].append(name)


FAILURES = []


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

def s1_sync_clean():
    print("[1] sync spec, clean run")
    ch = build_channel(["a"])
    spec = WorkerSpec(init_fn=fake_init, call_fn=fake_call, term_fn=fake_term,
                      args={"sleep": 0.0}, n_workers=3)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    check("status published on channel", ch.health is pool.status)
    check("becomes ready", wait_ready(pool))
    for n in range(30):
        submit(ch, "a", n)
    submit(ch, "a", 100, kind="reject")
    submit(ch, "a", 101, kind="boom")
    got = collect(ch, "a", 32)
    kinds = [r.kind for r in got]
    check("32 responses", len(got) == 32, f"got {len(got)}")
    check("30 ok", kinds.count(KIND_OK) == 30, str(kinds.count(KIND_OK)))
    check("1 reject", kinds.count(KIND_REJECT) == 1)
    check("1 infra", kinds.count(KIND_INFRA) == 1)
    ranks = {r.stats.get("worker_rank") for r in got if r.kind == KIND_OK}
    check("work spread over >1 worker", len(ranks) > 1, str(ranks))
    check("stats carry worker identity",
          all("duration_ms" in r.stats for r in got if r.kind == KIND_OK))
    check("ledger empty", pool.inflight_count() == 0,
          str(pool.inflight_count()))
    pool.stop(timeout=20)
    check("no failure recorded on clean stop", pool.status.failure_reason() is None,
          str(pool.status.failure_reason()))
    ch.cancel_join_threads()


def s2_async_concurrency():
    print("[2] async spec, max_inflight=8 — real concurrency")
    ch = build_channel(["a"])
    cur, peak = mp.Value("i", 0), mp.Value("i", 0)
    spec = WorkerSpec(init_fn=async_init, call_fn=async_call, term_fn=async_term,
                      args={"cur": cur, "peak": peak, "sleep": 0.3},
                      n_workers=1, max_inflight_per_worker=8)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    check("becomes ready", wait_ready(pool))
    for n in range(24):
        submit(ch, "a", n)
    got = collect(ch, "a", 24, timeout=40)
    check("24 responses", len(got) == 24, f"got {len(got)}")
    check("all ok", all(r.kind == KIND_OK for r in got))
    check("concurrency >= 2", peak.value >= 2, f"peak={peak.value}")
    check("concurrency <= declared 8", peak.value <= 8, f"peak={peak.value}")
    check("ledger drained", pool.inflight_count() == 0)
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s3_sigkill_respawn():
    print("[3] SIGKILL one worker of 3, on_worker_death=respawn")
    ch = build_channel(["a"])
    spec = WorkerSpec(init_fn=fake_init, call_fn=fake_call, term_fn=fake_term,
                      args={"sleep": 6.0}, n_workers=3,
                      on_worker_death=ON_DEATH_RESPAWN)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    wait_ready(pool)
    for n in range(3):
        submit(ch, "a", n)
    # let all three pick up work
    deadline = time.time() + 5
    while pool.inflight_count() < 3 and time.time() < deadline:
        time.sleep(0.05)
    check("all 3 workers hold a request", pool.inflight_count() == 3,
          str(pool.inflight_count()))
    victim_rank = 0
    with pool._lock:
        held = [(s_, r.req_id) for s_, r in pool._outstanding(victim_rank)]
    check("victim's assignment is recorded in the pool", len(held) == 1, str(held))
    victim_pid = pool._procs[victim_rank].pid
    t0 = time.time()
    os.kill(victim_pid, signal.SIGKILL)
    # the killed worker's request must come back as infra, fast
    recovered, elapsed = None, None
    end = time.time() + 5
    while time.time() < end:
        try:
            r = ch.response_queues["a"].get(timeout=0.2)
        except Empty:
            continue
        if r.kind == KIND_INFRA:
            recovered, elapsed = r, time.time() - t0
            break
    check("in-flight recovered as infra", recovered is not None)
    if recovered:
        check("recovered the RIGHT req_id", recovered.req_id == held[0][1],
              f"{recovered.req_id} vs {held[0][1]}")
        check("recovered in <= 1.5s", elapsed <= 1.5, f"{elapsed:.2f}s")
    # And the rank comes back. Poll on the PID changing, not on the rank being
    # present: the infra response is emitted BEFORE the supervisor pops the dead
    # proc, so the dead entry is still in _procs at this instant.
    end = time.time() + 10
    while time.time() < end:
        proc = pool._procs.get(victim_rank)
        if proc is not None and proc.pid != victim_pid:
            break
        time.sleep(0.1)
    check("rank respawned with a NEW pid",
          pool._procs.get(victim_rank) is not None
          and pool._procs[victim_rank].pid != victim_pid,
          f"old={victim_pid} now={getattr(pool._procs.get(victim_rank), 'pid', None)}")
    check("service never reported failed", pool.status.failure_reason() is None,
          str(pool.status.failure_reason()))
    check("still ready", pool.status.is_ready())
    collect(ch, "a", 2, timeout=15)
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s4_crash_looping_init():
    print("[4] crash-looping init_fn — respawn budget exhausted")
    ch = build_channel(["a"])
    spec = WorkerSpec(init_fn=crash_init, call_fn=fake_call, n_workers=2,
                      on_worker_death=ON_DEATH_RESPAWN, max_respawns=1)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    reason, end = None, time.time() + 40
    while time.time() < end:
        reason = pool.status.failure_reason()
        if reason:
            break
        time.sleep(0.1)
    check("service reports failure", reason is not None, str(reason))
    check("reason names it a startup failure",
          bool(reason) and "failed to start" in reason.lower(), str(reason))
    check("never became ready", not pool.status.is_ready())
    check("stop_event set", ch.stop_event.is_set())
    check("both ranks abandoned", len(pool._abandoned) == 2, str(pool._abandoned))
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s5_sigkill_abort():
    print("[5] SIGKILL one worker, on_worker_death=abort")
    ch = build_channel(["a"])
    spec = WorkerSpec(init_fn=fake_init, call_fn=fake_call, args={"sleep": 6.0},
                      n_workers=3, on_worker_death=ON_DEATH_ABORT)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    wait_ready(pool)
    for n in range(3):
        submit(ch, "a", n)
    deadline = time.time() + 5
    while pool.inflight_count() < 3 and time.time() < deadline:
        time.sleep(0.05)
    t0 = time.time()
    os.kill(pool._procs[1].pid, signal.SIGKILL)
    reason, end = None, time.time() + 5
    while time.time() < end:
        reason = pool.status.failure_reason()
        if reason:
            break
        time.sleep(0.05)
    elapsed = time.time() - t0
    check("failure_reason set", reason is not None, str(reason))
    check("within 1.5s (was 30s in v1.9.1)", elapsed <= 1.5, f"{elapsed:.2f}s")
    check("detail names the rank and exit code",
          bool(reason) and "rank 1" in reason and "-9" in reason, str(reason))
    check("stop_event set", ch.stop_event.is_set())
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s7_backend_gone():
    print("[7] call_fn raises BackendGone")
    ch = build_channel(["a"])
    spec = WorkerSpec(init_fn=fake_init, call_fn=fake_call, term_fn=fake_term,
                      args={"sleep": 0.0}, n_workers=1,
                      on_worker_death=ON_DEATH_ABORT)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    wait_ready(pool)
    submit(ch, "a", 1)
    submit(ch, "a", 2, kind="gone")
    got = collect(ch, "a", 2, timeout=15)
    check("both answered", len(got) == 2, f"got {len(got)}")
    check("the gone one is infra, not reject",
          any(r.kind == KIND_INFRA and "backend gone" in (r.error or "") for r in got),
          str([(r.kind, r.error) for r in got]))
    reason, end = None, time.time() + 10
    while time.time() < end:
        reason = pool.status.failure_reason()
        if reason:
            break
        time.sleep(0.1)
    check("service failed with the resource-gone reason", reason is not None, str(reason))
    check("reason distinguishes it from a plain crash",
          bool(reason) and "resource is gone" in reason, str(reason))
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s8_two_async_workers():
    print("[8] two async workers, one queue — load balancing + one killed")
    ch = build_channel(["a"])
    cur, peak = mp.Value("i", 0), mp.Value("i", 0)
    spec = WorkerSpec(init_fn=async_init, call_fn=async_call, term_fn=async_term,
                      args={"cur": cur, "peak": peak, "sleep": 0.2},
                      n_workers=2, max_inflight_per_worker=4,
                      on_worker_death=ON_DEATH_RESPAWN)
    pool = WorkerPool(spec, ch, logs_dir=LOGS)
    pool.start()
    check("becomes ready", wait_ready(pool))
    for n in range(40):
        submit(ch, "a", n)
    got = collect(ch, "a", 40, timeout=40)
    ranks = {r.stats.get("worker_rank") for r in got if r.kind == KIND_OK}
    check("40 responses", len(got) == 40, f"got {len(got)}")
    check("both workers drained", ranks == {0, 1}, str(ranks))
    # kill one; the other must keep serving
    os.kill(pool._procs[0].pid, signal.SIGKILL)
    time.sleep(1.2)
    for n in range(100, 120):
        submit(ch, "a", n)
    got2 = collect(ch, "a", 20, timeout=30)
    check("survivor kept draining", len(got2) == 20, f"got {len(got2)}")
    check("service never failed", pool.status.failure_reason() is None,
          str(pool.status.failure_reason()))
    pool.stop(timeout=20)
    ch.cancel_join_threads()


def s9_sync_rejects_concurrency():
    print("[9] a sync call_fn declaring concurrency is refused at construction")
    try:
        WorkerSpec(init_fn=fake_init, call_fn=fake_call,
                   max_inflight_per_worker=8).resolved_max_inflight()
        check("raises ValueError", False, "no error raised")
    except ValueError as e:
        check("raises ValueError", True)
        check("message explains why", "async def" in str(e), str(e))
    try:
        WorkerSpec(init_fn=fake_init, call_fn=fake_call,
                   on_worker_death="explode").resolved_on_worker_death()
        check("bad on_worker_death raises", False)
    except ValueError:
        check("bad on_worker_death raises", True)


SCENARIOS = {
    "1": s1_sync_clean, "2": s2_async_concurrency, "3": s3_sigkill_respawn,
    "4": s4_crash_looping_init, "5": s5_sigkill_abort, "7": s7_backend_gone,
    "8": s8_two_async_workers, "9": s9_sync_rejects_concurrency,
}

if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.makedirs(LOGS, exist_ok=True)
    want = sys.argv[1:] or list(SCENARIOS)
    for key in want:
        SCENARIOS[key]()
    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL SCENARIOS PASSED")
