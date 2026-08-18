"""
§8 scenario 6: SIGKILL the GROUP LEADER (the process that owns the pool).

Two spawn levels, exactly StageWeaver's shape:
    main  ->  group leader (builds channel, starts WorkerPool)
                 -> stage worker (reads channel.health, like cli() does)

Nothing cooperative can report a SIGKILLed pool, so the stage must infer it from
the stale heartbeat and abort — with rows reset, nothing marked FAILED.
"""
import multiprocessing as mp
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from vision_ingest.core.channel import build_channel
from vision_ingest.core.service import WorkerPool, WorkerSpec
from vision_ingest.core.status import STALE_S

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def w_init(args, rank, device):
    return {"rank": rank}


def w_call(ctx, payload, params):
    time.sleep(0.05)
    return {"ok": payload["n"]}


def stage_worker(channel, result_q):
    """Stands in for cli(): wait for ready, then poll failure_reason() forever."""
    status = channel.health
    result_q.put(("has_status", status is not None))
    t0 = time.time()
    while not status.is_ready():
        if time.time() - t0 > 60:
            result_q.put(("ready", False)); return
        time.sleep(0.1)
    result_q.put(("ready", True))

    t0 = time.time()
    while True:
        reason = status.failure_reason()
        if reason:
            result_q.put(("detected", round(time.time() - t0, 1), reason))
            return
        if time.time() - t0 > STALE_S + 30:
            result_q.put(("timeout", round(time.time() - t0, 1), None))
            return
        time.sleep(0.25)


def group_leader(result_q, ready_q):
    channel = build_channel(["a"])
    spec = WorkerSpec(init_fn=w_init, call_fn=w_call, n_workers=2)
    pool = WorkerPool(spec, channel, logs_dir=LOGS)
    pool.start()
    stage = mp.Process(target=stage_worker, args=(channel, result_q), daemon=False)
    stage.start()
    ready_q.put(os.getpid())
    stage.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.makedirs(LOGS, exist_ok=True)
    result_q, ready_q = mp.Queue(), mp.Queue()
    leader = mp.Process(target=group_leader, args=(result_q, ready_q))
    leader.start()
    leader_pid = ready_q.get(timeout=60)
    print(f"group leader pid={leader_pid}")

    events = {}
    for _ in range(2):
        msg = result_q.get(timeout=90)
        events[msg[0]] = msg[1:]
        print(f"  stage reports: {msg}")
    assert events.get("ready") == (True,), events

    print(f"SIGKILL the group leader (STALE_S={STALE_S})...")
    os.kill(leader_pid, signal.SIGKILL)
    t0 = time.time()
    msg = result_q.get(timeout=STALE_S + 60)
    wall = time.time() - t0
    print(f"  stage reports: {msg[0]} after {wall:.1f}s wall")
    print(f"  reason: {msg[2]}")
    ok = (msg[0] == "detected"
          and "not supervising" in (msg[2] or "")
          and wall <= STALE_S + 5)
    print("=> " + ("PASS" if ok else "FAIL"))
    leader.join(timeout=10)
    sys.exit(0 if ok else 1)
