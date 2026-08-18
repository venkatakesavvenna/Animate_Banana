"""
Why the pool dispatches to per-worker inboxes instead of letting N workers share
one `request_queue` (v1.9.2 §"As built" deviation 1).

No framework, no vLLM, no GPU: two consumers on one `mp.Queue`, SIGKILL the idle
one, then check whether the survivor can still consume.

`mp.Queue.get(timeout=...)` acquires the queue's shared reader lock, polls the
pipe, and releases it. A process SIGKILLed while holding that lock never releases
it — it is a POSIX semaphore in shared memory and CPython has no robust-futex
recovery — so every surviving consumer blocks on it forever.

Whether a given kill lands inside the lock is a race, so this runs several trials
and reports the rate. An IDLE consumer spends nearly all of its time inside
`get()`, which is why the rate is high rather than negligible: in the pool that
meant killing an idle worker would poison the queue, the pool would recover the
dead worker's requests, respawn it, report the service healthy — and consume
nothing ever again. A silent, permanent stall.

Run: python mp_only.py [trials]
"""
import multiprocessing as mp
import os
import signal
import sys
import time
from queue import Empty

N_ITEMS = 6


def consumer(q, out):
    while True:
        try:
            item = q.get(timeout=0.25)
        except Empty:
            continue
        if item is None:
            return
        out.put((os.getpid(), item))


def drain(out, want, timeout):
    got, end = [], time.time() + timeout
    while len(got) < want and time.time() < end:
        try:
            got.append(out.get(timeout=0.25))
        except Empty:
            pass
    return got


def trial(i):
    q, out = mp.Queue(), mp.Queue()
    procs = [mp.Process(target=consumer, args=(q, out), daemon=True) for _ in range(2)]
    for p in procs:
        p.start()

    for n in range(N_ITEMS):
        q.put(n)
    warm = drain(out, N_ITEMS, 20)
    if len(warm) != N_ITEMS:
        print(f"  trial {i}: inconclusive — warmup consumed {len(warm)}/{N_ITEMS}")
        return None

    # Both consumers are now idle, i.e. sitting inside q.get() with the shared
    # reader lock held for most of every 0.25s poll.
    time.sleep(0.5)
    os.kill(procs[0].pid, signal.SIGKILL)
    time.sleep(0.5)

    for n in range(100, 100 + N_ITEMS):
        q.put(n)
    after = drain(out, N_ITEMS, 5)
    wedged = len(after) < N_ITEMS
    print(f"  trial {i}: survivor consumed {len(after)}/{N_ITEMS} after the kill"
          f"  ->  {'QUEUE POISONED' if wedged else 'queue survived'}"
          f"   (qsize={q.qsize()})")

    for p in procs:
        if p.is_alive():
            p.terminate()
    q.cancel_join_thread()
    out.cancel_join_thread()
    return wedged


if __name__ == "__main__":
    mp.set_start_method("spawn")
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"two consumers on one mp.Queue; SIGKILL the idle one; {trials} trials")
    results = [r for r in (trial(i + 1) for i in range(trials)) if r is not None]
    wedged = sum(results)
    print(f"\npoisoned {wedged}/{len(results)} trials")
    if wedged:
        print("=> A shared-consumer mp.Queue does not survive a SIGKILLed consumer.\n"
              "   This is why WorkerPool is the request queue's only consumer and each\n"
              "   worker reads a private inbox it alone consumes — a killed worker can\n"
              "   then only poison the queue the pool is about to throw away.")
    else:
        print("=> No trial poisoned the queue this run. It is a race on whether the\n"
              "   victim held the queue's reader lock at that instant, so a clean run\n"
              "   is not evidence of safety — re-run with more trials.")
    sys.exit(0)
