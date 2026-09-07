#!/usr/bin/env python
"""Drive synthetic participants through the real HTTP API.

Two jobs: fill a scratch database so the admin views have something to show,
and load the server hard enough that concurrency bugs surface before three
real people meet them on Wednesday.

    python scripts/simulate_participants.py --n 30 --threads 8
    python scripts/simulate_participants.py --n 50 --threads 16 --stress

Response policies are not decoration -- they are how the QC rules get tested.
An `honest` participant must never be flagged and a `straightliner` must be,
and neither can be checked with a single well-behaved fake.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

BASE = "http://localhost:8607"


class Client:
    def __init__(self, base):
        self.base = base

    def call(self, path, data=None, pid=None, timeout=30):
        req = urllib.request.Request(
            self.base + path, method="POST" if data is not None else "GET",
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json",
                     **({"X-Participant": pid} if pid else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except Exception:
                return e.code, {}


POLICIES = ("honest", "straightliner", "speedrunner", "left_biased", "abandoner")


def answer(policy, question, rng):
    kind = question["type"]
    if kind == "likert5":
        return 4 if policy == "straightliner" else rng.randint(1, 5)
    if kind == "yesno":
        return True if policy == "straightliner" else rng.random() > 0.4
    if kind == "choice_ab":
        if policy == "left_biased":
            return "A"
        return rng.choice(["A", "B", "no_preference"])
    if kind == "score10":
        return 8 if policy == "straightliner" else rng.randint(0, 10)
    if kind == "choice_pair":
        return "A" if policy == "left_biased" else rng.choice(["A", "B", "tie"])
    if kind == "select":
        options = question.get("options") or [{"value": "x"}]
        return (options[0]["value"] if policy == "straightliner"
                else rng.choice(options)["value"])
    return "simulated note"


def run_one(client, index, policy, seed, stats, lock, stress=False):
    rng = random.Random(seed)
    status, reg = client.call("/api/register", {
        "display_name": "Sim %03d" % index,
        "education_level": rng.choice(["student", "faculty", "researcher"]),
        "roll_no": "SIM%03d" % index, "area": "simulated",
        "reads_papers": rng.choice(["weekly", "monthly", "rarely"]),
        "consent": True})
    if status != 200:
        with lock:
            stats["errors"].append("register %s" % status)
        return
    pid = reg["participant_id"]

    served = 0
    while True:
        t0 = time.time()
        status, trial = client.call("/api/trial/current", pid=pid)
        if status != 200:
            with lock:
                stats["errors"].append("trial %s %s" % (status, trial))
            return
        with lock:
            stats["latency"].append(time.time() - t0)
        if trial.get("done") or trial.get("needs_calibration"):
            break

        # An abandoner walks away mid-session, leaving an open trial behind.
        if policy == "abandoner" and served >= 3:
            with lock:
                stats["abandoned"] += 1
            break

        tid = trial["trial_id"]
        client.call("/api/trial/%s/answer" % tid,
                    {"question_id": "familiarity",
                     "value": rng.choice(["not_familiar", "somewhat", "familiar"])}, pid)

        # Playback events, so the QC derivations have something to read.
        events = []
        for slot in trial["slots"]:
            events += [
                {"trial_id": tid, "type": "animation_play", "slot": slot["slot"],
                 "t_video": 0.0, "client_seq": len(events)},
                {"trial_id": tid, "type": "animation_complete", "slot": slot["slot"],
                 "t_video": slot["duration"], "client_seq": len(events) + 1,
                 "payload": {"watched_fraction": 0.62 if policy == "speedrunner" else 0.97}}]
        client.call("/api/events", {"events": events}, pid)

        # A tournament has no form: two picks, posted with the submit.
        if trial.get("screen") == "tournament":
            r1 = "A" if policy == "left_biased" else rng.choice(["A", "B"])
            r2 = r1 if policy in ("left_biased", "straightliner") else rng.choice([r1, "C"])
            status, body = client.call("/api/trial/%s/submit" % tid,
                                       {"picks": [r1, r2]}, pid)
            if status != 200:
                with lock:
                    stats["errors"].append("tour submit %s %s" % (status, body))
                return
            served += 1
            continue

        # Questions reveal progressively (show_if), so answer in order and
        # only what the answers so far make visible -- exactly as the form does.
        answers = {}
        for q in trial["questions"]["questions"]:
            cond = q.get("show_if") or {}
            if any(answers.get(k) != v for k, v in cond.items()):
                continue
            if q.get("optional") and rng.random() < 0.7:
                continue
            answers[q["id"]] = answer(policy, q, rng)
            client.call("/api/trial/%s/answer" % tid,
                        {"question_id": q["id"], "value": answers[q["id"]],
                         "ms_since_open": rng.randint(4000, 90000)}, pid)
            # Revisions are expected behaviour, so exercise the append path --
            # but not on a gate question: flipping vfs after skipping ascs is
            # the one revision the real form undoes (it re-reveals the
            # dependents), and a blind re-post here just earns a 409.
            gates = {k for qq in trial["questions"]["questions"]
                     for k in (qq.get("show_if") or {})}
            if policy == "honest" and rng.random() < 0.08 and q["id"] not in gates:
                answers[q["id"]] = answer(policy, q, rng)
                client.call("/api/trial/%s/answer" % tid,
                            {"question_id": q["id"], "value": answers[q["id"]]}, pid)

        status, body = client.call("/api/trial/%s/submit" % tid, {}, pid)
        if status != 200:
            with lock:
                stats["errors"].append("submit %s %s" % (status, body))
            return
        served += 1

        # A reload mid-session must resume the identical trial, not mint one.
        if stress and rng.random() < 0.15:
            _, a = client.call("/api/trial/current", pid=pid)
            _, b = client.call("/api/trial/current", pid=pid)
            if a.get("trial_id") != b.get("trial_id"):
                with lock:
                    stats["errors"].append("resume mismatch")

    with lock:
        stats["served"].append(served)
        stats["policies"][policy] += served


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stress", action="store_true",
                    help="add reload storms and check resume under load")
    args = ap.parse_args()

    client = Client(args.base)
    try:
        urllib.request.urlopen(args.base + "/", timeout=5)
    except Exception as exc:                                   # noqa: BLE001
        print("server not reachable on %s: %s" % (args.base, exc))
        return 2

    stats = {"served": [], "errors": [], "latency": [], "abandoned": 0,
             "policies": Counter()}
    lock = threading.Lock()
    queue = list(range(args.n))
    rng = random.Random(args.seed)
    policies = [rng.choice(POLICIES) if args.stress else "honest"
                for _ in range(args.n)]

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                i = queue.pop()
            run_one(client, i, policies[i], args.seed * 1000 + i, stats, lock,
                    args.stress)

    started = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started

    total = sum(stats["served"])
    print("\n%d participants on %d threads in %.1fs" % (args.n, args.threads, elapsed))
    print("  trials submitted : %d" % total)
    print("  per participant  : min %s  median %s  max %s" % (
        min(stats["served"] or [0]),
        statistics.median(stats["served"]) if stats["served"] else 0,
        max(stats["served"] or [0])))
    if stats["latency"]:
        lat = sorted(stats["latency"])
        print("  assignment latency: median %.0fms  p95 %.0fms  max %.0fms" % (
            1000 * statistics.median(lat), 1000 * lat[int(len(lat) * 0.95)],
            1000 * lat[-1]))
    if args.stress:
        print("  abandoned mid-session: %d" % stats["abandoned"])
        print("  by policy: %s" % dict(stats["policies"]))
    print("  errors: %d" % len(stats["errors"]))
    for e in stats["errors"][:8]:
        print("    %s" % e)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
