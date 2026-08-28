"""Measure serving throughput while a run is in flight.

WHY THIS EXISTS
---------------
v3 telemetry (docs/ANIMATION_TREE_TELEMETRY.md) caught the server reporting
`Running: 1 reqs` against `--max-num-seqs 16` for 3.8 hours -- ~94% of serving
capacity idle, because the DRIVER was sequential, not because the GPUs were
slow. Aggregate tok/s alone does not reveal that; you have to look at how many
requests are actually resident. So this samples vLLM's own /metrics and reports
BOTH: throughput, and the concurrency that produced it.

    python3 scripts/throughput.py --port 8011 --interval 5 --out t.json
    # ...run a pipeline in another shell...
    # Ctrl-C, or --duration N

Reads the Prometheus endpoint, which is authoritative -- it is vLLM's own
accounting rather than something inferred from wall-clock at this end.
"""
from __future__ import annotations

import argparse, json, re, time, urllib.request


# vLLM exposes counters (monotonic totals) and gauges (instantaneous). Deltas of
# the counters give rate; the gauges give the concurrency picture.
COUNTERS = {"prompt": "vllm:prompt_tokens_total",
            "gen":    "vllm:generation_tokens_total",
            "done":   "vllm:request_success_total"}
GAUGES   = {"running": "vllm:num_requests_running",
            "waiting": "vllm:num_requests_waiting",
            "kv":      "vllm:gpu_cache_usage_perc"}


def scrape(port: int) -> dict:
    """One /metrics read -> flat {name: summed value}."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if not m:
            continue
        try:
            # Labelled series (one per model) are summed: a single server holds
            # one model here, so this is a sum over a single element.
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to sample; omit to run until interrupted")
    ap.add_argument("--label", default="", help="model name, recorded in the output")
    ap.add_argument("--out", help="write the JSON summary here")
    args = ap.parse_args()

    t0 = time.time()
    first = prev = scrape(args.port)
    prev_t = t0
    peak_run = peak_wait = 0
    # Only samples with work in flight count toward the active average. Idle
    # samples would otherwise drag the mean toward zero and hide the real rate.
    active = []

    print(f"sampling :{args.port} every {args.interval}s  (Ctrl-C to stop)\n")
    print(f"{'t':>6} {'prompt/s':>10} {'gen/s':>9} {'run':>4} {'wait':>5} {'kv%':>6}")
    try:
        while args.duration is None or time.time() - t0 < args.duration:
            time.sleep(args.interval)
            now, t = scrape(args.port), time.time()
            dt = t - prev_t
            p = (now.get(COUNTERS["prompt"], 0) - prev.get(COUNTERS["prompt"], 0)) / dt
            g = (now.get(COUNTERS["gen"], 0) - prev.get(COUNTERS["gen"], 0)) / dt
            run = now.get(GAUGES["running"], 0)
            wait = now.get(GAUGES["waiting"], 0)
            kv = now.get(GAUGES["kv"], 0) * 100
            peak_run, peak_wait = max(peak_run, run), max(peak_wait, wait)
            if run > 0 or g > 0:
                active.append((p, g, run))
            print(f"{t-t0:6.0f} {p:10.1f} {g:9.1f} {run:4.0f} {wait:5.0f} {kv:6.2f}")
            prev, prev_t = now, t
    except KeyboardInterrupt:
        print("\n(interrupted)")

    total_t = time.time() - t0
    dp = scrape(args.port).get(COUNTERS["prompt"], 0) - first.get(COUNTERS["prompt"], 0)
    dg = scrape(args.port).get(COUNTERS["gen"], 0) - first.get(COUNTERS["gen"], 0)
    dn = scrape(args.port).get(COUNTERS["done"], 0) - first.get(COUNTERS["done"], 0)

    summary = {
        "label": args.label, "elapsed_s": round(total_t, 1),
        "prompt_tokens": int(dp), "generation_tokens": int(dg),
        "requests_completed": int(dn),
        "prompt_tok_s_overall": round(dp / total_t, 1) if total_t else 0,
        "gen_tok_s_overall": round(dg / total_t, 1) if total_t else 0,
        # The honest number for "how fast is this model": measured only while
        # requests were actually resident.
        "gen_tok_s_active": round(sum(a[1] for a in active) / len(active), 1) if active else 0,
        "mean_running_active": round(sum(a[2] for a in active) / len(active), 2) if active else 0,
        "peak_running": peak_run, "peak_waiting": peak_wait,
        "s_per_request": round(total_t / dn, 1) if dn else None,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    if summary["peak_running"] <= 1:
        print("\n  WARNING: peak concurrency <= 1. The driver is the bottleneck,\n"
              "  not the GPUs -- this is the v3 failure repeating.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
