#!/usr/bin/env python3
"""
Benchmark fetch_batch, mark_done, mark_failed against a live PostgreSQL DB.

Pre-condition: DB must have at least (--n-batches * --batch-size) paths in state=0.
Run ingest first.
"""
import argparse
import sys
import time

from vision_ingest.db.db import DB


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DB operations")
    p.add_argument("--pg-host", required=True)
    p.add_argument("--pg-port", type=int, default=5432)
    p.add_argument("--pg-dbname", required=True)
    p.add_argument("--pg-user", required=True)
    p.add_argument("--pg-password", default=None)
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Paths per fetch_batch call (default: 1000)")
    p.add_argument("--n-batches", type=int, default=200,
                   help="Number of batches to fetch (default: 200)")
    return p.parse_args()


def _print_stats(label, times, total_paths):
    if not times:
        return
    avg_ms = sum(times) / len(times) * 1000
    total_s = sum(times)
    rate = total_paths / total_s if total_s > 0 else float("inf")
    print(f"  → {label}: avg {avg_ms:.1f}ms/batch  total {total_s:.2f}s  {rate:,.0f} paths/s")


def main():
    args = parse_args()
    pg_config = {
        "host": args.pg_host,
        "port": args.pg_port,
        "dbname": args.pg_dbname,
        "user": args.pg_user,
    }
    if args.pg_password:
        pg_config["password"] = args.pg_password

    db = DB(pg_config=pg_config)

    N = args.n_batches
    BS = args.batch_size
    log_every = max(1, N // 10)

    print(f"\n{'='*60}")
    print(f"Target: {N} batches x {BS:,} = {N * BS:,} paths total")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # fetch_batch
    # ------------------------------------------------------------------
    print(f"\n[fetch_batch]  batch_size={BS:,}  n_batches={N}")
    all_batches = []
    fetch_times = []

    for i in range(N):
        t0 = time.time()
        batch = db.fetch_batch(BS)
        elapsed = time.time() - t0

        if not batch:
            print(f"  No more state=0 paths at batch {i + 1}. Stopping early.")
            break

        all_batches.append(batch)
        fetch_times.append(elapsed)

        if (i + 1) % log_every == 0:
            rate = len(batch) / elapsed if elapsed > 0 else float("inf")
            print(f"  batch {i + 1:4d}/{N}: {elapsed * 1000:.1f}ms  {rate:,.0f} paths/s")

    total_fetched = sum(len(b) for b in all_batches)
    _print_stats("fetch_batch", fetch_times, total_fetched)

    if not all_batches:
        print("Nothing fetched. Ensure DB has state=0 paths. Exiting.")
        db.close()
        sys.exit(1)

    # Split: first half → mark_done, second half → mark_failed
    mid = max(1, len(all_batches) // 2)
    done_batches = all_batches[:mid]
    fail_batches = all_batches[mid:]

    # ------------------------------------------------------------------
    # mark_done
    # ------------------------------------------------------------------
    if done_batches:
        total = sum(len(b) for b in done_batches)
        print(f"\n[mark_done]  {total:,} paths in {len(done_batches)} calls")
        times = []
        for i, batch in enumerate(done_batches):
            t0 = time.time()
            db.mark_done(batch)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % max(1, len(done_batches) // 10) == 0:
                rate = len(batch) / elapsed if elapsed > 0 else float("inf")
                print(f"  batch {i + 1:4d}/{len(done_batches)}: {elapsed * 1000:.1f}ms  {rate:,.0f} paths/s")
        _print_stats("mark_done", times, total)

    # ------------------------------------------------------------------
    # mark_failed
    # ------------------------------------------------------------------
    if fail_batches:
        total = sum(len(b) for b in fail_batches)
        print(f"\n[mark_failed]  {total:,} paths in {len(fail_batches)} calls")
        times = []
        for i, batch in enumerate(fail_batches):
            t0 = time.time()
            db.mark_failed(batch)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % max(1, len(fail_batches) // 10) == 0:
                rate = len(batch) / elapsed if elapsed > 0 else float("inf")
                print(f"  batch {i + 1:4d}/{len(fail_batches)}: {elapsed * 1000:.1f}ms  {rate:,.0f} paths/s")
        _print_stats("mark_failed", times, total)

    print(f"\n{'='*60}")
    db.close()


if __name__ == "__main__":
    main()
