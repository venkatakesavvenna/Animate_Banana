#!/usr/bin/env python3
"""
Benchmark fetch_batch, mark_done, mark_failed against a local SQLite DB.

The DB file lives inside the examples/db_performance_testing/ folder so that
any node with access to the shared network path can run this script against
the same database without any server setup.

Pre-condition: DB must have at least (--n-batches * --batch-size) paths in state=0.
Run ingest first:

    python -m vision_ingest.drivers.db_driver_sql \
        --mode ingest \
        --db-path ./bench_sql_data \
        --folder image_paths.txt \
        --logs-path ./logs_sql \
        --batch-size 32000
"""
import argparse
import os
import sys
import time

from vision_ingest.db.db import DB

# Default DB location — inside the examples folder so any node with the
# shared NFS/Lustre mount can reach it via the same absolute or relative path.
DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), "bench_sql_data")


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark SQLite DB operations (fetch_batch / mark_done / mark_failed)"
    )
    p.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help=(
            "Directory that contains images.db and seen_cache.db "
            f"(default: {DEFAULT_DB_DIR})"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Paths per fetch_batch call (default: 1000)",
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=200,
        help="Number of batches to fetch (default: 200)",
    )
    return p.parse_args()


def _print_stats(label: str, times: list, total_paths: int) -> None:
    if not times:
        return
    avg_ms = sum(times) / len(times) * 1000
    total_s = sum(times)
    rate = total_paths / total_s if total_s > 0 else float("inf")
    print(f"  → {label}: avg {avg_ms:.1f}ms/batch  total {total_s:.2f}s  {rate:,.0f} paths/s")


def main():
    args = parse_args()

    db_file = os.path.join(args.db_dir, "images.db")
    seen_file = os.path.join(args.db_dir, "seen_cache.db")

    if not os.path.exists(db_file):
        print(
            f"❌  Database not found at: {db_file}\n"
            "    Run the ingest step first — see README_sql.md for instructions."
        )
        sys.exit(1)

    print(f"Opening DB:         {db_file}")
    print(f"Opening seen-cache: {seen_file}")

    # verify=False — skip count-drift check so the benchmark starts immediately.
    db = DB(path=db_file, seen_path=seen_file, verify=False)

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
    all_batches: list[list[str]] = []
    fetch_times: list[float] = []

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
        print(
            "Nothing fetched. Ensure DB has state=0 paths.\n"
            "Re-run ingest or reset stuck/failed paths via db_driver_sql."
        )
        db.close()
        sys.exit(1)

    # Split: first half → mark_done, second half → mark_failed
    # (mirrors bench_ops.py behaviour exactly)
    mid = max(1, len(all_batches) // 2)
    done_batches = all_batches[:mid]
    fail_batches = all_batches[mid:]

    # ------------------------------------------------------------------
    # mark_done
    # ------------------------------------------------------------------
    if done_batches:
        total = sum(len(b) for b in done_batches)
        print(f"\n[mark_done]  {total:,} paths in {len(done_batches)} calls")
        times: list[float] = []
        log_every_inner = max(1, len(done_batches) // 10)
        for i, batch in enumerate(done_batches):
            t0 = time.time()
            db.mark_done(batch)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % log_every_inner == 0:
                rate = len(batch) / elapsed if elapsed > 0 else float("inf")
                print(
                    f"  batch {i + 1:4d}/{len(done_batches)}: "
                    f"{elapsed * 1000:.1f}ms  {rate:,.0f} paths/s"
                )
        _print_stats("mark_done", times, total)

    # ------------------------------------------------------------------
    # mark_failed
    # ------------------------------------------------------------------
    if fail_batches:
        total = sum(len(b) for b in fail_batches)
        print(f"\n[mark_failed]  {total:,} paths in {len(fail_batches)} calls")
        times = []
        log_every_inner = max(1, len(fail_batches) // 10)
        for i, batch in enumerate(fail_batches):
            t0 = time.time()
            db.mark_failed(batch)
            elapsed = time.time() - t0
            times.append(elapsed)
            if (i + 1) % log_every_inner == 0:
                rate = len(batch) / elapsed if elapsed > 0 else float("inf")
                print(
                    f"  batch {i + 1:4d}/{len(fail_batches)}: "
                    f"{elapsed * 1000:.1f}ms  {rate:,.0f} paths/s"
                )
        _print_stats("mark_failed", times, total)

    # ------------------------------------------------------------------
    # Final DB health snapshot
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    health = db.get_db_health()
    print("📊 Final DB health:")
    print(f"   Main DB size : {health['main_db_size_mb']:.2f} MB")
    print(f"   Total images : {health['total_images_main']:,}")
    sd = health.get("state_distribution", {})
    for key, val in sd.items():
        label_map = {
            "state_0": "pending",
            "state_1": "in-progress",
            "state_2": "done",
            "state_3": "failed",
            "state_4": "upstream-ready",
        }
        label = label_map.get(key, key)
        print(f"   {key} ({label}): {val['count']:,}")
    print(f"{'='*60}")

    db.close()


if __name__ == "__main__":
    main()