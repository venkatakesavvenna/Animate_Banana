"""Run the vfs_video (E1b) judge over cells that already have a frame-judged VFS.

Only cells with BOTH scores are useful, because the whole point is the
comparison -- so the cell list is derived from the existing records rather than
from the GT coverage matrix.

WHY A SEPARATE EVALS ROOT IS NOT OPTIONAL
-----------------------------------------
`run_eval.main()` sets `record["provenance"]` unconditionally from the judge it
was invoked with. Running this into the live `evals/` tree would relabel every
Qwen-produced vfs/ascs/sss/gps score in each touched record as
`judge_model: gemini-3.7-flash` -- silently, and it would destroy the audit
trail for numbers already published. `run_eval` would also skip cells whose
record exists unless `--force`, and `--force` without `--stages` re-runs the
whole ~70-call tree over the baseline.

Into a fresh root none of that applies: no --force is needed, records stand
alone, and rollback is `rm -rf`. `--evals-root` also redirects the backend
response cache, so cached video responses stay isolated too.

WAVES
-----
All 64 cells are affordable ($0 on free-tier keys), but ordering them makes the
study answerable early:

  1  the cells where the frame judge has variance (VFS < 1.0). 86% of cells sit
     at exactly 1.0, so these are the only ones a rank correlation can use.
  2  a stratified sample of the VFS==1.0 ceiling, balanced on ascs_pass. This is
     the discrimination test: does the video judge separate cells the frame
     judge tied? That, not the correlation, decides whether video VFS can
     absorb ASCS.
  3  the remainder, for the census.

Usage:
    python3 scripts/vfs_video_sweep.py --wave 1 [--dry-run]
    python3 scripts/vfs_video_sweep.py --wave 2 --limit 20
    python3 scripts/vfs_video_sweep.py --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTAINER = "animatebanana-v4"
PY = "/environments/img_2_svg_pretraining/bin/python"
LIVE = REPO / "data/animatebench_v3_cache/animatebench_v3/evals"
VROOT = "data/animatebench_v3_cache/animatebench_v3/evals_video_judge"
SKIP_DIRS = {"_stale_prompts_2026-08-19"}
CONFIG_OF = {"bench_v3_or": "bench_v3_or.yaml",
             "bench_v3_or_svg": "bench_v3_or_svg.yaml"}


def cells() -> list[dict]:
    out = []
    for path in sorted(LIVE.glob("*/*/*/animation.json")):
        config = path.parent.parent.parent.name
        if config in SKIP_DIRS or config not in CONFIG_OF:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("vfs") is None:
            continue
        out.append({"config": config, "style": path.parent.parent.name,
                    "sample": path.parent.name, "vfs": rec["vfs"],
                    "ascs_pass": rec.get("ascs_pass")})
    return out


def already_done(cell: dict) -> bool:
    p = (REPO / VROOT / cell["config"] / cell["style"] / cell["sample"]
         / "animation.json")
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # An error-only record must not block its own retry -- that has bitten this
    # repo repeatedly (arch01225, and 43 records after the key-limit failure).
    return rec.get("vfs_video") is not None


def wave(n: int, all_cells: list[dict], limit: int | None) -> list[dict]:
    if n == 1:
        picked = [c for c in all_cells if c["vfs"] < 0.999]
    elif n == 2:
        ceiling = [c for c in all_cells if c["vfs"] >= 0.999]
        by_bucket = defaultdict(list)
        for c in ceiling:
            by_bucket[(c["style"], bool(c["ascs_pass"]))].append(c)
        picked, i = [], 0
        # Round-robin across (style, ascs_pass) so the sample is balanced on
        # the variable the discrimination test is about, not just on style.
        while any(by_bucket.values()):
            for key in sorted(by_bucket, key=str):
                if by_bucket[key]:
                    picked.append(by_bucket[key].pop(0))
            i += 1
            if i > 100:
                break
    else:
        picked = list(all_cells)
    return picked[:limit] if limit else picked


def run(cell: dict) -> int:
    inner = (
        f"cd /code && PYTHONPATH=src PIPELINE_KEY_FILE=api_keys.csv {PY} -u "
        f"-m img_2_svg_pretraining.animatebench.run_eval animation "
        f"--config src/img_2_svg_pretraining/pipeline/configs/{CONFIG_OF[cell['config']]} "
        f"--style {cell['style']} --only {cell['sample']} "
        f"--stages vfs_video --judge-backend gemini_video --evals-root {VROOT}"
    )
    proc = subprocess.run(
        ["docker", "exec", "-u", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
         CONTAINER, "bash", "-lc", inner],
        capture_output=True, text=True)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3,
                    help="cells judged concurrently. Quota is not the binding "
                         "limit (75 free-tier keys x 20/day = 1500 against ~64 "
                         "calls); wall clock is, at ~135s per cell. Kept modest: "
                         "KeyRing's backoff is shaped for per-minute quotas, and "
                         "8 concurrent workers once landed zero calls in 30 min "
                         "by marking keys exhausted faster than the ring cleared.")
    args = ap.parse_args()

    all_cells = cells()
    picked = wave(0 if args.all else args.wave, all_cells, args.limit)
    todo = [c for c in picked if not already_done(c)]

    print(f"{len(all_cells)} scored cell(s) | wave selects {len(picked)} | "
          f"{len(todo)} not yet video-judged")
    for c in todo:
        print(f"  {c['config']:16} {c['style']:22} {c['sample'][:34]:34} "
              f"vfs={c['vfs']:.3f} ascs={c['ascs_pass']}")
    if args.dry_run or not todo:
        return

    started = time.time()
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    counter = {"n": 0}

    def work(c):
        code = run(c)
        ok = already_done(c)
        with lock:
            counter["n"] += 1
            print(f"[{counter['n']}/{len(todo)}] {c['config']}/{c['style']}/"
                  f"{c['sample']} exit={code} scored={ok} "
                  f"({time.time()-started:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(work, todo))
    print(f"=== finished {len(todo)} cell(s) in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
