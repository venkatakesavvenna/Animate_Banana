"""Probe the Gemini key pool and maintain a file of keys that still have quota.

WHY THIS EXISTS
---------------
The free tier's binding limit is not per minute, it is per DAY:

    quotaId:   GenerateRequestsPerDayPerProjectPerModel-FreeTier
    quotaValue: 20                    # requests/day/key for gemini-3.6-flash

So a key is not "slow", it is *finished* for the day, and no amount of backoff
brings it back. `KeyRing` is built for the per-minute case: it marks a key
exhausted, and when every key is marked it clears all the marks, because a
per-minute quota does reset. Against a per-day quota that turns into a loop --
75 keys, 56 of them spent, `max_retries` set to `len(ring) + 1` -- and a single
call spends minutes re-uploading the same figure to keys that cannot answer.
That is what a run that landed four calls in fifty minutes was doing.

Pointing PIPELINE_KEY_FILE at a pool that is *only* live keys fixes it without
touching KeyRing, whose behaviour is right for every other backend.

PROBING IS NOT FREE. Each probe spends one of that key's 20 daily requests, so
a full sweep of 75 keys costs 75 -- about a tenth of a day's total budget.
Refresh the live subset (cheap) and only fall back to the whole pool when the
subset has nearly run out.

    python scripts/gemini_keys.py probe            # full pool, writes status
    python scripts/gemini_keys.py refresh          # re-probe live subset only
    python scripts/gemini_keys.py refresh --min 5  # ...widening if too few left
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "api_keys.csv"
LIVE = REPO / "api_keys.live.csv"
STATUS = REPO / "logs" / "key_status.json"
MODEL = "gemini-3.6-flash"
TRANSIENT = (0, 500, 502, 503, 504)
ATTEMPTS = 3


def _probe_once(key: str, model: str) -> tuple[int, str]:
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = json.dumps({
        "contents": [{"parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 1, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, "ok"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode()).get("error", {})
            return e.code, f"{err.get('status', '')} {err.get('message', '')}"[:160]
        except Exception:
            return e.code, ""
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"[:160]


def probe(key: str, model: str = MODEL) -> tuple[str, int, str]:
    """A verdict that survived retries. A one-shot 503 filed as a dead key is
    a key that never gets checked again."""
    code, detail = 0, "not attempted"
    for attempt in range(ATTEMPTS):
        code, detail = _probe_once(key, model)
        if code not in TRANSIENT:
            break
        time.sleep(2 * (attempt + 1))
    return key, code, detail


def read_keys(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    field = list(rows[0])[0]
    service = next((f for f in rows[0] if f.strip().lower() == "service"), None)
    seen, keys = set(), []
    for row in rows:
        key = (row.get(field) or "").strip()
        if not key or key in seen:
            continue
        if service and (row.get(service) or "").strip().lower() not in ("google", "gemini"):
            continue
        seen.add(key)
        keys.append(key)
    return keys


def write_live(keys: list[str], path: Path = LIVE) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("Key, Service\n")
        for key in keys:
            fh.write(f"{key}, Google\n")


def _mask(key: str) -> str:
    """Enough to identify a key in a report; not enough to use it."""
    return f"{key[:10]}...{key[-6:]}" if len(key) > 20 else "***"


def sweep(keys: list[str], model: str) -> list[tuple[str, int, str]]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda k: probe(k, model), keys))


def report(results: list[tuple[str, int, str]]) -> list[str]:
    buckets: dict[str, list[str]] = {}
    for key, code, detail in results:
        label = {200: "OK", 429: "QUOTA (429, spent for today)"}.get(code) or str(code)
        buckets.setdefault(label, []).append(key)
    for label in sorted(buckets, key=lambda l: -len(buckets[l])):
        print(f"{len(buckets[label]):3d}  {label}")
    live = buckets.get("OK", [])
    print(f"\n{len(live)}/{len(results)} usable  "
          f"(~{len(live) * 20} requests left today at 20/key)")
    return live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("probe", "refresh"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min", type=int, default=4,
                    help="refresh: if fewer live keys than this survive, "
                         "re-probe the whole pool instead")
    args = ap.parse_args()

    if args.action == "probe":
        keys = read_keys(POOL)
        print(f"probing all {len(keys)} key(s) against {args.model} "
              f"-- costs {len(keys)} of today's requests\n", flush=True)
        results = sweep(keys, args.model)
    else:
        keys = read_keys(LIVE) or read_keys(POOL)
        print(f"refreshing {len(keys)} live key(s)\n", flush=True)
        results = sweep(keys, args.model)
        live = [k for k, c, _ in results if c == 200]
        if len(live) < args.min:
            widen = [k for k in read_keys(POOL) if k not in {r[0] for r in results}]
            print(f"only {len(live)} left; widening to {len(widen)} untested key(s)",
                  flush=True)
            results += sweep(widen, args.model)

    live = report(results)
    write_live(live)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    # Masked, never the raw key: this file lives under logs/, which is
    # tracked, and a full key here is a credential leak waiting for someone to
    # `git add -A`. api_keys.live.csv carries the real values and is the one
    # path in .gitignore's `*.csv` rule -- that is the only file allowed to
    # hold them.
    STATUS.write_text(json.dumps(
        {"model": args.model, "live": len(live), "total": len(results),
         "by_key": {_mask(k): c for k, c, _ in results}}, indent=2),
        encoding="utf-8")
    print(f"wrote {LIVE} and {STATUS}")
    if not live:
        print("\nNo key has quota left today. Runs will fail until reset.",
              file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
