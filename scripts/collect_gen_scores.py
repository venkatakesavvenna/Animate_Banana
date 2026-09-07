#!/usr/bin/env python3
"""Collect SSS/GPS/NAS for one (model, set) into per-metric JSON + a CSV.

Layout matches the ablation deliverable:
    data/gen_scores/<MODEL>_<SET>/{SSS,GPS,NAS}/<sample>.json
    data/gen_scores/<MODEL>_<SET>/all_scores.csv

A cell whose record exists but whose metric is absent is reported as
UNMEASURED rather than 0 -- a missing score and a genuine zero must never
render the same way, or sorting ascending puts every unmeasured cell at the
top of the "best" list.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = ("sss", "gps", "nas")
SHARED = ("suite", "style", "stages_run", "stages_skipped", "written_at", "provenance")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", dest="set_", required=True)
    a = ap.parse_args()

    cache = REPO / f"data/gen_cache/{a.model}_{a.set_}"
    records = sorted(cache.glob("*/evals/*/*/*/animation.json"))
    if not records:
        print(f"no records under {cache}", file=sys.stderr)
        return 1

    out = REPO / f"data/gen_scores/{a.model}_{a.set_}"
    out.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0

    for rec in records:
        d = json.loads(rec.read_text(encoding="utf-8"))
        sample = rec.parent.name
        row = {"sample": sample, "style": d.get("style") or "",
               "model": a.model, "set": a.set_}

        for m in METRICS:
            fields = {k: v for k, v in d.items() if k == m or k.startswith(m + "_")}
            label = m.upper()
            if fields:
                doc = {"sample": sample, "model": a.model, "set": a.set_,
                       "metric": label,
                       **{k: d.get(k) for k in SHARED if k in d}, **fields}
                (out / label).mkdir(parents=True, exist_ok=True)
                (out / label / f"{sample}.json").write_text(
                    json.dumps(doc, indent=2, default=str), encoding="utf-8")
                n += 1
            row[m] = d.get(m)
            row[f"{m}_measured"] = d.get(m) is not None
        row["skipped"] = json.dumps(d.get("stages_skipped") or {})
        rows.append(row)

    rows.sort(key=lambda r: (r["style"], r["sample"]))
    with (out / "all_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"{a.model}/{a.set_}: {n} metric files, {len(rows)} cells -> {out}")
    for m in METRICS:
        vals = [r[m] for r in rows if isinstance(r[m], (int, float))]
        miss = sum(1 for r in rows if not r[f"{m}_measured"])
        mean = f"{sum(vals)/len(vals):.4f}" if vals else "n/a"
        print(f"  {m.upper():4s} measured={len(vals):3d} unmeasured={miss:3d} mean={mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
