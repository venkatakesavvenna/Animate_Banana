#!/usr/bin/env python3
"""SSS/GPS/NAS for the 193 (zero-shot) set, one folder per model.

    <out>/<Model>/{SSS,GPS,NAS}/<sample>.json
    <out>/all_scores.csv

Model folder names match the VIDEO folders already in gdrive:Zero_Shot_193, so
the scores sit beside the animations they describe.

A metric that is absent is reported as UNMEASURED, never as 0 -- `nas` in
particular is legitimately absent wherever a cell carries no narration, and a
missing value that renders as 0 sorts to the top of a "best" list.
"""
from __future__ import annotations
import csv, json, glob, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/zs_scores"
LABEL = {
    "bench_zs_or_gemini37flash": "AnimateBanana",
    "bench_zs_svg_gemma4_31b":   "Gemma-4-31B",
    "bench_zs_svg_glm46v":       "GLM-4.6V",
    "bench_zs_svg_qwen38_27b":   "Qwen3.8-27B",
    "bench_zs_svg_qwen3vl235b":  "Qwen3-VL-235B",
}
METRICS = ("sss", "gps", "nas")
SHARED = ("suite", "style", "stages_run", "stages_skipped", "written_at", "provenance")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for f in sorted(glob.glob(str(REPO / "data/animatebench_zs*cache/**/evals/*/*/*/animation.json"),
                              recursive=True)):
        parts = f.split("/evals/")[1].split("/")
        cfg = parts[0]
        label = LABEL.get(cfg)
        if label is None:
            continue
        sample = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        row = {"model": label, "sample": sample, "style": d.get("style") or ""}
        for m in METRICS:
            fields = {k: v for k, v in d.items() if k == m or k.startswith(m + "_")}
            if fields:
                doc = {"model": label, "sample": sample, "metric": m.upper(),
                       **{k: d.get(k) for k in SHARED if k in d}, **fields}
                (OUT / label / m.upper()).mkdir(parents=True, exist_ok=True)
                (OUT / label / m.upper() / f"{sample}.json").write_text(
                    json.dumps(doc, indent=2, default=str), encoding="utf-8")
                n += 1
            row[m] = d.get(m)
            row[m + "_measured"] = d.get(m) is not None
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["style"], r["sample"]))
    with (OUT / "all_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {n} metric files, {len(rows)} cells -> {OUT}\n")
    import statistics as st, collections
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in METRICS:
            if isinstance(r[m], (int, float)): per[r["model"]][m].append(r[m])
    print(f"{'model':18s} {'cells':>5} {'SSS':>16} {'GPS':>16} {'NAS':>16}")
    for mdl in sorted(per):
        c = sum(1 for r in rows if r["model"] == mdl)
        out = f"  {mdl:16s} {c:5d}"
        for m in METRICS:
            v = per[mdl][m]
            out += f"   {st.mean(v):.3f} (n={len(v):3d})" if v else f"   {'n/a':>15}"
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
