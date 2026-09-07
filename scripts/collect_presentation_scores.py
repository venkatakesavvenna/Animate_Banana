#!/usr/bin/env python3
"""Collect the talk video-metric records into per-metric JSON + a CSV.

Mirrors the ablation-score layout the user asked for previously:
    <out>/VFS_BAND/<sample>.json
    <out>/ASCS_VIDEO/<sample>.json
plus all_scores.csv over every cell.

`vfs_band_ordinal` is A->0 .. D->3, so SMALLER IS BETTER, while `vfs_band_pass`
is a boolean. A `sort_key` column is emitted that is always higher-is-better
(3 - ordinal), because sorting on the raw ordinal ranks BAND D as best -- a
silent wrong answer with nothing on screen to reveal it.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALS = REPO / "data/presentations_cache/presentations_ds/evals"
OUT = REPO / "data/presentation_scores"

METRICS = {"VFS_BAND": "vfs_band", "ASCS_VIDEO": "ascs_video"}
SHARED = ("suite", "style", "stages_run", "stages_skipped", "written_at", "provenance")


def main() -> int:
    records = sorted(EVALS.glob("*/*/*/animation.json"))
    if not records:
        print(f"no records under {EVALS}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0

    for rec in records:
        d = json.loads(rec.read_text(encoding="utf-8"))
        sample = rec.parent.name
        style = d.get("style") or ""
        row = {"sample": sample, "style": style}

        for label, m in METRICS.items():
            fields = {k: v for k, v in d.items() if k == m or k.startswith(m + "_")}
            if not fields:
                continue
            out = {"sample": sample, "style": style, "metric": label,
                   **{k: d.get(k) for k in SHARED if k in d}, **fields}
            (OUT / label).mkdir(parents=True, exist_ok=True)
            (OUT / label / f"{sample}.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8")
            n += 1

        band, ordinal = d.get("vfs_band"), d.get("vfs_band_ordinal")
        row["vfs_band"] = band
        row["vfs_band_ordinal"] = ordinal
        # ALWAYS higher-is-better, so a plain descending sort is correct.
        row["vfs_band_sort_key"] = (3 - ordinal) if isinstance(ordinal, int) else None
        row["vfs_band_pass"] = d.get("vfs_band_pass")
        row["ascs_video"] = d.get("ascs_video")
        row["ascs_video_pass"] = d.get("ascs_video_pass")
        row["frames_judged"] = d.get("vfs_band_frame_count")
        row["deck_source"] = d.get("vfs_band_source")
        row["measured"] = band is not None
        rows.append(row)

    rows.sort(key=lambda r: (r["style"], r["sample"]))
    with (OUT / "all_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {n} metric files + all_scores.csv ({len(rows)} cells) -> {OUT}")
    from collections import Counter
    print("  vfs_band:", dict(Counter(r["vfs_band"] for r in rows)))
    print("  ascs_video:", dict(Counter(r["ascs_video"] for r in rows)))
    unmeasured = [r["sample"] for r in rows if not r["measured"]]
    if unmeasured:
        print(f"  !! {len(unmeasured)} cells produced NO band: {unmeasured[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
