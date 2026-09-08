#!/usr/bin/env python3
"""Ablation SSS/GPS/NAS -> per-ablation, per-metric JSON + a combined CSV.

    data/ablation_scores/<NN_Label>/{SSS,GPS,NAS}/<sample>.json
    data/ablation_scores/all_scores.csv

Rebuilt from the eval records on every run rather than appended to, so a cell
re-judged after an error stub was purged overwrites its old score instead of
leaving a stale one behind.
"""
from __future__ import annotations
import csv, json, glob, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/ablation_scores"
LABEL = {
    "full": "00_AnimateBanana_Full",
    "stage1_no_critic": "01_Stage1_no_Critic",
    "stage2_no_image": "02_Stage2_no_Diagram_Image",
    "sequencer_no_xml": "03_Sequencer_no_XML",
    "narration_no_context": "04_Narration_no_Context",
    "stage2_no_critic": "05_Stage2_no_Critic",
    "designer_no_image": "06_Designer_no_Image",
    "stage3_no_critic": "07_Stage3_no_Critic",
}
METRICS = ("sss", "gps", "nas")
SHARED = ("style", "suite", "provenance", "written_at",
          "stages_requested", "stages_run", "stages_skipped")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for abl, label in sorted(LABEL.items(), key=lambda kv: kv[1]):
        for f in sorted(glob.glob(str(REPO / f"data/ablation_cache/{abl}/**/evals/*/*/*/animation.json"),
                                  recursive=True)):
            d = json.load(open(f))
            sample = os.path.basename(os.path.dirname(f))
            row = {"ablation": label, "ablation_config": abl,
                   "sample": sample, "style": d.get("style") or ""}
            for m in METRICS:
                fields = {k: v for k, v in d.items() if k == m or k.startswith(m + "_")}
                if fields:
                    doc = {"sample": sample, "ablation": label,
                           "ablation_config": abl, "metric": m.upper(),
                           **{k: d.get(k) for k in SHARED if k in d}, **fields}
                    (OUT / label / m.upper()).mkdir(parents=True, exist_ok=True)
                    (OUT / label / m.upper() / f"{sample}.json").write_text(
                        json.dumps(doc, indent=2, default=str), encoding="utf-8")
                    n += 1
                row[m] = d.get(m)
                row[m + "_measured"] = d.get(m) is not None
            rows.append(row)

    rows.sort(key=lambda r: (r["ablation"], r["style"], r["sample"]))
    with (OUT / "all_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {n} metric files, {len(rows)} cells -> {OUT}")
    import statistics as st, collections
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for m in METRICS:
            if isinstance(r[m], (int, float)): per[r["ablation"]][m].append(r[m])
    for a in sorted(per):
        c = sum(1 for r in rows if r["ablation"] == a)
        line = f"  {a:28s} n={c:3d}"
        for m in METRICS:
            v = per[a][m]
            line += f"  {m.upper()}={st.mean(v):.3f}({len(v)})" if v else f"  {m.upper()}=n/a"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
