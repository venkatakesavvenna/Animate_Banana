"""Score a finished sweep: what did each model actually produce?

    sweep_score.py <sweep_dir> <dataset_name> <limit>

Compilation is the objective signal at both ends -- Stage 1a code and Stage 3
animation either compile or they don't -- and exported frames prove the whole
chain ran. Everything else is a count of artifacts on disk.
"""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    sweep = pathlib.Path(sys.argv[1])
    dataset = sys.argv[2]
    limit = sys.argv[3] if len(sys.argv) > 3 else "?"

    from img_2_svg_pretraining.viewer.compile import compile_tikz

    rows = []
    for d in sorted(p for p in sweep.iterdir() if p.is_dir()):
        cache = d / "cache" / dataset
        row = {"model": d.name, "code": 0, "code_ok": 0, "xml": 0, "seq": 0,
               "anim": 0, "anim_ok": 0, "frames": 0, "errors": []}

        def artifacts(kind: str, pattern: str) -> list[pathlib.Path]:
            base = cache / kind
            return sorted(base.glob(pattern)) if base.exists() else []

        for tex in artifacts("code", "*/*.tex"):
            row["code"] += 1
            result = compile_tikz(tex.read_text(errors="replace"), d / "_c")
            if result.ok:
                row["code_ok"] += 1
            elif len(row["errors"]) < 3:
                first = next((l.strip()[:90] for l in result.log.splitlines()
                              if l.startswith("!")), "")
                if first:
                    row["errors"].append(f"{tex.stem}: {first}")

        row["xml"] = len(artifacts("xml", "*/*.xml"))
        row["seq"] = (len(artifacts("sequence_final", "*/*.json"))
                      or len(artifacts("sequence", "*/*.json")))

        for tex in (artifacts("animation_final", "*/*.tex")
                    or artifacts("animation", "*/*.tex")):
            row["anim"] += 1
            if compile_tikz(tex.read_text(errors="replace"), d / "_a").ok:
                row["anim_ok"] += 1

        exports = cache / "exports"
        if exports.exists():
            row["frames"] = (len(list(exports.glob("*/*/frame_*.png")))
                             or len(list(exports.glob("*/*/*.mp4"))))
        rows.append(row)

    print(f"MODEL SWEEP RESULTS   (of {limit} samples each)\n")
    header = (f"{'model':16}{'1a code':>9}{'compiles':>10}{'xml':>6}"
              f"{'seq':>6}{'anim':>6}{'anim ok':>9}{'frames':>8}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: -x["code_ok"]):
        print(f"{r['model']:16}{r['code']:>9}{r['code_ok']:>10}{r['xml']:>6}"
              f"{r['seq']:>6}{r['anim']:>6}{r['anim_ok']:>9}{r['frames']:>8}")

    print("\ncolumns: 1a code = documents produced | compiles = latexmk accepted")
    print("         xml/seq = stage 2 artifacts   | anim ok  = animation compiles")
    print("         frames  = exported frames (end-to-end proof)\n")

    for r in rows:
        if r["errors"]:
            print(f"{r['model']} first compile errors:")
            for e in r["errors"]:
                print("   ", e)

    (sweep / "report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
