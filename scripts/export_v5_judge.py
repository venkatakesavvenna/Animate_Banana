#!/usr/bin/env python3
"""Export the v5 SSS/GPS/NAS judge results in Medha's hand-off layout.

    <out>/<Model>/<METRIC>/<sample_id>.json

One file per (model, metric, sample), carrying every timestep's letter band
and the per-rule reasoning strings the FINAL prompts emit -- the point of the
layout is that she can read the reasoning of every step, reconstruct the
sequence, and compute her own aggregations (hamming distance etc.) without
our mean getting in the way. The cell-level mean is still included, labelled
as ours.

Letters are the native scale of the final prompts; the stored records carry
the numeric mapping (A=4, B=3, C=2, D=1). Both are exported so nothing has
to be re-derived. `is_valid: false` timesteps (unparseable judge reply) and
NAS's unnarrated steps appear explicitly rather than being silently absent,
so a gap in a sequence is always distinguishable from a short sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BENCHES = {
    "v5": {
        "models": {  # export dir name -> eval config dir name
            "Gemini_3.7_Flash": "bench_v5_or_gemini37flash",
            "Gemma_4-31B": "bench_v5_svg_gemma4_31b",
            "Qwen3.8-27B": "bench_v5_svg_qwen38_27b",
            "GLM-4.6V": "bench_v5_svg_glm46v",
            "Qwen3-VL-235B": "bench_v5_svg_qwen3vl235b",
        },
        "roots": (REPO / "data/animatebench_v5_cache/animatebench_v5/evals",
                  REPO / "data/animatebench_v5_or_cache/animatebench_v5/evals"),
    },
    "zs": {  # the 193-sample zero-shot set; no Gemini by request
        "models": {
            "Gemma_4-31B": "bench_zs_svg_gemma4_31b",
            "Qwen3.8-27B": "bench_zs_svg_qwen38_27b",
            "GLM-4.6V": "bench_zs_svg_glm46v",
            "Qwen3-VL-235B": "bench_zs_svg_qwen3vl235b",
        },
        "roots": (REPO / "data/animatebench_zs_cache/animatebench_zs/evals",),
    },
}
BAND_TO_LETTER = {4: "A", 3: "B", 2: "C", 1: "D", 0: "E"}

METRIC_RULE_KEYS = {
    "sss": ("rule_1_appropriateness_reasoning", "rule_2_coherence_reasoning"),
    "gps": ("rule_1_volume_reasoning", "rule_2_complexity_reasoning",
            "rule_3_relevance_reasoning"),
    "nas": ("rule_1_alignment_reasoning", "rule_2_narrative_context_reasoning",
            "rule_3_coherence_and_factuality_reasoning"),
}


def export_cell(record: dict, metric: str, sample: str, style: str,
                model: str) -> dict:
    steps = []
    for e in record.get(f"{metric}_step_detail") or []:
        band = e.get("final_band")
        step = {
            "timestep": e.get("timestep"),
            "frame": e.get("frame"),
            "final_score": BAND_TO_LETTER.get(band),
            "final_band": band,
            "is_valid": bool(e.get("is_valid")),
        }
        for k in METRIC_RULE_KEYS[metric]:
            step[k] = e.get(k)
        if metric == "nas":
            step["narration"] = e.get("narration")
        steps.append(step)
    out = {
        "sample": sample,
        "model": model,
        "style": style,
        "metric": metric.upper(),
        "band_scale": "A=4 B=3 C=2 D=1; our score = mean(final_band over valid steps) / 4",
        "score": record.get(metric),
        "band_mean": record.get(f"{metric}_band_mean"),
        "steps_total": record.get(f"{metric}_steps_total"),
        "steps_scored": record.get(f"{metric}_steps_scored"),
        "steps_invalid": record.get(f"{metric}_steps_invalid"),
        "errors": record.get(f"{metric}_errors") or [],
        "timesteps": steps,
    }
    if metric == "nas":
        out["steps_unnarrated"] = record.get("nas_steps_unnarrated")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", choices=sorted(BENCHES), default="v5")
    args = ap.parse_args()
    out_root = Path(args.out) if args.out else \
        REPO / f"data/judge_exports/{args.bench}_final_prompts"
    bench = BENCHES[args.bench]

    written, missing = 0, []
    for model, cfg in bench["models"].items():
        for root in bench["roots"]:
            base = root / cfg
            if not base.is_dir():
                continue
            for rec_path in sorted(base.glob("*/*/animation.json")):
                style, sample = rec_path.parent.parent.name, rec_path.parent.name
                record = json.loads(rec_path.read_text())
                for metric in ("sss", "gps", "nas"):
                    # NAS on an unnarrated cell has nothing to hand over, but an
                    # empty file that SAYS so beats a hole in her directory.
                    cell = export_cell(record, metric, sample, style, model)
                    dst = out_root / model / metric.upper() / f"{sample}.json"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(json.dumps(cell, indent=2, ensure_ascii=False))
                    written += 1
                    if cell["score"] is None and not cell["timesteps"]:
                        missing.append(f"{model}/{metric.upper()}/{sample}")
    print(f"wrote {written} files under {out_root}")
    if missing:
        print(f"{len(missing)} cell(s) have no judged timesteps (NAS-unnarrated "
              f"or skipped); their files carry the empty state explicitly:")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
