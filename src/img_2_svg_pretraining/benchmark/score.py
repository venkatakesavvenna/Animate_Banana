"""Scoring pass: reads a model's inference results (benchmark/results/<model>.jsonl,
produced by run_benchmark.py) and adds DINO visual-similarity scores plus
VLM-as-judge scores, writing an augmented .jsonl (streamed record-by-record,
same rationale as run_benchmark.py -- avoids holding the whole results set
in memory) and a summary.

Run separately from inference so the model-under-test isn't sharing GPU
memory with the DINO model and the judge model:

    python -m img_2_svg_pretraining.benchmark.score --model qwen-3.5-9b \\
        --judge-repo Qwen/Qwen3.5-9B

Pass --skip-judge to only compute DINO scores (no judge model load needed).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_GPUS = "1,2,3,4,5,6,7"

RESULTS_DIR = Path(__file__).parent / "results"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model registry key whose results to score (must match an existing benchmark/results/<model>.jsonl)")
    parser.add_argument("--judge-repo", type=str, default=None, help="HF repo id of the judge model (required unless --skip-judge)")
    parser.add_argument("--judge-trust-remote-code", action="store_true")
    parser.add_argument("--skip-judge", action="store_true", help="only compute DINO scores, skip VLM-as-judge")
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS, help="CUDA_VISIBLE_DEVICES value (default excludes GPU 0)")
    args = parser.parse_args()
    if not args.skip_judge and not args.judge_repo:
        parser.error("--judge-repo is required unless --skip-judge is set")
    return args


def score_model(model_name: str, judge_repo: str | None, judge_trust_remote_code: bool, skip_judge: bool):
    from img_2_svg_pretraining.benchmark.metrics.dino import DinoScorer
    from img_2_svg_pretraining.benchmark.metrics.judge import JudgeRunner

    in_path = RESULTS_DIR / f"{model_name}.jsonl"
    lines = in_path.read_text().splitlines()
    total = len(lines)

    dino_scorer = DinoScorer()
    judge_runner = JudgeRunner(judge_repo, trust_remote_code=judge_trust_remote_code) if not skip_judge else None

    out_path = RESULTS_DIR / f"{model_name}.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.tmp")

    summary_records: list[dict] = []  # only the small fields summarize() needs, not full raw_output/tex
    try:
        with tmp_path.open("w") as out_f:
            for i, line in enumerate(lines):
                record = json.loads(line)
                if not record["compiled_ok"] or not record["rendered_png_path"]:
                    record["dino_score"] = None
                    record["judge_score"] = None
                    record["judge_rationale"] = "skipped: compile failed"
                else:
                    gt_path = Path(record["image_path"])
                    rendered_path = Path(record["rendered_png_path"])

                    record["dino_score"] = dino_scorer.score(gt_path, rendered_path)

                    if judge_runner is None:
                        record["judge_score"] = None
                        record["judge_rationale"] = "skipped: --skip-judge"
                    else:
                        result = judge_runner.judge(gt_path, rendered_path)
                        record["judge_score"] = result.score
                        record["judge_rationale"] = result.rationale

                    print(f"[{i+1}/{total}] {record['sample_id']}: "
                          f"dino={record['dino_score']:.3f} judge={record['judge_score']}")

                out_f.write(json.dumps(record) + "\n")
                summary_records.append({
                    "compiled_ok": record["compiled_ok"],
                    "dino_score": record.get("dino_score"),
                    "judge_score": record.get("judge_score"),
                })
    finally:
        if judge_runner is not None:
            judge_runner.unload()

    tmp_path.replace(out_path)

    summary = summarize(summary_records, model_name)
    summary_path = RESULTS_DIR / f"{model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} and {summary_path}")
    print(json.dumps(summary, indent=2))


def summarize(records: list[dict], model_name: str) -> dict:
    n = len(records)
    n_compiled = sum(r["compiled_ok"] for r in records)
    dino_scores = [r["dino_score"] for r in records if r.get("dino_score") is not None]
    judge_scores = [r["judge_score"] for r in records if r.get("judge_score") is not None]
    return {
        "model": model_name,
        "n_samples": n,
        "n_compiled_ok": n_compiled,
        "compile_rate": n_compiled / n if n else 0.0,
        "dino_mean": sum(dino_scores) / len(dino_scores) if dino_scores else None,
        "judge_mean": sum(judge_scores) / len(judge_scores) if judge_scores else None,
        "n_scored_dino": len(dino_scores),
        "n_scored_judge": len(judge_scores),
    }


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    score_model(args.model, args.judge_repo, args.judge_trust_remote_code, args.skip_judge)


if __name__ == "__main__":
    main()
