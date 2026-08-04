"""Run Stage 1a across every candidate model and report what each achieved.

The question this answers is "how many samples does each model get right?",
where "right" is deliberately mechanical: the generated document must compile.
That is the only Stage 1a property worth trusting without a human, and a model
that cannot produce compiling TikZ cannot drive the later stages at all.

Reported per model:
  loaded      the weights loaded on this GPU at all
  produced    a document was extracted from the response
  compiled    latexmk accepted it  <-- the headline number
  nodes       mean `xml id` count, since Stage 2 references those ids

Each model runs in its own subprocess so one OOM or bad architecture cannot
take the rest of the sweep with it, and so VRAM is fully returned between
models rather than relying on `del` + `empty_cache`.

    python -m img_2_svg_pretraining.pipeline.tests.eval_models --gpu 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Candidates, in rough ascending size. Only locally-cached repos are listed --
# a sweep that spends an hour downloading is a sweep nobody runs twice.
#
# `venv` and `attn` are per-model because the two environments differ in a way
# that matters:
#
#   img_2_svg_pretraining  NGC torch 2.7.0  flash_attn 2.7.3 WORKS
#   gemma4                 torch 2.11.0     flash_attn broken (ABI mismatch),
#                                           but has the transformers gemma4 needs
#
# So Qwen runs in the main venv with flash attention, and Gemma 4 runs in its
# own venv with `eager`. Not `sdpa` anywhere: it dispatches into cuDNN, which
# raises "No valid execution plans built" in this container -- observed on both
# Gemma 4 and Qwen3-VL, so it is the container's cuDNN, not one architecture.
MAIN_VENV = "/environments/img_2_svg_pretraining/bin/python"
GEMMA4_VENV = "/environments/gemma4/bin/python"

CANDIDATES = {
    "qwen2.5-vl-7b": {
        "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "attn": "flash_attention_2",
        "venv": MAIN_VENV,
    },
    "qwen3-vl-8b": {
        "hf_repo": "Qwen/Qwen3-VL-8B-Instruct",
        "attn": "flash_attention_2",
        "venv": MAIN_VENV,
    },
    "gemma4-12b": {
        "hf_repo": "google/gemma-4-12B-it",
        "attn": "eager",
        "venv": GEMMA4_VENV,
    },
    "qwen3.6-27b": {
        "hf_repo": "Qwen/Qwen3.6-27B",
        "attn": "flash_attention_2",
        "venv": MAIN_VENV,
    },
    "gemma4-31b": {
        "hf_repo": "google/gemma-4-31B-it",
        "attn": "eager",
        "venv": GEMMA4_VENV,
    },
}


def build_config(name: str, spec: dict, out: Path, dataset: str,
                 limit: int, batch_size: int) -> Path:
    """A single-model config for one sweep entry."""
    import yaml

    base = yaml.safe_load(
        (Path(__file__).parent.parent / "configs" / "hf_local.yaml").read_text())

    base["dataset"] = {"root": dataset, "limit": limit}
    base["cache_root"] = str(out / "cache")
    base["backends"] = {
        name: {
            "type": "hf_local",
            "hf_repo": spec["hf_repo"],
            "model": name,
            "model_class": spec.get("model_class", "image_text_to_text"),
            "attn_implementation": spec["attn"],
            "trust_remote_code": spec.get("trust_remote_code", False),
            "dtype": "bfloat16",
            "device": "cuda",
            "batch_size": batch_size,
        }
    }
    # Only Stage 1a is under test; point every agent at this model anyway so
    # the config stays valid.
    for section in ("transmuter", "planner", "animator"):
        for agent in base.get(section, {}).values():
            if isinstance(agent, dict) and "backend" in agent:
                agent["backend"] = name
    base["transmuter"]["raster_integrator"]["enabled"] = False

    path = out / f"cfg_{name}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def score(cache_root: Path, model: str) -> dict:
    """Compile every document this model produced."""
    from img_2_svg_pretraining.viewer.compile import compile_tikz

    code_dir = next((cache_root / "code").glob("*"), None) if (cache_root / "code").exists() else None
    if code_dir is None:
        return {"produced": 0, "compiled": 0, "nodes": 0.0, "samples": {}}

    samples, compiled, nodes = {}, 0, []
    for tex in sorted(code_dir.glob("*.tex")):
        code = tex.read_text(encoding="utf-8")
        result = compile_tikz(code, cache_root / "_compile")
        ok = bool(result.ok)
        compiled += ok
        n = code.count("xml id")
        nodes.append(n)
        first_error = next(
            (l.strip()[:110] for l in result.log.splitlines() if l.startswith("!")),
            "") if not ok else ""
        samples[tex.stem] = {"compiled": ok, "nodes": n, "error": first_error}

    return {
        "produced": len(samples),
        "compiled": compiled,
        "nodes": round(sum(nodes) / len(nodes), 1) if nodes else 0.0,
        "samples": samples,
    }


def run_one(name: str, spec: dict, args) -> dict:
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(name, spec, out, args.dataset, args.limit, args.batch_size)

    cmd = [spec.get("venv", args.python), "-m",
           "img_2_svg_pretraining.pipeline.run_pipeline",
           "convert-code", "--config", str(cfg), "--gpu", args.gpu, "--force"]

    print(f"\n=== {name} ({spec['hf_repo']}) ===", flush=True)
    started = time.time()
    log = out / "run.log"
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=args.cwd, stdout=fh,
                              stderr=subprocess.STDOUT, timeout=args.timeout)
    elapsed = time.time() - started

    tail = log.read_text(errors="replace")[-4000:]
    loaded = "Loading weights" in tail or "sample(s) via" in tail
    result = {"model": name, "repo": spec["hf_repo"], "exit": proc.returncode,
              "loaded": loaded, "seconds": round(elapsed)}
    result.update(score(out / "cache" / Path(args.dataset).name, name))

    # Keep the reason a model produced nothing, not just the zero.
    if result["produced"] == 0:
        fail = [l for l in tail.splitlines()
                if "Error" in l or "error" in l or l.startswith("  FAIL")]
        result["failure"] = fail[-1][:200] if fail else "no output, no error line"

    print(f"    compiled {result['compiled']}/{result['produced']} "
          f"in {result['seconds']}s", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="0")
    p.add_argument("--limit", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dataset", default="/code/data/test_benchmark")
    p.add_argument("--out", default="/tmp/model_eval")
    p.add_argument("--cwd", default="/code")
    p.add_argument("--python", default="/environments/gemma4/bin/python")
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--only", nargs="+", help="evaluate only these models")
    args = p.parse_args()

    todo = {k: v for k, v in CANDIDATES.items()
            if not args.only or k in args.only}

    rows = []
    for name, spec in todo.items():
        try:
            rows.append(run_one(name, spec, args))
        except subprocess.TimeoutExpired:
            rows.append({"model": name, "repo": spec["hf_repo"], "loaded": False,
                         "produced": 0, "compiled": 0, "nodes": 0.0,
                         "seconds": round(args.timeout),
                         "failure": f"timed out after {args.timeout:.0f}s"})
            print(f"    TIMEOUT", flush=True)

    report = Path(args.out) / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, indent=2))

    print(f"\n{'model':16} {'compiled':>10} {'produced':>9} {'nodes':>7} {'secs':>6}")
    print("-" * 54)
    for r in sorted(rows, key=lambda r: -r.get("compiled", 0)):
        print(f"{r['model']:16} {r.get('compiled', 0):>10} "
              f"{r.get('produced', 0):>9} {r.get('nodes', 0):>7} "
              f"{r.get('seconds', 0):>6}")
        if r.get("failure"):
            print(f"    ! {r['failure']}")
    print(f"\nfull report: {report}")


if __name__ == "__main__":
    main()
