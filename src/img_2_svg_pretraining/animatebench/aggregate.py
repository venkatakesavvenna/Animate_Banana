"""Roll per-cell eval records up into a per-model summary.

WHAT WAS MISSING
----------------
Everything in this package reports one cell at a time. `results.render_report`
emits rows=metrics x columns=SAMPLES; the viewer's scoreboard emits one row per
sample. Neither answers "how did model A do against model B", which is the
whole question a multi-model bench exists to ask. `scripts/compare_judges.py`
was the only code here that took a mean, and it covered one suite and printed
fixed-width text.

FOUR KINDS OF ABSENCE, KEPT APART
---------------------------------
Averaging is the easy part. What makes an aggregate wrong is quietly folding
"we did not measure this" into "this scored zero", and there are four distinct
ways a number can be missing here:

1. **Not measured.** The key is absent or null -- an optional judged metric that
   was not requested, or a GT-derived one with no reference. Also how a failed
   suite arrives: a record that errored holds only `{error, provenance, suite,
   written_at}` and no metric keys at all.
2. **Not applicable.** `repetition_rate` is only defined for three styles
   (`animation_quality.REPETITION_STYLES`); elsewhere the record says
   `repetition_status == "not_specified"`. On a bench of progressive_reveal and
   colour_pop that is nearly every cell, so counting those as 0.0 would
   manufacture a perfect score out of a metric that was never computed.
3. **A gate, not a score.** `rendering_fidelity` is forced to exactly 0.0 when
   the diagram does not compile -- verified against the records, where every
   csr=0.0 cell carries fidelity=0.0 while its `component_accuracy` stays as
   high as 0.95. Averaged flat, "did not compile" is indistinguishable from
   "compiled and looked wrong". Both readings are legitimate, so both are
   reported: `rendering_fidelity` over every cell, and
   `rendering_fidelity (compiled)` over the cells that compiled.
4. **Never attempted.** The cell produced no record at all. This is the one that
   inverts a comparison if ignored: a model that crashed on five samples is
   averaged over its survivors, so failing more makes it look better. Hence
   `cells` beside every `n` -- how many cells *should* exist, taken from the
   dataset's ground-truth coverage rather than from what happened to be scored.

MICRO, NOT MACRO. Styles are pooled by raw per-cell value. Set5 splits 16/2/2,
so averaging per-style means would hand a 2-sample style the same weight as a
16-sample one.

    python -m img_2_svg_pretraining.animatebench.aggregate \\
      --config src/.../configs/bench_v4_svg_qwen3vl32b.yaml \\
      --dataset data/animatebench_v4 --format md csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import statistics
import sys
from pathlib import Path

from . import descriptions, results

# Metrics whose per-cell value is a 0/1 indicator. Their mean IS a pass rate, so
# it is labelled as one and no median is shown -- the median of a 0/1 column is
# 0 or 1 and reads like a score.
#
# `csr` and `anim_csr` are floats in the records (0.0/1.0), not bools, so this
# cannot be inferred from the Python type; `sscr_pass` and `ascs_pass` are real
# bools. Listing them explicitly and then CHECKING the assumption at runtime is
# the only way that stays honest if a metric's type changes underneath.
RATE_METRICS = {"csr", "anim_csr", "sscr_pass", "ascs_pass", "compiles"}

# Unbounded counts. One catastrophic cell drags a mean somewhere unrepresentative,
# so the median leads and the mean is reported beside it.
COUNT_METRICS = {"arrow_omission_count", "element_omission_count"}

# metric -> (status field, value that means "not applicable here")
NOT_APPLICABLE = {"repetition_rate": ("repetition_status", "not_specified")}

# metric -> (gate field, extra row name). When the gate is falsy the metric was
# forced to a floor rather than measured, so a second row reports it over the
# cells where it was a real judgement.
GATED = {"rendering_fidelity": ("csr", "rendering_fidelity (compiled)")}


def _is_indicator(value) -> bool:
    return isinstance(value, bool) or value in (0, 1, 0.0, 1.0)


def _values(records: dict, suite: str, metric: str, gated_only: bool = False):
    """Per-sample values for one metric, with every absence rule applied.

    Returns (values, skipped) where `skipped` counts cells excluded for a
    reason other than "no record" -- i.e. not-applicable or gated-out.
    """
    values, skipped = [], 0
    for record in records.values():
        rec = record.get(suite)
        if not rec:
            continue                                  # suite never ran here
        value = rec.get(metric)
        if value is None:
            continue                                  # not measured (or errored)

        na = NOT_APPLICABLE.get(metric)
        if na and rec.get(na[0]) == na[1]:
            skipped += 1
            continue

        if gated_only:
            gate = GATED[metric][0]
            if not rec.get(gate):
                skipped += 1
                continue

        values.append(value)
    return values, skipped


def _summarise_metric(metric: str, values: list) -> dict:
    """mean/median/rate for one metric's values, shaped by its class."""
    if not values:
        return {"n": 0}
    numeric = [float(v) for v in values]
    out = {"n": len(numeric), "mean": statistics.fmean(numeric)}

    if metric in RATE_METRICS:
        # Verify rather than trust: a metric that stopped being an indicator
        # would otherwise be silently reported as a percentage.
        if all(_is_indicator(v) for v in values):
            out["kind"] = "rate"
            out["rate"] = out["mean"]
            return out
        out["note"] = "declared a rate but holds non-indicator values"

    out["median"] = statistics.median(numeric)
    out["kind"] = "count" if metric in COUNT_METRICS else "score"
    return out


def expected_cells(dataset_root: Path | None, style: str, target: str = "svg") -> int | None:
    """How many cells this (style, target) SHOULD have, from GT coverage.

    The denominator that keeps a crashed run from outscoring a complete one.
    None when no dataset was given, in which case callers fall back to the
    empirical union across models -- still a real denominator, just a weaker one.
    """
    if not dataset_root:
        return None
    root = Path(dataset_root)
    if not root.is_dir():
        return None
    return sum(
        1 for sample in root.iterdir()
        if (sample / "reference" / "seq" /
            f"{style}_{sample.name}_{target}.json").exists()
    )


def summarise(root: Path, dataset_root: Path | None = None,
              configs: list[str] | None = None,
              styles: list[str] | None = None,
              target: str = "svg") -> dict:
    """Aggregate every record under `root` into a nested summary."""
    data = results.collect(root, configs=configs)

    # Empirical denominator: every sample any model produced for this style.
    # Used only when the dataset's own coverage is unavailable.
    seen: dict[str, set] = {}
    for cfg in data:
        for style, samples in data[cfg].items():
            seen.setdefault(style, set()).update(samples)

    out: dict = {"configs": {}, "target": target}
    for cfg in sorted(data):
        cfg_styles = sorted(s for s in data[cfg] if not styles or s in styles)
        entry: dict = {"styles": {}, "all": {}}

        # Each config has its own target, and GT coverage is per target -- the
        # SVG and TikZ configs of the same bench do NOT have the same cells.
        # The records say which target they were scored for, so prefer that
        # over the CLI's single global value.
        cfg_target = next(
            (rec.get("target") for samples in data[cfg].values()
             for suites in samples.values() for rec in suites.values()
             if rec.get("target")), target)

        pooled: dict[str, dict[str, list]] = {}
        for style in cfg_styles:
            records = data[cfg][style]
            expected = expected_cells(dataset_root, style, cfg_target)
            block = {
                "scored": len(records),
                "cells": expected if expected is not None else len(seen.get(style, ())),
                "cells_source": "ground truth" if expected is not None else "observed",
                "errors": sum(1 for r in records.values()
                              for s in r.values() if "error" in s),
                "metrics": {},
            }
            for suite, _label in descriptions.TABLE_SUITES:
                for metric in descriptions.ordered(suite):
                    values, skipped = _values(records, suite, metric)
                    stat = _summarise_metric(metric, values)
                    if skipped:
                        stat["not_applicable"] = skipped
                    block["metrics"][f"{suite}.{metric}"] = stat
                    pooled.setdefault(f"{suite}.{metric}", {"v": [], "skip": 0})
                    pooled[f"{suite}.{metric}"]["v"] += values
                    pooled[f"{suite}.{metric}"]["skip"] += skipped

                    if metric in GATED:
                        gv, gskip = _values(records, suite, metric, gated_only=True)
                        name = f"{suite}.{GATED[metric][1]}"
                        gstat = _summarise_metric(metric, gv)
                        if gskip:
                            gstat["not_applicable"] = gskip
                        block["metrics"][name] = gstat
                        pooled.setdefault(name, {"v": [], "skip": 0})
                        pooled[name]["v"] += gv
                        pooled[name]["skip"] += gskip
            entry["styles"][style] = block

        entry["all"] = {
            "scored": sum(b["scored"] for b in entry["styles"].values()),
            "cells": sum(b["cells"] for b in entry["styles"].values()),
            "errors": sum(b["errors"] for b in entry["styles"].values()),
            "metrics": {},
        }
        for name, acc in pooled.items():
            metric = name.split(".", 1)[1].split(" ")[0]
            stat = _summarise_metric(metric, acc["v"])
            if acc["skip"]:
                stat["not_applicable"] = acc["skip"]
            entry["all"]["metrics"][name] = stat
        entry["target"] = cfg_target
        out["configs"][cfg] = entry
    return out


# -- rendering -------------------------------------------------------------

def _cell(stat: dict) -> str:
    if not stat or not stat.get("n"):
        na = (stat or {}).get("not_applicable")
        return f"— ({na} n/a)" if na else "—"
    if stat.get("kind") == "rate":
        body = f"{stat['rate']:.0%}"
    elif stat.get("kind") == "count":
        body = f"{stat['median']:.1f} / {stat['mean']:.1f}"
    else:
        body = f"{stat['mean']:.3f}"
    body += f" ({stat['n']})"
    if stat.get("not_applicable"):
        body += f" [{stat['not_applicable']} n/a]"
    return body


# `describe()["better"]` has three values, not two: "higher", "lower", and
# "pass" for the two indicator metrics. Folding "pass" into the else branch
# renders a pass rate with a down-arrow, i.e. states the opposite of the truth.
_ARROW = {"higher": "↑", "lower": "↓", "pass": "↑"}


def _label(name: str) -> str:
    suite, metric = name.split(".", 1)
    base = metric.split(" ")[0]
    info = descriptions.describe(base)
    arrow = _ARROW.get(info.get("better"), "?")
    label = info.get("label", base)
    if metric != base:                       # the "(compiled)" variant
        label += " " + metric[len(base):].strip()
    kind = ""
    if base in RATE_METRICS:
        kind = " ·rate"
    elif base in COUNT_METRICS:
        kind = " ·med/mean"
    return f"{label} {arrow}{kind}"


def to_markdown(summary: dict, scope: str = "all") -> str:
    """Rows = metrics, columns = models -- the transpose of `render_report`."""
    cfgs = sorted(summary["configs"])
    if not cfgs:
        return "# AnimateBench aggregate\n\n_No records found._\n"

    lines = ["# AnimateBench aggregate", ""]
    lines.append(f"Target **{summary['target']}**, {len(cfgs)} model(s). "
                 "Cells show the statistic with `(n)` — how many cells it is "
                 "over. Styles are pooled by raw per-cell value (micro), so a "
                 "2-sample style does not weigh as much as a 16-sample one.")
    lines.append("")

    def table(getter, title):
        lines.append(f"## {title}\n")
        head = ["metric"] + cfgs
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        cov = ["**cells scored**"]
        for cfg in cfgs:
            blk = getter(cfg)
            cov.append(f"{blk['scored']}/{blk['cells']}" if blk else "—")
        lines.append("| " + " | ".join(cov) + " |")

        for suite, group in descriptions.TABLE_SUITES:
            names = [f"{suite}.{m}" for m in descriptions.ordered(suite)]
            names += [f"{suite}.{GATED[m][1]}" for m in descriptions.ordered(suite)
                      if m in GATED]
            rows = []
            for name in names:
                cells = [_cell((getter(cfg) or {}).get("metrics", {}).get(name))
                         for cfg in cfgs]
                if any(c != "—" for c in cells):
                    rows.append("| " + " | ".join([_label(name)] + cells) + " |")
            if rows:
                lines.append(f"| **{group}** |" + " |" * len(cfgs))
                lines.extend(rows)
        lines.append("")

    if scope in ("all", "both"):
        table(lambda c: summary["configs"][c]["all"], "All styles (pooled)")
    if scope in ("style", "both"):
        every = sorted({s for c in cfgs for s in summary["configs"][c]["styles"]})
        for style in every:
            table(lambda c, s=style: summary["configs"][c]["styles"].get(s), style)
    return "\n".join(lines)


def to_csv(summary: dict, out_path: Path) -> Path:
    """Tidy/long: one row per (config, style, metric, stat)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.writer(fh)
        w.writerow(["config", "style", "suite", "metric", "kind",
                    "mean", "median", "rate", "n", "cells", "not_applicable"])
        for cfg, entry in sorted(summary["configs"].items()):
            blocks = list(entry["styles"].items()) + [("__all__", entry["all"])]
            for style, block in blocks:
                for name, stat in block["metrics"].items():
                    suite, metric = name.split(".", 1)
                    w.writerow([cfg, style, suite, metric, stat.get("kind", ""),
                                stat.get("mean", ""), stat.get("median", ""),
                                stat.get("rate", ""), stat.get("n", 0),
                                block.get("cells", ""),
                                stat.get("not_applicable", 0)])
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals-root", type=Path)
    ap.add_argument("--config", type=Path,
                    help="pipeline config, to locate the evals root and target")
    ap.add_argument("--dataset", type=Path,
                    help="dataset root, for the ground-truth cell denominator")
    ap.add_argument("--configs", nargs="+", help="restrict to these config stems")
    ap.add_argument("--styles", nargs="+")
    ap.add_argument("--target", default=None, choices=("svg", "tikz"))
    ap.add_argument("--scope", default="both", choices=("all", "style", "both"))
    ap.add_argument("--format", nargs="+", default=["md"],
                    choices=("md", "csv", "json"))
    ap.add_argument("--out", type=Path, help="output stem (default: <evals>/aggregate)")
    args = ap.parse_args()

    root, target, dataset = args.evals_root, args.target, args.dataset
    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from img_2_svg_pretraining.pipeline.config import load_config
        cfg = load_config(str(args.config))
        root = root or results.evals_root(Path(cfg.cache_root),
                                          Path(cfg.dataset_root).name)
        target = target or cfg.target
        dataset = dataset or Path(cfg.dataset_root)
    if not root:
        raise SystemExit("need --evals-root or --config")

    summary = summarise(root, dataset_root=dataset, configs=args.configs,
                        styles=args.styles, target=target or "svg")
    stem = args.out or Path(root) / "aggregate"

    if "md" in args.format:
        md = to_markdown(summary, scope=args.scope)
        Path(f"{stem}.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"\n-> {stem}.md")
    if "csv" in args.format:
        print(f"-> {to_csv(summary, Path(f'{stem}.csv'))}")
    if "json" in args.format:
        Path(f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"-> {stem}.json")


if __name__ == "__main__":
    main()
