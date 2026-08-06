"""Result records and the aggregate report.

One JSON file per (config, style, sample, suite). Records keep per-element
detail, not just the headline number: a PAA of 0.7 is only actionable if the
file says *which* elements were mis-parented, so every metric writes its
evidence alongside its score.

Layout under the pipeline cache (beside the artifacts being scored):

    cache/<dataset>/evals/
      alignment/<config_name>/<sample>.json      style-independent
      <config_name>/<style>/<sample>/<suite>.json
      report.json / report.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def evals_root(cache_root: Path, dataset: str) -> Path:
    return Path(cache_root) / dataset / "evals"


def suite_path(root: Path, config_name: str, style: str, sample_id: str,
               suite: str) -> Path:
    return root / config_name / style / sample_id / f"{suite}.json"


def alignment_path(root: Path, config_name: str, sample_id: str) -> Path:
    return root / "alignment" / config_name / f"{sample_id}.json"


def write_record(path: Path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_record(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# -- aggregate report ------------------------------------------------------

# suite -> the headline numbers to lift into the report table.
HEADLINES: dict[str, list[str]] = {
    "stage1": ["csr", "component_accuracy", "rendering_fidelity", "code_quality"],
    "xml": ["paa", "edge_f1", "depth_violation_rate", "matched_coverage"],
    "sequence": ["coverage_precision", "coverage_recall", "tof", "dovr",
                 "sscr_pass"],
    "stage3": ["anim_csr", "anim_code_quality", "aif"],
}


def collect(root: Path) -> dict:
    """Every suite record under evals/, keyed (config, style, sample)."""
    out: dict[str, dict] = {}
    for path in sorted(Path(root).glob("*/*/*/*.json")):
        suite = path.stem
        sample = path.parent.name
        style = path.parent.parent.name
        config = path.parent.parent.parent.name
        if config == "alignment":
            continue
        record = read_record(path)
        if record is None:
            continue
        out.setdefault(config, {}).setdefault(style, {}) \
           .setdefault(sample, {})[suite] = record
    return out


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(root: Path) -> tuple[dict, str]:
    """Aggregate every record into (report dict, markdown table)."""
    data = collect(root)
    lines: list[str] = ["# AnimateBench evaluation report", ""]

    for config in sorted(data):
        lines.append(f"## {config}")
        for style in sorted(data[config]):
            lines.append(f"\n### {style}\n")
            samples = sorted(data[config][style])
            header = ["metric"] + samples
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            for suite, keys in HEADLINES.items():
                for key in keys:
                    row = [f"{suite}.{key}"]
                    present = False
                    for sample in samples:
                        record = data[config][style][sample].get(suite, {})
                        value = record.get(key)
                        present = present or value is not None
                        row.append(_fmt(value))
                    if present:
                        lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return data, "\n".join(lines)
