"""AnimateBench evaluation CLI.

Scores one pipeline config's artifacts against the reference bundle:

    run_eval all      --config configs/bench_gemini.yaml --style progressive_reveal
    run_eval xml      --config ... --only CVPR_2025_pipe00041
    run_eval sequence | stage1 | stage3
    run_eval report                       # aggregate every eval written so far

Suites split by what they need:

    stage1    diagram code            compile + judge
    xml       structure XML vs GT     needs the element alignment
    sequence  animation sequence      GT-free except coverage
    stage3    animation code          compile + diff + judge

The programmatic half runs with no API key at all; judge metrics report a
judge error and leave their fields null rather than failing the suite, so a
run without keys still produces every mechanical number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import ConfigError, load_config
from img_2_svg_pretraining.pipeline.samples import discover_samples
from img_2_svg_pretraining.pipeline.schema import AnimationSequence

from . import results
from .alignment import Alignment, align
from .gt import DiagramXml, load_ground_truth
from .judge import JudgeError, make_judge
from .metrics import stage1_code, stage2_sequence, stage2_xml, stage3_anim

SUITES = ("stage1", "xml", "sequence", "stage3")


def _paths_for(cfg, style: str) -> CachePaths:
    """Artifact paths for one style. The style is part of the sequence
    lineage, so it must be set before paths resolve."""
    cfg.style = style
    cfg.raw["animation_style"] = style
    return CachePaths.from_config(cfg)


def _get_alignment(cfg, paths: CachePaths, root: Path, config_name: str,
                   sample, judge, force: bool) -> Alignment | None:
    """The cached GT<->pred element alignment, judged once per sample.

    Style-independent: the structure XML does not change with animation style,
    so five styles share one alignment and one judge call.
    """
    path = results.alignment_path(root, config_name, sample.id)
    if path.exists() and not force:
        return Alignment.load(path)
    if judge is None:
        return None

    gt = load_ground_truth(cfg.dataset_root, sample.id)
    if not gt.xml_path().exists() or not paths.xml(sample.id).exists():
        return None

    try:
        alignment = align(judge, gt.xml(), DiagramXml.load(paths.xml(sample.id)),
                          sample.image_path)
    except JudgeError as e:
        print(f"    [{sample.id}] alignment failed: {e}", file=sys.stderr)
        return None

    alignment.save(path)
    matched = len(alignment.matched_gt())
    print(f"    [{sample.id}] aligned {matched} GT element(s), "
          f"{len(alignment.gt_unmatched)} unmatched")
    return alignment


def _run_stage1(cfg, paths, sample, judge, include_quality) -> dict:
    code_path = paths.resolve_code(sample.id)
    return stage1_code.run(judge, Path(code_path).read_text(encoding="utf-8"),
                           sample.image_path, paths.compile_cache(),
                           include_quality=include_quality)


def _run_xml(cfg, paths, sample, alignment) -> dict:
    pred_path = paths.xml(sample.id)
    if not pred_path.exists():
        return {"suite": "xml", "error": f"no predicted XML at {pred_path}"}
    gt = load_ground_truth(cfg.dataset_root, sample.id)
    pred = DiagramXml.load(pred_path)
    if alignment is None:
        # Depth consistency needs no GT, so report it rather than nothing.
        record = {"suite": "xml", "alignment_missing": True}
        record.update(stage2_xml.depth_consistency(pred))
        return record
    return stage2_xml.run(gt.xml(), pred, alignment)


def _run_sequence(cfg, paths, sample, alignment, style) -> dict:
    seq_path = paths.sequence_narrated(sample.id)
    if not seq_path.exists():
        seq_path = paths.sequence(sample.id)
    if not seq_path.exists():
        return {"suite": "sequence", "error": f"no sequence at {seq_path}"}
    xml_path = paths.xml(sample.id)
    if not xml_path.exists():
        return {"suite": "sequence", "error": "no predicted XML to check against"}

    gt = load_ground_truth(cfg.dataset_root, sample.id)
    gt_seq = gt.sequence(style) if gt.has_style(style) else None
    return stage2_sequence.run(gt_seq, AnimationSequence.load(seq_path),
                               DiagramXml.load(xml_path), alignment, style)


def _run_stage3(cfg, paths, sample, judge, include_quality) -> dict:
    anim_path = paths.animation_final(sample.id)
    if not anim_path.exists():
        anim_path = paths.animation(sample.id)
    if not anim_path.exists():
        return {"suite": "stage3", "error": f"no animation code at {anim_path}"}
    diagram = Path(paths.resolve_code(sample.id)).read_text(encoding="utf-8")
    work = paths.root / "evals" / "_compile_work" / sample.id
    return stage3_anim.run(judge, diagram, anim_path.read_text(encoding="utf-8"),
                           work, include_quality=include_quality)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=list(SUITES) + ["all", "report"])
    ap.add_argument("--config", help="pipeline config whose run is being scored")
    ap.add_argument("--style", default="progressive_reveal")
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute records (and the alignment) even if cached")
    ap.add_argument("--no-judge", action="store_true",
                    help="programmatic metrics only; no API calls")
    ap.add_argument("--include-quality", action="store_true",
                    help="also run the optional code-quality judgements")
    ap.add_argument("--judge-backend", default="gemini_flash")
    args = ap.parse_args()

    if args.suite == "report":
        # Report needs a config only to locate the cache root.
        if not args.config:
            ap.error("report needs --config to locate the cache")
        cfg = load_config(args.config)
        root = results.evals_root(cfg.cache_root, cfg.dataset_root.name)
        data, markdown = results.render_report(root)
        results.write_record(root / "report.json", {"configs": data})
        (root / "report.md").write_text(markdown, encoding="utf-8")
        print(markdown)
        print(f"\nwritten: {root / 'report.md'}")
        return

    if not args.config:
        ap.error("--config is required")
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        raise SystemExit(2)

    config_name = Path(args.config).stem
    paths = _paths_for(cfg, args.style)
    root = results.evals_root(cfg.cache_root, cfg.dataset_root.name)

    samples = discover_samples(cfg.dataset_root, limit=args.limit, only=args.only)
    if not samples:
        print(f"no samples under {cfg.dataset_root}", file=sys.stderr)
        raise SystemExit(1)

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    needs_judge = not args.no_judge and (
        {"stage1", "stage3"} & set(suites) or {"xml", "sequence"} & set(suites))

    judge = None
    if needs_judge:
        try:
            judge = make_judge(cfg, cache_root=paths.root,
                               raw_dir=root / "raw" / config_name,
                               backend_name=args.judge_backend)
            print(f"judge: {judge.model}")
        except Exception as e:
            print(f"judge unavailable ({e}); running programmatic metrics only",
                  file=sys.stderr)

    print(f"{len(samples)} sample(s) | style={args.style} | config={config_name}")

    try:
        for sample in samples:
            print(f"\n=== {sample.id}")
            alignment = None
            if {"xml", "sequence"} & set(suites):
                alignment = _get_alignment(cfg, paths, root, config_name, sample,
                                           judge, args.force)

            for suite in suites:
                out = results.suite_path(root, config_name, args.style, sample.id, suite)
                if out.exists() and not args.force:
                    print(f"  {suite}: cached")
                    continue
                try:
                    if suite == "stage1":
                        record = _run_stage1(cfg, paths, sample, judge,
                                             args.include_quality)
                    elif suite == "xml":
                        record = _run_xml(cfg, paths, sample, alignment)
                    elif suite == "sequence":
                        record = _run_sequence(cfg, paths, sample, alignment, args.style)
                    else:
                        record = _run_stage3(cfg, paths, sample, judge,
                                             args.include_quality)
                except Exception as e:
                    record = {"suite": suite, "error": f"{type(e).__name__}: {e}"}

                record["provenance"] = {
                    "config": config_name, "style": args.style,
                    "sample": sample.id, "config_fingerprint": cfg.fingerprint(),
                    **(judge.provenance() if judge else {}),
                }
                results.write_record(out, record)
                headline = {k: record.get(k) for k in results.HEADLINES.get(suite, [])
                            if record.get(k) is not None}
                note = record.get("error") or ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in headline.items())
                print(f"  {suite}: {note or 'written'}")
    finally:
        if judge is not None:
            judge.unload()

    print(f"\nrecords under {root}")


if __name__ == "__main__":
    main()
