"""AnimateBanana pipeline CLI.

Three stages, each a set of agents:

  STAGE 1  DIAGRAM TRANSMUTER
    convert-code       image -> animation-aware diagram code
    integrate-rasters  locate raster regions with the VLM -> crop into the code

  STAGE 2  ANIMATION PLANNER
    strategize         traversal strategy (OVERVIEW_FIRST | DETAIL_FIRST)
    parse              diagram structure/hierarchy XML
    sequence           animation sequence, conditioned on the style
    critique-sequence  review and repair the sequence
    narrate            write the spoken narration into the sequence

  STAGE 3  DIAGRAM ANIMATOR
    design             animation code from the sequence
    critique-animation detect errors and repair
    export             PDF / MP4 / GIF / PPTX

Each agent runs over the whole sample set and hands off to the next purely
through on-disk artifacts, so only the model a given agent needs is ever
loaded, and any agent can be re-run alone after a prompt change.

Any run can be narrowed to a window of stages with `--from`/`--to`, which is
how a partially-complete run is resumed without recomputing what already
succeeded. Both bounds are inclusive.

Examples:
    run_pipeline.py convert-code --config configs/default.yaml --limit 1
    run_pipeline.py stage2 --config configs/default.yaml
    run_pipeline.py all --config configs/default.yaml --only CVPR_2025_pipe00002
    run_pipeline.py all --from sequence            # resume: sequence -> export
    run_pipeline.py all --from parse --to design   # just the middle
    run_pipeline.py paths --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"

# subcommand -> (module path, agent label)
STAGE_MODULES = {
    "convert-code": ("transmuter.code_converter", "code_converter"),
    "integrate-rasters": ("transmuter.raster_integrator", "raster_integrator"),
    "strategize": ("planner.strategizer", "strategizer"),
    "parse": ("planner.parser", "parser"),
    "sequence": ("planner.sequencer", "sequencer"),
    "critique-sequence": ("planner.critic", "planner_critic"),
    "narrate": ("planner.narrative_writer", "narrative_writer"),
    "design": ("animator.designer", "designer"),
    "critique-animation": ("animator.critic", "animator_critic"),
    "export": ("animator.exporter", "exporter"),
}

STAGE_GROUPS = {
    "stage1": ["convert-code", "integrate-rasters"],
    "stage2": ["strategize", "parse", "sequence", "critique-sequence", "narrate"],
    "stage3": ["design", "critique-animation", "export"],
}
STAGE_GROUPS["all"] = STAGE_GROUPS["stage1"] + STAGE_GROUPS["stage2"] + STAGE_GROUPS["stage3"]

# Canonical agent order, which is what `--from`/`--to` slice against. The
# groups above are contiguous windows of this list, so a range never has to be
# reconciled against them separately.
STAGE_ORDER = STAGE_GROUPS["all"]


def resolve_stages(stage: str, start: str | None = None, stop: str | None = None) -> list[str]:
    """The agents to run, honouring `--from`/`--to`.

    Without either flag this is just the named subcommand or group. With them
    the selection is narrowed to a window of `STAGE_ORDER`, so `all --from
    sequence` resumes a partially-complete run and `all --to parse` stops
    before the expensive half. Both bounds are inclusive.

    Narrowing applies to whatever was named, rather than always slicing the
    full pipeline: `stage2 --from sequence` stays inside stage 2.
    """
    stages = STAGE_GROUPS.get(stage, [stage])

    for flag, name in (("--from", start), ("--to", stop)):
        if name is not None and name not in STAGE_ORDER:
            raise SystemExit(
                f"{flag}: unknown stage '{name}'. Known: {', '.join(STAGE_ORDER)}")

    if start is not None:
        begin = STAGE_ORDER.index(start)
        stages = [s for s in stages if STAGE_ORDER.index(s) >= begin]
    if stop is not None:
        end = STAGE_ORDER.index(stop)
        stages = [s for s in stages if STAGE_ORDER.index(s) <= end]

    if not stages:
        # Silently running nothing looks like success; say why the window is
        # empty, since the usual cause is the bounds being the wrong way round.
        raise SystemExit(
            f"no stages selected: '{stage}' contains nothing between "
            f"'{start or STAGE_ORDER[0]}' and '{stop or STAGE_ORDER[-1]}'")
    return stages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--config", default=str(DEFAULT_CONFIG),
                       help="pipeline YAML config (default: %(default)s)")
        p.add_argument("--limit", type=int, default=None,
                       help="process at most N samples (overrides dataset.limit)")
        p.add_argument("--only", nargs="+", metavar="ID",
                       help="process only these sample ids")
        p.add_argument("--force", action="store_true",
                       help="recompute even when the artifact is cached")
        p.add_argument("--gpu", default=None,
                       help="CUDA_VISIBLE_DEVICES for local backends")
        p.add_argument("--from", dest="from_stage", default=None, metavar="STAGE",
                       help="start at this stage, skipping earlier ones (inclusive)")
        p.add_argument("--to", dest="to_stage", default=None, metavar="STAGE",
                       help="stop after this stage (inclusive)")
        return p

    for name in list(STAGE_MODULES) + list(STAGE_GROUPS):
        common(sub.add_parser(name, help=_help_for(name)))

    inspect = sub.add_parser("paths", help="print the artifact paths this config resolves to")
    inspect.add_argument("--config", default=str(DEFAULT_CONFIG))
    inspect.add_argument("--sample", default="<id>")

    check = sub.add_parser("samples", help="list discovered samples and their tiers")
    check.add_argument("--config", default=str(DEFAULT_CONFIG))
    check.add_argument("--limit", type=int, default=None)
    check.add_argument("--only", nargs="+", metavar="ID")

    return parser


def _help_for(name: str) -> str:
    if name in STAGE_GROUPS:
        return f"run: {', '.join(STAGE_GROUPS[name])}"
    return f"run the {STAGE_MODULES[name][1]} agent"


def _load(cfg_path: str):
    from .config import ConfigError, load_config
    try:
        return load_config(cfg_path)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        raise SystemExit(2)


def _discover(cfg, args):
    from .samples import discover_samples
    limit = args.limit if args.limit is not None else (cfg.limit or None)
    return discover_samples(cfg.dataset_root, limit=limit, only=getattr(args, "only", None))


def _run_stage(name: str, cfg, samples, force: bool) -> bool:
    """Run one agent. Returns False if it produced no successful output."""
    import importlib

    module_path, agent = STAGE_MODULES[name]
    try:
        module = importlib.import_module(f".{module_path}", package=__package__)
    except ImportError as e:
        # An agent that isn't implemented yet is skipped, not a failure --
        # it must not halt a multi-stage run. Stage 1b is optional by design.
        print(f"\n[{name}] skipped: not implemented ({e})")
        return True

    agent_cfg = cfg.agents.get(agent)
    if agent_cfg is None:
        print(f"\n[{name}] skipped: no '{agent}' section in the config")
        return True
    if agent_cfg.option("enabled", True) is False:
        print(f"\n[{name}] skipped: disabled in the config")
        return True

    report = module.run(cfg, samples, force=force)
    report.print_report()
    # An unresolved artifact still feeds the next stage, so only a stage that
    # produced nothing at all should halt the chain.
    return bool(report.produced)


def main() -> None:
    args = _build_parser().parse_args()

    # Must precede any torch-touching import: torch locks device visibility at
    # first CUDA import, so setting this later has no effect.
    if getattr(args, "gpu", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = _load(args.config)

    if args.stage == "paths":
        from .cache import CachePaths
        for key, value in CachePaths.from_config(cfg).describe(args.sample).items():
            print(f"{key:16} {value}")
        return

    if args.stage == "samples":
        for sample in _discover(cfg, args):
            tiers = ",".join(sample.available_tiers())
            print(f"{sample.id:26} {sample.image_path.name:30} tiers={tiers}")
        return

    samples = _discover(cfg, args)
    if not samples:
        print(f"no samples found under {cfg.dataset_root}", file=sys.stderr)
        raise SystemExit(1)

    print(f"{len(samples)} sample(s) | style={cfg.style} | target={cfg.target} "
          f"| config={Path(args.config).name}")

    stages = resolve_stages(args.stage, args.from_stage, args.to_stage)
    if stages != STAGE_GROUPS.get(args.stage, [args.stage]):
        print(f"running: {' -> '.join(stages)}")
    for name in stages:
        ok = _run_stage(name, cfg, samples, args.force)
        # In a multi-stage run, a stage that produced nothing means every
        # later stage would fail on missing inputs -- stop with the real cause
        # visible rather than burying it under cascading errors.
        if not ok and len(stages) > 1:
            print(f"\nstopping: '{name}' produced no output", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
