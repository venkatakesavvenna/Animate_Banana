"""Human-in-the-loop correction tool for pipeline Stage 1 (D2C + rasters).

Two screens:

  1. `/sample/<id>`  -- source image + generated diagram code side by side.
     Copy the exact prompt+image out to try a stronger model by hand, paste
     an improved version back, and (explicitly) promote it into the
     pipeline's own `code_final` so Stage 2 onward picks it up.
  2. `/rasters/<id>`  -- drag/resize the auto-detected raster boxes over the
     source image, then re-crop and re-splice into a human `code_final`.

Every write lands in a new, tool-owned cache subtree (`annotations/`,
`code_human/`, `code_final_human/`) so a pipeline re-run can never silently
clobber a human edit, and a human edit never masquerades as pipeline output.
Only the explicit "Promote" action touches `CachePaths.code_final`.

Built for 3-4 people running this locally against disjoint sample subsets
(`--only`/`--shard`); see docs/annotate_merge.md for combining results
afterward when working without shared storage.

Run inside the project docker container:
    python -m img_2_svg_pretraining.pipeline.annotate.app \\
        --config pipeline/configs/default.yaml --only CVPR_2025_pipe00041

Then open http://<host>:7862. The port must be published at container
creation (`docker run -p 7862:7862`), like the other inspector ports.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from ..animator import critic as animator_critic
from ..animator import designer, exporter
from ..cache import CODE_SUFFIX, CachePaths, write_text
from ..config import load_config
from ..planner import critic as planner_critic
from ..planner import narrative_writer, parser, sequencer, strategizer
from ..runner import AgentContext
from ..samples import discover_samples
from ..schema import AnimationSequence
from ..styles import STYLES
from ..transmuter import code_converter, raster_integrator
from ..transmuter import critic as diagram_critic
from ..transmuter.raster_integrator import _crop
from ..transmuter.tikz_rasters import find_placeholders, splice
from . import store, xml_edit
from .requests import build_copyable_request
from .shard import parse_shard, partition_ids

app = Flask(__name__)

STATE: dict = {"cfg": None, "paths": None, "samples": [], "by_id": {},
               "style_pool": sorted(STYLES)}


def _init(config_path: str, only: list[str] | None, shard: str | None,
          styles: list[str] | None = None) -> None:
    cfg = load_config(config_path)
    samples = discover_samples(cfg.dataset_root)
    ids = [s.id for s in samples]

    if shard:
        i, n = parse_shard(shard)
        wanted = set(partition_ids(ids, n, i))
        samples = [s for s in samples if s.id in wanted]
    if only:
        wanted = set(only)
        samples = [s for s in samples if s.id in wanted]

    pool = sorted(styles) if styles else sorted(STYLES)
    unknown = [s for s in pool if s not in STYLES]
    if unknown:
        raise SystemExit(f"unknown style(s) {unknown}; known: {sorted(STYLES)}")

    STATE.update(
        cfg=cfg,
        paths=CachePaths.from_config(cfg),
        samples=samples,
        by_id={s.id: s for s in samples},
        style_pool=pool,
    )


def _sample(sample_id: str):
    sample = STATE["by_id"].get(sample_id)
    if sample is None:
        abort(404, f"unknown sample '{sample_id}' (outside this instance's shard/--only set)")
    return sample


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


def _ctx() -> AgentContext:
    return AgentContext(STATE["cfg"], "code_converter")


# -- Screen 1: D2C code review -------------------------------------------

@app.get("/api/samples")
def api_samples():
    paths: CachePaths = STATE["paths"]
    out = []
    for sample in STATE["samples"]:
        ann = store.load(paths.root, sample.id)
        out.append({
            "id": sample.id,
            "title": sample.title,
            "has_code": paths.code(sample.id).exists(),
            "has_code_final": paths.code_final(sample.id).exists(),
            "stage1a_status": ann["stage1a"]["status"],
            "stage1b_status": ann["stage1b"]["status"],
        })
    return jsonify({"samples": out})


@app.get("/api/image/<sample_id>")
def api_image(sample_id):
    return send_file(_sample(sample_id).image_path)


def _stage_state(sample_id: str, ann: dict) -> dict:
    """Which Stage-1 artifacts exist, and which are stale.

    "Stale" is decided by modification time against the artifact upstream of
    it, so editing 1a visibly invalidates the splice and the critique instead
    of leaving the reviewer to remember that it did.
    """
    paths: CachePaths = STATE["paths"]
    human_1a = ann["stage1a"]["human_code_path"]

    def mtime(path) -> float | None:
        p = Path(path) if path else None
        return p.stat().st_mtime if p and p.is_file() else None

    # A pasted 1a override supersedes the converter's own output as the thing
    # everything downstream should have been built from.
    t_1a = max(filter(None, [mtime(paths.code(sample_id)), mtime(human_1a)]), default=None)
    t_1b = mtime(paths.code_final(sample_id))
    t_1c = mtime(paths.code_reviewed(sample_id))

    stale_1b = bool(t_1a and t_1b and t_1b < t_1a)
    stale_1c = bool(t_1c and ((t_1a and t_1c < t_1a) or (t_1b and t_1c < t_1b)))

    return {
        "has_1a": t_1a is not None,
        "has_1b": t_1b is not None,
        "has_1c": t_1c is not None,
        "stale_1b": stale_1b,
        "stale_1c": stale_1c,
        # What the reviewer should run to make everything current again.
        "needs": (["1b", "1c"] if (stale_1b or not t_1b)
                  else ["1c"] if (stale_1c or not t_1c) else []),
    }


@app.get("/api/sample/<sample_id>")
def api_sample(sample_id):
    sample = _sample(sample_id)
    paths: CachePaths = STATE["paths"]
    code = _read(paths.code(sample_id))
    ann = store.load(paths.root, sample_id)
    return jsonify({
        "id": sample_id,
        "title": sample.title,
        "code": code,
        "code_lineage": paths.code_lineage,
        "human_code": _read(Path(ann["stage1a"]["human_code_path"]))
                      if ann["stage1a"]["human_code_path"] else None,
        "code_reviewed": _read(paths.code_reviewed(sample_id)),
        "stages": _stage_state(sample_id, ann),
        "annotation": ann,
    })


# The full pipeline, in the order run_pipeline.py defines it. The order is not
# a preference: 1b reads the raw converter output and rewrites code_final from
# it, so running it after 1c silently discards the critic's repairs; and 1c
# renders the diagram to compare it against the figure, so running it before
# 1b scores a diagram of empty placeholder boxes and "fixes" the missing
# images by inventing crops that do not exist yet. Each stage invalidates
# every later one, which is what `_STAGE_ORDER` encodes.
_STAGE_ORDER = ["1a", "1b", "1c", "2a", "2b", "2c", "2d", "2e", "3a", "3b", "3c"]

_STAGES = {
    "1a": ("convert-code", code_converter),
    "1b": ("integrate-rasters", raster_integrator),
    "1c": ("critique-diagram", diagram_critic),
    "2a": ("strategize", strategizer),
    "2b": ("parse", parser),
    "2c": ("sequence", sequencer),
    "2d": ("critique-sequence", planner_critic),
    "2e": ("narrate", narrative_writer),
    "3a": ("design", designer),
    "3b": ("critique-animation", animator_critic),
    "3c": ("export", exporter),
}

# Stages whose output is a sampling decision rather than a measurement, so a
# temperature override is meaningful. 1b is detection (pinned at 0.0 for
# reproducibility) and 3c is the exporter, which has no model in it at all.
_CREATIVE = {"1a", "1c", "2a", "2b", "2c", "2d", "2e", "3a", "3b"}


def _effective_1a(sample_id: str) -> tuple[str | None, bool]:
    """The Stage-1a code everything downstream should be working from.

    A pasted 1a supersedes the converter's output: it is what 1b splices into
    and what 1c critiques, so it must also be what the raster screen parses
    placeholders out of. Reading `paths.code()` there instead would list the
    converter's raster nodes while the splice used the reviewer's -- the boxes
    on screen would not be the boxes being edited.

    Returns `(code, is_human)`.
    """
    paths: CachePaths = STATE["paths"]
    ann = store.load(paths.root, sample_id)
    human = ann["stage1a"].get("human_code_path")
    if human:
        code = _read(Path(human))
        if code:
            return code, True
    return _read(paths.code(sample_id)), False


@contextmanager
def _human_1a_applied(sample_id: str):
    """Make a pasted 1a *be* the Stage-1a artifact for the duration of a run.

    `raster_integrator` reads `paths.code()` and nothing else -- it predates
    this tool and has no notion of a human override -- so a reviewer who
    rewrote 1a by hand would watch 1b re-splice the original converter output
    and silently discard the edit. Rather than patch a stage the CLI shares,
    swap the file underneath it and restore afterwards, so `paths.code()` is
    the reviewer's code exactly while the stage runs.

    The original is kept beside it as `.pre_human`, not deleted: it is the
    only copy of what the converter actually produced, and the eval suite
    still needs to be able to tell the two apart.
    """
    paths: CachePaths = STATE["paths"]
    ann = store.load(paths.root, sample_id)
    human = ann["stage1a"].get("human_code_path")
    target = paths.code(sample_id)

    if not human or not Path(human).is_file():
        yield False
        return

    with _swapped(human, target):
        yield True


@contextmanager
def _swapped(human_path, target: Path):
    """Put a human file at `target` for the duration, then put the original back.

    The stages this tool drives read pipeline-owned paths directly and have no
    notion of a human override -- so rather than fork a stage the CLI shares,
    the file underneath it is swapped and restored. The original is kept as
    `.pre_human` while the swap is live: it is the only copy of what the
    pipeline actually produced, and restoring from it in `finally` means a
    crashed run cannot leave a human file impersonating pipeline output.
    """
    target = Path(target)
    backup = target.with_suffix(target.suffix + ".pre_human")
    existed = target.is_file()
    if existed and not backup.exists():
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(human_path, target)
    try:
        yield
    finally:
        if backup.exists():
            shutil.copy2(backup, target)
            backup.unlink()
        elif not existed:
            # Nothing was there to restore, so leaving the copy behind would
            # plant a human file at a pipeline-owned path with nothing to
            # mark it as human -- precisely the confusion this whole
            # swap-and-restore dance exists to prevent.
            target.unlink(missing_ok=True)


@contextmanager
def _human_xml_applied(sample_id: str, paths: CachePaths, skip: bool = False):
    """Make a hand-edited structure XML *be* the XML for the duration of a run.

    `sequencer._load_xml` reads `paths.xml(id)` with no fallback and raises
    without it, and the planner critic reads it twice more. Skipped when the
    parser itself is in the run: it would overwrite the swapped file, and the
    human edit would come back looking like model output.
    """
    ann = store.load(paths.root, sample_id)
    human = ann["stage2b"].get("human_xml_path")
    if skip or not human or not Path(human).is_file():
        yield False
        return
    with _swapped(human, paths.xml(sample_id)):
        yield True


@contextmanager
def _human_sequence_applied(sample_id: str, paths: CachePaths, skip: bool = False):
    """Make a hand-edited sequence *be* `sequence_final` for the duration.

    That is the path the designer, the animation critic and the narrative
    writer all read first, so this is what makes "render video" animate the
    corrected sequence. Skipped when the sequence critic is in the run, since
    `sequence_final` is its own output path.
    """
    ann = store.load(paths.root, sample_id)
    human = ann["stage2d"].get("human_sequence_path")
    if skip or not human or not Path(human).is_file():
        yield False
        return
    with _swapped(human, paths.sequence_final(sample_id)):
        yield True


@contextmanager
def _style_applied(style: str | None):
    """Run under one animation style.

    `cfg` is a single process-global object shared by every request, and the
    style is baked into `sequence_lineage` -- so without restoring it, a run
    triggered by one request would write into another request's style.
    `compare.py::_paths_for` mutates and never restores, which is safe only
    because that viewer is read-only.

    Both keys are set: `cfg.style` is what the lineage and the prompt blocks
    read, `cfg.raw["animation_style"]` is what `fingerprint()` -- and so
    provenance -- reads. `run_pipeline.py` sets the same pair.
    """
    if not style:
        yield
        return
    cfg = STATE["cfg"]
    previous_style, previous_raw = cfg.style, cfg.raw.get("animation_style")
    cfg.style = style
    cfg.raw["animation_style"] = style
    try:
        yield
    finally:
        cfg.style = previous_style
        if previous_raw is None:
            cfg.raw.pop("animation_style", None)
        else:
            cfg.raw["animation_style"] = previous_raw


def _style_for(sample_id: str) -> str:
    """This sample's style, assigning one on first use.

    Seeded on the sample id rather than global `random`, so the draw is
    reproducible: every annotator's sharded instance derives the same style,
    and a lost record re-derives rather than re-rolls. Assignment happens
    inside the store's locked read-modify-write so two concurrent first
    visits cannot assign two different styles.
    """
    paths: CachePaths = STATE["paths"]
    record = store.load(paths.root, sample_id)
    assigned = record["style"].get("assigned")
    if assigned in STYLES:
        return assigned

    pool = STATE["style_pool"]
    choice = random.Random(sample_id).choice(pool)

    def mutate(rec):
        if rec["style"].get("assigned") in STYLES:
            return                      # another request won the race
        rec["style"].update(assigned=choice, source="random",
                            assigned_at=store.now())
        rec["style"]["history"].append(
            {"style": choice, "source": "random", "at": store.now()})

    return store.update(paths.root, sample_id, mutate)["style"]["assigned"]


def _styled_paths() -> CachePaths:
    """CachePaths resolved under whatever style is currently applied.

    `STATE["paths"]` was built once at startup from the config's default
    style, so anything style-keyed -- sequence, exports, audio, marked --
    would resolve to the wrong directory through it. Must be called inside
    `_style_applied`; CachePaths is a cheap dataclass, so building one per
    request costs nothing.
    """
    return CachePaths.from_config(STATE["cfg"])


@contextmanager
def _temperature(value: float | None, stages: list[str]):
    """Temporarily raise the sampling temperature of the stages being run.

    Regenerating with the config's own temperature returns the same answer:
    the backend caches on a fingerprint that includes the params, so an
    identical request is served from cache rather than resampled. Changing
    the temperature both asks the model to explore and makes the request a
    genuine cache miss, which is what "generate something else" needs.

    Detection (1b) is deliberately left alone -- localisation is not a
    creative task and its config pins temperature 0.0 for reproducibility.
    """
    if value is None:
        yield
        return

    creative = [s for s in stages if s in _CREATIVE]
    agents = [_STAGES[s][1].AGENT for s in creative]
    saved: dict[str, float | None] = {}
    for name in agents:
        params = STATE["cfg"].agent(name).params
        saved[name] = params.get("temperature")
        params["temperature"] = value
    try:
        yield
    finally:
        for name, previous in saved.items():
            params = STATE["cfg"].agent(name).params
            if previous is None:
                params.pop("temperature", None)
            else:
                params["temperature"] = previous


def _run_stages(sample, stages: list[str], force: bool) -> tuple[list[dict], str | None]:
    """Run stages in pipeline order, stopping at the first failure.

    Returns `(results, error)`. A stage that fails aborts the rest: the later
    stages consume its output, so running them anyway would score or splice
    into an artifact the reviewer has just been told is broken.
    """
    results: list[dict] = []
    for stage in sorted(stages, key=_STAGE_ORDER.index):
        label, module = _STAGES[stage]
        try:
            report = module.run(STATE["cfg"], [sample], force=force)
        except Exception as e:
            return results, f"{label}: {type(e).__name__}: {e}"

        outcome = report.outcomes[0] if report.outcomes else None
        if outcome is None:
            return results, f"{label}: produced no outcome"
        results.append({"stage": stage, "label": label,
                        "status": outcome.status, "detail": outcome.detail})
        if outcome.status == "failed":
            return results, f"{label}: {outcome.detail}"
    return results, None


def _run_with_human_edits(sample, stages: list[str],
                          force: bool) -> tuple[list[dict], str | None]:
    """Run stages with every human artifact standing in for its pipeline path.

    The stages read pipeline-owned paths and know nothing about human edits,
    so each override is swapped in underneath them -- except where the stage
    that *produces* that artifact is itself in the run, which would overwrite
    the edit and pass it off as model output.

    Style is outermost because the inner swaps resolve lineage-keyed paths and
    would otherwise back up the wrong file entirely.

    Each applied swap is reported as a result row: without it there is no way
    to tell from the outside whether an edit was actually consumed, which is
    exactly how this bug class went unnoticed in Stage 1.
    """
    sample_id = sample.id
    style = _style_for(sample_id)
    notes: list[dict] = []

    with _style_applied(style):
        paths = _styled_paths()
        with _human_1a_applied(sample_id) as used_1a, \
             _human_xml_applied(sample_id, paths, skip="2b" in stages) as used_xml, \
             _human_sequence_applied(sample_id, paths, skip="2d" in stages) as used_seq:
            if used_1a:
                notes.append({"stage": "1a", "label": "human override", "status": "applied",
                              "detail": "ran against your edited diagram code"})
            if used_xml:
                notes.append({"stage": "2b", "label": "human override", "status": "applied",
                              "detail": "ran against your edited structure XML"})
            if used_seq:
                notes.append({"stage": "2d", "label": "human override", "status": "applied",
                              "detail": "ran against your edited sequence"})
            results, error = _run_stages(sample, stages, force)

    return notes + results, error


@app.post("/api/run-stages/<sample_id>")
def api_run_stages(sample_id):
    """Run one or more Stage-1 agents, always in pipeline order.

    Body: `{stages: ["1b", "1c"], force: bool, temperature: float|null}`. The
    caller says what changed; ordering and the fact that later stages must
    follow is enforced here rather than trusted to the UI.
    """
    sample = _sample(sample_id)
    body = request.json or {}
    force = bool(body.get("force", False))
    stages = body.get("stages") or []
    unknown = [s for s in stages if s not in _STAGES]
    if not stages or unknown:
        abort(400, f"stages must be a non-empty subset of {_STAGE_ORDER}; got {stages}")

    temperature = body.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            abort(400, f"temperature must be a number, got {body.get('temperature')!r}")
        if not 0.0 <= temperature <= 2.0:
            abort(400, "temperature must be between 0.0 and 2.0")

    ctx = _ctx()
    try:
        with _temperature(temperature, stages):
            results, error = _run_with_human_edits(sample, stages, force)
    finally:
        ctx.unload()

    payload = {"ok": error is None, "results": results,
               "temperature": temperature,
               "code": _read(STATE["paths"].code(sample_id))}
    if error:
        payload["error"] = error
        return jsonify(payload), 422
    return jsonify(payload)


def _resolve_render_code(sample_id: str, source: str) -> tuple[str | None, str]:
    """The code one screen wants to see, and the label for where it came from.

    Screens disagree about what "current" means, and the disagreement is the
    point: the D2C screen must not show spliced rasters before the reviewer
    has touched them, or the render stops reflecting the code beside it.

      "diagram"  stage 1a/1c only -- never raster-spliced
      "rasters"  the raster-spliced version, human first
      "best"     highest-priority artifact overall (what Stage 2 would read)
    """
    paths: CachePaths = STATE["paths"]
    ann = store.load(paths.root, sample_id)
    human_1a = ann["stage1a"]["human_code_path"]
    human_1b = ann["stage1b"]["code_final_human_path"]

    def read(path) -> str | None:
        return _read(Path(path)) if path else None

    if source == "diagram":
        # The pipeline's own resolution order, which is what the reviewer is
        # here to judge: 1c's repaired version, else the splice, else the raw
        # converter output. An earlier version of this skipped anything
        # containing \includegraphics, to keep rasters off the diagram screen
        # -- but after a correctly ordered run *every* reviewed artifact is
        # spliced, so that rejected the only versions that compile and fell
        # through to the raw 1a the critic exists to repair.
        for label, code in (("human 1a", read(human_1a)),
                            ("code_reviewed", _read(paths.code_reviewed(sample_id))),
                            ("code_final", _read(paths.code_final(sample_id))),
                            ("code (1a)", _read(paths.code(sample_id)))):
            if code:
                return code, label
        return None, ""

    if source == "rasters":
        # `code_reviewed` sits above `code_final` here for the same reason the
        # insert route splices into it: 1c reviews the spliced diagram, so it
        # is the raster-bearing version *plus* the repairs that often make it
        # compile at all. Reading code_final directly showed the reviewer the
        # pre-critic bugs (pipe00045's unescaped `rgb:` colour expression).
        for label, code in (("human 1b", read(human_1b)),
                            ("code_reviewed", _read(paths.code_reviewed(sample_id))),
                            ("code_final", _read(paths.code_final(sample_id)))):
            if code:
                return code, label
        return None, ""

    for label, code in (("human 1a", read(human_1a)),
                        ("human 1b", read(human_1b)),
                        ("code_reviewed", _read(paths.code_reviewed(sample_id))),
                        ("code_final", _read(paths.code_final(sample_id))),
                        ("code (1a)", _read(paths.code(sample_id)))):
        if code:
            return code, label
    return None, ""


@app.get("/api/render/<sample_id>")
def api_render(sample_id):
    """Compiled render, built on demand. `?source=diagram|rasters|best`."""
    from ...viewer.compile import compile_tikz

    paths: CachePaths = STATE["paths"]
    source = request.args.get("source", "best")
    if source not in ("diagram", "rasters", "best"):
        abort(400, "source must be diagram, rasters or best")

    code, label = _resolve_render_code(sample_id, source)
    if not code:
        return jsonify({"ok": False, "error": f"no code available for source '{source}'"}), 404
    result = compile_tikz(code, paths.compile_cache())
    if not result.ok or result.png_path is None:
        return jsonify({"ok": False, "from": label, "log": result.log[-4000:]}), 422
    response = send_file(result.png_path)
    # Lets the UI say which artifact it is looking at without a second call.
    response.headers["X-Render-Source"] = label
    return response


@app.get("/api/copy-block/<sample_id>")
def api_copy_block(sample_id):
    """The exact request one agent would send. `?agent=code_converter|parser`."""
    sample = _sample(sample_id)
    which = request.args.get("agent", "code_converter")
    modules = {"code_converter": code_converter, "parser": parser}
    if which not in modules:
        abort(400, f"agent must be one of {sorted(modules)}")

    ctx = AgentContext(STATE["cfg"], which)
    # The parser reads whatever code Stage 1 settled on, so a human 1a fix has
    # to be in force or the copied prompt shows code the run would not use.
    with _human_1a_applied(sample_id):
        block = build_copyable_request(ctx, sample, modules[which])
    return jsonify({
        "prompt": block.prompt_text,
        "model": block.model,
        "image_url": f"/api/image/{sample_id}",
    })


@app.post("/api/human-code/<sample_id>")
def api_human_code(sample_id):
    _sample(sample_id)
    body = request.json or {}
    text = body.get("text", "").strip()
    source = body.get("source", "pasted")
    if not text:
        abort(400, "empty code")

    paths: CachePaths = STATE["paths"]
    out = paths.root / "code_human" / paths.code_lineage / f"{sample_id}{CODE_SUFFIX[STATE['cfg'].target]}"
    write_text(out, text)

    def mutate(record):
        record["stage1a"]["status"] = "human_code_provided"
        record["stage1a"]["code_lineage_at_review"] = paths.code_lineage
        record["stage1a"]["human_code_path"] = str(out)
        record["stage1a"]["human_code_source"] = source
        record["stage1a"]["human_code_updated_at"] = store.now()

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "path": str(out), "annotation": record})


@app.post("/api/comment/<sample_id>")
def api_comment(sample_id):
    _sample(sample_id)
    body = request.json or {}
    text = body.get("text", "").strip()
    author = body.get("author", "unknown")
    target = body.get("target", "general")
    if not text:
        abort(400, "empty comment")

    paths: CachePaths = STATE["paths"]

    def mutate(record):
        if record["stage1a"]["status"] == "unreviewed":
            record["stage1a"]["status"] = "commented"
        record["stage1a"]["comments"].append({
            "id": f"c{len(record['stage1a']['comments']) + 1}",
            "author": author, "timestamp": store.now(),
            "text": text, "target": target,
        })

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "annotation": record})


@app.post("/api/promote/<sample_id>")
def api_promote(sample_id):
    """The one route that writes a pipeline-owned path. Explicit, auditable."""
    _sample(sample_id)
    body = request.json or {}
    stage = body.get("stage")
    if stage not in ("stage1a", "stage1b", "stage2b", "stage2d"):
        abort(400, "stage must be one of stage1a, stage1b, stage2b, stage2d")
    author = body.get("author", "unknown")

    paths: CachePaths = STATE["paths"]
    ann = store.load(paths.root, sample_id)

    sources = {
        "stage1a": ann["stage1a"]["human_code_path"],
        "stage1b": ann["stage1b"]["code_final_human_path"],
        "stage2b": ann["stage2b"]["human_xml_path"],
        "stage2d": ann["stage2d"]["human_sequence_path"],
    }
    src = sources[stage]
    if not src or not Path(src).exists():
        abort(404, f"no human artifact to promote for {stage}")

    text = Path(src).read_text(encoding="utf-8")
    flag = "promoted_to_code_final"
    if stage == "stage2b":
        out = paths.xml(sample_id)
        flag = "promoted_to_xml"
    elif stage == "stage2d":
        # sequence_final is style-keyed and is what the designer reads first,
        # so promoting there is what feeds Stage 3.
        with _style_applied(_style_for(sample_id)):
            out = _styled_paths().sequence_final(sample_id)
        flag = "promoted_to_sequence_final"
    else:
        out = paths.code_final(sample_id)
    write_text(out, text)

    def mutate(record):
        record[stage]["status"] = "promoted"
        record[stage][flag] = True
        record[stage]["promoted_by"] = author
        record[stage]["promoted_at"] = store.now()

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "path": str(out), "annotation": record})


@app.post("/api/discard/<sample_id>")
def api_discard(sample_id):
    """Write off the whole annotation for this sample as unsalvageable.

    Deletes every human-owned artifact and reverts a promotion, so the sample
    falls back to whatever the pipeline itself produced. Deliberately callable
    at any point -- the reviewer decides a sample is a lost cause, not the
    tool. The record is kept (status "discarded", with the reason) rather than
    removed: that a sample was tried and abandoned is a finding worth keeping,
    and without it the next person just repeats the work.
    """
    _sample(sample_id)
    body = request.json or {}
    author = body.get("author", "unknown")
    reason = (body.get("reason") or "").strip()

    paths: CachePaths = STATE["paths"]
    ann = store.load(paths.root, sample_id)
    removed: list[str] = []

    def unlink(path) -> None:
        if not path:
            return
        p = Path(path)
        if p.is_file():
            p.unlink()
            removed.append(str(p))

    unlink(ann["stage1a"]["human_code_path"])
    unlink(ann["stage1b"]["code_final_human_path"])
    for box in ann["stage1b"]["boxes"]:
        unlink(box.get("crop_path"))

    unlink(ann["stage2b"].get("human_xml_path"))
    unlink(ann["stage2d"].get("human_sequence_path"))

    # A promotion copied human work into a pipeline-owned path; discarding the
    # work has to take that copy with it, or the pipeline keeps reading the
    # thing the reviewer just declared unusable.
    if (ann["stage1a"].get("promoted_to_code_final")
            or ann["stage1b"].get("promoted_to_code_final")):
        unlink(paths.code_final(sample_id))
    if ann["stage2b"].get("promoted_to_xml"):
        unlink(paths.xml(sample_id))
    if ann["stage2d"].get("promoted_to_sequence_final"):
        # sequence_final is style-keyed, so it has to be resolved under the
        # style the promotion was made for.
        with _style_applied(ann["stage2d"].get("style_at_review")):
            unlink(_styled_paths().sequence_final(sample_id))

    def mutate(record):
        for stage in ("stage1a", "stage1b"):
            record[stage]["promoted_to_code_final"] = False
            record[stage].pop("promoted_by", None)
            record[stage].pop("promoted_at", None)
        record["stage1a"].update(status="discarded", human_code_path=None,
                                 human_code_source=None, human_code_updated_at=None)
        record["stage1b"].update(status="discarded", boxes=[],
                                 code_final_human_path=None)
        record["stage2b"].update(status="discarded", human_xml_path=None,
                                 human_xml_source=None, human_xml_updated_at=None,
                                 ids_added=[], ids_removed=[],
                                 promoted_to_xml=False)
        record["stage2d"].update(status="discarded", human_sequence_path=None,
                                 human_sequence_updated_at=None,
                                 validation_at_save=[],
                                 promoted_to_sequence_final=False)
        record["discarded"] = {"by": author, "at": store.now(), "reason": reason}

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "removed": removed, "annotation": record})


# -- Screen 2: raster placement -------------------------------------------

@app.post("/api/run-rasterizer/<sample_id>")
def api_run_rasterizer(sample_id):
    """Stage 1b, always followed by 1c.

    Never runs 1b alone: it rewrites code_final from the raw converter output,
    so whatever the critic had already repaired is gone the moment it lands.
    Re-critiquing is the only thing that makes the new code_final valid again.
    """
    sample = _sample(sample_id)
    force = bool((request.json or {}).get("force", False))
    paths: CachePaths = STATE["paths"]
    if not paths.code(sample_id).exists():
        abort(422, "no diagram code yet; generate Stage 1a first")

    ctx = _ctx()
    try:
        # Both stages run under the swap, so the splice and the critique are
        # built from the reviewer's 1a rather than the converter's.
        with _human_1a_applied(sample_id) as applied:
            # 1c is forced regardless: its cached artifact was reviewed
            # against the previous splice, so reusing it would describe a
            # diagram that no longer exists.
            results, error = _run_stages(sample, ["1b"], force)
            if error is None:
                more, error = _run_stages(sample, ["1c"], True)
                results += more
            if applied:
                results.insert(0, {"stage": "1a", "label": "human override",
                                   "status": "applied",
                                   "detail": "spliced and critiqued your edited 1a"})
    finally:
        ctx.unload()

    payload = {"ok": error is None, "results": results}
    if error:
        payload["error"] = error
        return jsonify(payload), 422
    return jsonify(payload)


@app.get("/api/rasters/<sample_id>")
def api_rasters(sample_id):
    _sample(sample_id)
    paths: CachePaths = STATE["paths"]
    code, from_human = _effective_1a(sample_id)
    if not code:
        return jsonify({"available": False, "reason": "no diagram code yet"})

    placeholders = find_placeholders(code)
    directory = paths.rasters(sample_id)
    detections_path = directory / "detections.json"

    import json as _json
    regions_by_id: dict[str, list] = {}
    # A detection run that failed records why. Surfacing it matters: a
    # truncated response leaves zero regions, which is indistinguishable from
    # "never run" unless the reason is shown -- and the fix (raise the
    # agent's max_tokens) is not something a reviewer can guess from an
    # empty box list.
    detection_error: str | None = None
    if detections_path.exists():
        try:
            data = _json.loads(detections_path.read_text())
            detection_error = data.get("error")
            for r in data.get("regions", []):
                regions_by_id[r["xml_id"]] = r.get("bbox")
        except (OSError, _json.JSONDecodeError, KeyError):
            pass

    ann = store.load(paths.root, sample_id)
    human_boxes = {b["xml_id"]: b["human_bbox"] for b in ann["stage1b"]["boxes"]}

    # Boxes are image-pixel coordinates, and detection is the only thing that
    # produces them. `RasterPlaceholder.bbox()` is tempting here and wrong:
    # it returns TikZ canvas coordinates meant for overlaying a *compiled
    # render*, not the source figure, so using it as a fallback yields boxes
    # that crop the wrong region -- or, when the two spaces disagree enough,
    # sit off the image entirely. A placeholder with no detection has no
    # position yet; say so rather than invent one.
    boxes = [
        {
            "xml_id": p.xml_id,
            "label": p.text[:80],
            "auto_bbox": regions_by_id.get(p.xml_id),
            "human_bbox": human_boxes.get(p.xml_id) or regions_by_id.get(p.xml_id),
        }
        for p in placeholders
    ]
    located = sum(1 for b in boxes if b["auto_bbox"])

    return jsonify({
        "available": True, "boxes": boxes,
        "from_human_1a": from_human,
        "located": located,
        "undetected": len(boxes) - located,
        "detection_error": detection_error,
        "has_code_final_human": bool(ann["stage1b"]["code_final_human_path"]),
    })


@app.post("/api/insert-raster/<sample_id>")
def api_insert_raster(sample_id):
    sample = _sample(sample_id)
    body = request.json or {}
    edits: dict[str, list[float]] = body.get("boxes") or {}
    author = body.get("author", "unknown")
    if not edits:
        abort(400, "no boxes given")

    paths: CachePaths = STATE["paths"]
    # Splice into the best repaired version, not the raw converter output.
    # `splice()` only rewrites placeholder node bodies and leaves the rest of
    # the document byte-identical, so re-splicing an already-spliced artifact
    # swaps its crops while keeping 1c's fixes -- and those fixes are often
    # the only reason it compiles at all (pipe00041's `child_node/.style`).
    # Starting from raw 1a here silently reverted the critic every time a
    # reviewer adjusted a box.
    ann_now = store.load(paths.root, sample_id)
    human_1a = ann_now["stage1a"].get("human_code_path")
    code, base = None, ""
    for label, candidate in (
        ("human 1a", _read(Path(human_1a)) if human_1a else None),
        ("code_reviewed", _read(paths.code_reviewed(sample_id))),
        ("code_final", _read(paths.code_final(sample_id))),
        ("code (1a)", _read(paths.code(sample_id))),
    ):
        if candidate:
            code, base = candidate, label
            break
    if not code:
        abort(404, "no diagram code; run Stage 1a first")

    work_dir = paths.rasters(sample_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    replacements: dict[str, str] = {}
    box_records = []
    placeholders = {p.xml_id: p for p in find_placeholders(code)}
    for xml_id, bbox in edits.items():
        if xml_id not in placeholders:
            continue
        crop_path = _crop(sample.image_path, tuple(bbox),
                           work_dir / f"crop_{xml_id}.human.png")
        if crop_path is None:
            continue
        replacements[xml_id] = crop_path.resolve().as_posix()
        box_records.append({
            "xml_id": xml_id,
            "human_bbox": list(bbox),
            "adjusted_by": author,
            "adjusted_at": store.now(),
            "crop_path": str(crop_path),
            "inserted": True,
        })

    if not replacements:
        return jsonify({"ok": False, "error": "no boxes produced a usable crop"}), 422

    new_code, replaced = splice(code, replacements)
    out = paths.root / "code_final_human" / paths.code_lineage / f"{sample_id}{CODE_SUFFIX[STATE['cfg'].target]}"
    write_text(out, new_code)

    def mutate(record):
        record["stage1b"]["status"] = "adjusted"
        record["stage1b"]["code_lineage_at_review"] = paths.code_lineage
        existing = {b["xml_id"]: b for b in record["stage1b"]["boxes"]}
        for b in box_records:
            existing[b["xml_id"]] = b
        record["stage1b"]["boxes"] = list(existing.values())
        record["stage1b"]["code_final_human_path"] = str(out)

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "replaced": replaced, "base": base,
                    "path": str(out), "annotation": record})


# -- Screen 3: structure XML ---------------------------------------------

def _effective_xml(sample_id: str, paths: CachePaths) -> tuple[str | None, bool]:
    """The XML everything downstream should work from, human override first."""
    ann = store.load(paths.root, sample_id)
    human = ann["stage2b"].get("human_xml_path")
    if human:
        text = _read(Path(human))
        if text:
            return text, True
    return _read(paths.xml(sample_id)), False


@app.get("/api/xml/<sample_id>")
def api_xml(sample_id):
    _sample(sample_id)
    paths: CachePaths = STATE["paths"]           # xml is not style-keyed
    machine = _read(paths.xml(sample_id))
    effective, from_human = _effective_xml(sample_id, paths)
    ann = store.load(paths.root, sample_id)

    tree = None
    parse_error = None
    if effective:
        try:
            tree = xml_edit.to_tree(xml_edit.parse(effective))
        except ET.ParseError as e:
            parse_error = str(e)

    code_path = None
    try:
        code_path = str(paths.resolve_code(sample_id))
    except FileNotFoundError:
        pass

    return jsonify({
        "id": sample_id,
        "xml": effective,
        "from_human": from_human,
        "has_machine_xml": machine is not None,
        "tree": tree,
        "parse_error": parse_error,
        "xml_lineage": paths.xml_lineage,
        "classes": list(xml_edit.CLASSES),
        # What the parser actually read, so the reviewer judges the XML
        # against the same code the pipeline used.
        "code": _effective_1a(sample_id)[0],
        "code_path": code_path,
        "annotation": ann,
    })


@app.post("/api/parse-xml/<sample_id>")
def api_parse_xml(sample_id):
    """Raw text -> tree, for the editor's mode switch.

    Server-side so the tree is built by the same ElementTree the pipeline
    parses with, rather than a second implementation in JS that could
    disagree about what is well-formed.
    """
    _sample(sample_id)
    text = (request.json or {}).get("text") or ""
    try:
        return jsonify({"ok": True, "tree": xml_edit.to_tree(xml_edit.parse(text))})
    except ET.ParseError as e:
        return jsonify({"ok": False, "error": f"not well-formed XML: {e}"}), 400


@app.post("/api/serialize-xml/<sample_id>")
def api_serialize_xml(sample_id):
    """Tree -> raw text, showing exactly what a save would write."""
    _sample(sample_id)
    tree = (request.json or {}).get("tree") or []
    return jsonify({"ok": True, "text": xml_edit.serialize(xml_edit.from_tree(tree))})


@app.post("/api/human-xml/<sample_id>")
def api_human_xml(sample_id):
    _sample(sample_id)
    body = request.json or {}
    source = body.get("source", "pasted")

    if body.get("tree") is not None:
        text = xml_edit.serialize(xml_edit.from_tree(body["tree"]))
    else:
        text = (body.get("text") or "").strip()
    if not text:
        abort(400, "empty XML")

    # Well-formedness is non-negotiable: `element_ids` returns an empty set on
    # malformed XML rather than raising, which would silently make every
    # sequence focus entry look dangling.
    try:
        root = xml_edit.parse(text)
    except ET.ParseError as e:
        return jsonify({"ok": False, "error": f"not well-formed XML: {e}"}), 400

    paths: CachePaths = STATE["paths"]
    machine = _read(paths.xml(sample_id))
    added, removed = xml_edit.id_diff(text, machine)

    # Removing an id breaks every sequence step that focuses it, so check the
    # sequence that exists for this sample's current style.
    style = _style_for(sample_id)
    with _style_applied(style):
        styled = _styled_paths()
        conflicts = xml_edit.focus_conflicts(
            removed, styled.sequence_final(sample_id)) or xml_edit.focus_conflicts(
            removed, styled.sequence(sample_id))

    out = paths.root / "xml_human" / paths.xml_lineage / f"{sample_id}.xml"
    write_text(out, text)

    def mutate(record):
        record["stage2b"].update(
            status="human_xml_provided",
            xml_lineage_at_review=paths.xml_lineage,
            human_xml_path=str(out),
            human_xml_source=source,
            human_xml_updated_at=store.now(),
            ids_added=added,
            ids_removed=removed,
        )

    record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "path": str(out), "root_tag": root.tag,
                    "ids_added": added, "ids_removed": removed,
                    "focus_conflicts": conflicts, "annotation": record})


# -- Screen 4: sequence + video -------------------------------------------

def _effective_sequence(sample_id: str, paths: CachePaths) -> tuple[str | None, str]:
    """The sequence to show, and where it came from."""
    ann = store.load(paths.root, sample_id)
    human = ann["stage2d"].get("human_sequence_path")
    for label, path in (("human", Path(human) if human else None),
                        ("sequence_final", paths.sequence_final(sample_id)),
                        ("sequence", paths.sequence(sample_id))):
        if path is not None:
            text = _read(path)
            if text:
                return text, label
    return None, ""


def _validate_sequence(text: str, sample_id: str, paths: CachePaths) -> dict:
    """Structural + style validation, exactly as the pipeline does it.

    Pure and local -- two file reads, no model call -- because it is the same
    `validate()` the sequence critic and the benchmark run.
    """
    from ..planner.parser import load_element_ids

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {"ok": False, "problems": [f"not valid JSON: {e}"], "dialect": None}

    # from_dict silently converts the bench dialect, losing `parent` links.
    # Report it rather than letting the conversion happen invisibly.
    dialect = ("bench" if isinstance(data, dict) and "sequence" in data
               and "nodes" not in data else "native")
    try:
        seq = AnimationSequence.from_dict(data)
    except Exception as e:
        return {"ok": False, "problems": [f"{type(e).__name__}: {e}"], "dialect": dialect}

    xml_path = paths.xml(sample_id)
    element_ids = load_element_ids(xml_path) if xml_path.exists() else None
    max_depth = STATE["cfg"].agent("sequencer").option("max_depth", None)
    problems = seq.validate(element_ids=element_ids, max_depth=max_depth)
    return {"ok": not problems, "problems": problems, "dialect": dialect,
            "steps": len(seq.nodes), "style": seq.style}


@app.get("/api/sequence-state/<sample_id>")
def api_sequence_state(sample_id):
    _sample(sample_id)
    style = _style_for(sample_id)
    with _style_applied(style):
        paths = _styled_paths()
        text, origin = _effective_sequence(sample_id, paths)
        exports = paths.exports(sample_id)
        mp4 = exports / "animation.mp4"
        frames_dir = exports / "frames"
        marked = sorted((paths.root / "marked" / paths.sequence_lineage)
                        .glob(f"{sample_id}.round*.png"))
        validation = (_validate_sequence(text, sample_id, paths) if text else None)
        payload = {
            "id": sample_id,
            "style": style,
            "styles": STATE["style_pool"],
            "sequence_lineage": paths.sequence_lineage,
            "sequence": text,
            "origin": origin,
            "validation": validation,
            "has_video": mp4.is_file(),
            "frames": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
            "has_marked": bool(marked),
            "other_styles": _other_styles_with_work(sample_id, style),
            "annotation": store.load(paths.root, sample_id),
        }
    return jsonify(payload)


def _other_styles_with_work(sample_id: str, current: str) -> list[str]:
    """Styles this sample already has a sequence for.

    A style flip resolves to different paths, so the screen empties. Saying
    where the work went stops that looking like data loss.
    """
    found = []
    for name in STATE["style_pool"]:
        if name == current:
            continue
        with _style_applied(name):
            paths = _styled_paths()
            if paths.sequence_final(sample_id).exists() or paths.sequence(sample_id).exists():
                found.append(name)
    return found


@app.post("/api/style/<sample_id>")
def api_style(sample_id):
    _sample(sample_id)
    style = (request.json or {}).get("style")
    if style not in STYLES:
        abort(400, f"unknown style '{style}'; known: {sorted(STYLES)}")

    def mutate(record):
        record["style"].update(assigned=style, source="manual",
                               assigned_at=store.now())
        record["style"]["history"].append(
            {"style": style, "source": "manual", "at": store.now()})

    record = store.update(STATE["paths"].root, sample_id, mutate)
    return jsonify({"ok": True, "style": style, "annotation": record})


@app.post("/api/validate-sequence/<sample_id>")
def api_validate_sequence(sample_id):
    _sample(sample_id)
    text = (request.json or {}).get("text") or ""
    with _style_applied(_style_for(sample_id)):
        return jsonify(_validate_sequence(text, sample_id, _styled_paths()))


@app.post("/api/human-sequence/<sample_id>")
def api_human_sequence(sample_id):
    _sample(sample_id)
    text = ((request.json or {}).get("text") or "").strip()
    if not text:
        abort(400, "empty sequence")

    style = _style_for(sample_id)
    with _style_applied(style):
        paths = _styled_paths()
        validation = _validate_sequence(text, sample_id, paths)
        if validation.get("dialect") is None:
            return jsonify({"ok": False, **validation}), 400

        # Round-trip through the schema so what lands on disk is always the
        # native dialect the loaders expect, whatever was pasted in.
        seq = AnimationSequence.from_dict(json.loads(text))
        out = paths.root / "sequence_human" / paths.sequence_lineage / f"{sample_id}.json"
        write_text(out, seq.to_json())

        def mutate(record):
            record["stage2d"].update(
                status="human_sequence_provided",
                sequence_lineage_at_review=paths.sequence_lineage,
                style_at_review=style,
                human_sequence_path=str(out),
                human_sequence_updated_at=store.now(),
                # Saved even when invalid: repairing violations is the critic's
                # job, and refusing a work-in-progress would be hostile.
                validation_at_save=validation.get("problems", []),
            )

        record = store.update(paths.root, sample_id, mutate)
    return jsonify({"ok": True, "path": str(out), "validation": validation,
                    "annotation": record})


@app.post("/api/run-stage2/<sample_id>")
def api_run_stage2(sample_id):
    """Run a Stage-2/3 slice under this sample's style.

    Body: `{stages: [...], force: bool, temperature: float|null}`.
    """
    sample = _sample(sample_id)
    body = request.json or {}
    stages = body.get("stages") or []
    force = bool(body.get("force", False))
    unknown = [s for s in stages if s not in _STAGES]
    if not stages or unknown:
        abort(400, f"stages must be a non-empty subset of {_STAGE_ORDER}; got {stages}")

    temperature = body.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            abort(400, "temperature must be a number")
        if not 0.0 <= temperature <= 2.0:
            abort(400, "temperature must be between 0.0 and 2.0")

    ctx = _ctx()
    try:
        with _temperature(temperature, stages):
            results, error = _run_with_human_edits(sample, stages, force)
    finally:
        ctx.unload()

    # `run_iterative_agent` writes its best attempt and reports "unresolved"
    # rather than "failed" when it runs out of rounds. Surface that as its own
    # state -- the run did produce an artifact, but not a converged one.
    payload = {"ok": error is None, "results": results,
               "unresolved": [r["stage"] for r in results
                              if r.get("status") == "unresolved"]}
    if error:
        payload["error"] = error
        return jsonify(payload), 422
    return jsonify(payload)


@app.get("/api/video/<sample_id>")
def api_video(sample_id):
    _sample(sample_id)
    with _style_applied(_style_for(sample_id)):
        path = _styled_paths().exports(sample_id) / "animation.mp4"
    if not path.is_file():
        abort(404, "no video; render it first")
    return send_file(path, mimetype="video/mp4")


def _frame_paths(sample_id: str) -> list[Path]:
    with _style_applied(_style_for(sample_id)):
        frames_dir = _styled_paths().exports(sample_id) / "frames"
    if not frames_dir.is_dir():
        return []
    # pdftoppm's zero-padding means a plain sort gives frame-1, frame-10,
    # frame-2; sort numerically instead.
    return sorted(frames_dir.glob("*.png"),
                  key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))


@app.get("/api/frames/<sample_id>")
def api_frames(sample_id):
    _sample(sample_id)
    return jsonify({"count": len(_frame_paths(sample_id))})


@app.get("/api/frame/<sample_id>/<int:index>")
def api_frame(sample_id, index):
    _sample(sample_id)
    frames = _frame_paths(sample_id)
    if not frames:
        abort(404, "no frames")
    return send_file(frames[max(0, min(index, len(frames) - 1))])


@app.get("/api/marked/<sample_id>")
def api_marked(sample_id):
    """The Set-of-Mark overlay the sequence critic saw: ids drawn on the figure."""
    _sample(sample_id)
    with _style_applied(_style_for(sample_id)):
        paths = _styled_paths()
        marks = sorted((paths.root / "marked" / paths.sequence_lineage)
                       .glob(f"{sample_id}.round*.png"))
    if not marks:
        abort(404, "no marked figure")
    return send_file(marks[-1])


# -- pages ------------------------------------------------------------

@app.get("/")
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/render_panel.js")
def render_panel_js():
    """Render/discard controls, shared by both screens so they behave alike."""
    return send_file(Path(__file__).parent / "render_panel.js",
                     mimetype="application/javascript")


@app.get("/rasters/<sample_id>")
def rasters_page(sample_id):
    _sample(sample_id)
    return (Path(__file__).parent / "rasters.html").read_text(encoding="utf-8")


@app.get("/xml/<sample_id>")
def xml_page(sample_id):
    _sample(sample_id)
    return (Path(__file__).parent / "xml.html").read_text(encoding="utf-8")


@app.get("/sequence/<sample_id>")
def sequence_page(sample_id):
    _sample(sample_id)
    return (Path(__file__).parent / "sequence.html").read_text(encoding="utf-8")


def main() -> None:
    # Not `parser`: that name is the planner's parser module at module scope,
    # and shadowing it here would break every route that runs stage 2b.
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--config", required=True)
    argp.add_argument("--port", type=int, default=7862)
    argp.add_argument("--host", default="0.0.0.0")
    argp.add_argument("--only", nargs="+", metavar="ID",
                      help="restrict this instance to these sample ids")
    argp.add_argument("--shard", metavar="I/N",
                      help="restrict to the I-th of N deterministic shards, e.g. 1/4")
    argp.add_argument("--styles", nargs="+", metavar="STYLE",
                      help="pool to draw each sample's animation style from "
                           f"(default: all of {sorted(STYLES)})")
    args = argp.parse_args()

    _init(args.config, args.only, args.shard, args.styles)
    print(f"{len(STATE['samples'])} sample(s) | config {args.config}")
    print(f"style pool: {', '.join(STATE['style_pool'])}")
    print(f"open http://<host>:{args.port}")
    # threaded: a "render video" request can run for minutes (latexmk plus a
    # multi-round critic), and must not block the figure and API requests the
    # rest of the UI makes while it runs.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
