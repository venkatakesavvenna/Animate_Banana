"""Stage completion gates: one definition of "done", read by both halves.

A reviewer moves to the next stage only when the current one is finished, and
"finished" must mean exactly one thing. So each gate is a pair:

  - a *machine* check, re-evaluated on every read (does it render, does it
    validate, does it parse)
  - a *human* check, the reviewer's sign-off

Both must hold. Re-evaluating the machine half on every read is what stops an
approval from freezing a stage: edit the sequence after approving it and the
gate closes again, because the violations come back.

The same function backs the UI and the API. A disabled button is a suggestion;
the 409 from `/api/run-stages` is the rule. Duplicating the criteria in
JavaScript would have let the two drift, which is exactly the ambiguity this
module exists to remove.

Narration (2e) has no gate on purpose. It is written into an already-valid
sequence and only touches the `narrative` field, so there is nothing for a
reviewer to approve and nothing it can invalidate.
"""
from __future__ import annotations

from pathlib import Path

from ..cache import CachePaths
from . import store

# Stage -> the stage whose gate must pass before it may run. Only the stages a
# human actually gates appear as *prerequisites*; a stage absent from the
# values (2e, 3b, 3c) is reachable as soon as its predecessor is clean.
PREREQUISITE = {
    "1b": "stage1a",
    "1c": "stage1a",
    "2b": "stage1b",
    "2c": "stage2b",
    "2d": "stage2c",
    # 2e (narrate) has no prerequisite on purpose. It only fills the
    # `narrative` field of whatever sequence exists and cannot invalidate
    # anything, and gating it behind 2d made the "run without the critic" path
    # unreachable -- the request was to skip 2d, and the gate then refused the
    # narration that runs alongside it.
    "3a": "stage2c",
    "3b": "stage3a",
    "3c": "stage3a",
}

# Reviewer-facing order, and the label each gate reports under.
STAGE_LABELS = {
    "stage1a": "1a diagram code",
    "stage1b": "1b rasters",
    "stage2b": "2b structure XML",
    "stage2c": "2c sequence",
    "stage2d": "2d reviewed sequence",
    "stage3a": "3a animation",
}


def _check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _renders(code: str | None, target: str, cache_dir: Path) -> tuple[bool, str]:
    """Whether code compiles. Empty code is a failure, not a pass."""
    if not code:
        return False, "no code"
    from ..render import render_code

    result = render_code(code, target, cache_dir)
    if result.ok:
        return True, ""
    tail = [ln for ln in (result.log or "").splitlines() if ln.startswith("!")]
    return False, (tail[0] if tail else "does not compile")[:200]


def animation_renders(code: str | None, target: str, cache_dir: Path) -> tuple[bool, str]:
    """Whether animation code *exports*, which is a stricter test than rendering.

    A single frame can compile while the export fails: the exporter feeds the
    document through `to_multipage_pdf_source`, and that rewrite is where
    m3grounder's `\\foreach` collides with its `#1`-parameterised styles
    ("Illegal parameter number in definition of \\pgffor@body"). Checking only
    `render_code` here reported that sample as fine right up until export --
    exactly the "stage says ok, artifact is broken" failure this gate exists
    to catch.

    Cached by content hash, like `compile_tikz` already is. Without it every
    gate read re-ran a full multipage compile -- measured at 5s per page load,
    repeated identically on every reload, on all five screens.
    """
    if not code:
        return False, "no code"
    if target != "tikz":
        # SVG has no multipage rewrite; frames come from a headless browser.
        return _renders(code, target, cache_dir)

    import hashlib
    import json as _json
    import tempfile

    from ..export.render import compile_pdf
    from ..export.tikz_source import to_multipage_pdf_source

    source = to_multipage_pdf_source(code)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    # A separate namespace from the PNG cache: same code, different question
    # ("does the whole deck build" vs "does page one rasterise").
    marker = Path(cache_dir) / "gate" / f"{digest}.json"
    if marker.is_file():
        try:
            cached = _json.loads(marker.read_text(encoding="utf-8"))
            return bool(cached["ok"]), cached.get("detail", "")
        except (OSError, ValueError, KeyError):
            pass       # unreadable cache entry is not a reason to fail the gate

    try:
        with tempfile.TemporaryDirectory() as tmp:
            compile_pdf(source, Path(tmp) / "gate.pdf")
        result = (True, "")
    except Exception as exc:                            # noqa: BLE001
        tail = [ln for ln in str(exc).splitlines()
                if ln.startswith("!") and "Fatal error" not in ln]
        result = (False, (tail[0] if tail else "does not export")[:200])

    # Failures are cached too: a document that does not compile will not start
    # compiling on its own, and re-proving it costs the same 5s every time.
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_json.dumps({"ok": result[0], "detail": result[1]}),
                          encoding="utf-8")
    except OSError:
        pass           # caching is an optimisation, never a correctness bound
    return result


def gate_status(sample_id: str, stage: str, paths: CachePaths, target: str,
                *, code_for: dict | None = None) -> dict:
    """Whether `stage` is complete. `{stage, label, passed, checks, blocking}`.

    `paths` must already be resolved under the sample's style and target --
    2c/2d/3a artifacts are style-keyed, and checking them against the default
    lineage silently reports on files the reviewer never saw.
    """
    ann = store.load(paths.root, sample_id)
    block = ann.get(stage) or {}
    checks: list[dict] = []

    if stage == "stage1a":
        code = (code_for or {}).get("diagram")
        ok, detail = _renders(code, target, paths.compile_cache())
        checks.append(_check("code renders", ok, detail))

    elif stage == "stage1b":
        code = (code_for or {}).get("rasters")
        ok, detail = _renders(code, target, paths.compile_cache())
        checks.append(_check("code renders", ok, detail))

    elif stage == "stage2b":
        xml_path = paths.xml(sample_id)
        # Per-target slot: the tikz and svg structure XMLs are separate
        # artifacts, and the gate must judge the one this target reads.
        human = store.human_xml_path(ann, target)
        path = Path(human) if human and Path(human).is_file() else xml_path
        if not path.is_file():
            checks.append(_check("XML parses", False, "no XML"))
        else:
            import xml.etree.ElementTree as ET
            try:
                ET.fromstring(path.read_text(encoding="utf-8"))
                checks.append(_check("XML parses", True))
            except ET.ParseError as exc:
                checks.append(_check("XML parses", False, str(exc)[:200]))

    elif stage in ("stage2c", "stage2d"):
        checks.append(_sequence_check(sample_id, stage, paths, block))

    elif stage == "stage3a":
        # Always re-checked rather than trusting `compiles_at_save`: that flag
        # records the single-frame render done at save time, which is the
        # weaker test (see `animation_renders`). The gate has to agree with
        # what export will actually do.
        human = block.get("human_animation_path")
        path = Path(human) if human and Path(human).is_file() else None
        if path is None:
            path = paths.animation_final(sample_id)
            if not path.is_file():
                path = paths.animation(sample_id)
        code = path.read_text(encoding="utf-8") if path.is_file() else None
        ok, detail = animation_renders(code, target, paths.compile_cache())
        checks.append(_check("animation exports", ok, detail))

    else:
        raise ValueError(f"no gate defined for stage {stage!r}")

    checks.append(_check("reviewer approved", bool(block.get("approved")),
                         "" if block.get("approved") else "not signed off"))

    blocking = [c["name"] for c in checks if not c["passed"]]
    return {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, stage),
        "passed": not blocking,
        "checks": checks,
        "blocking": blocking,
    }


def _sequence_check(sample_id: str, stage: str, paths: CachePaths,
                    block: dict) -> dict:
    """A sequence is done when it validates against the diagram with 0 problems."""
    from ..schema import AnimationSequence

    human = block.get("human_sequence_path")
    if human and Path(human).is_file():
        path = Path(human)
    elif stage == "stage2d":
        path = paths.sequence_final(sample_id)
    else:
        path = paths.sequence(sample_id)
    if not path.is_file():
        return _check("sequence validates", False, "no sequence")

    try:
        seq = AnimationSequence.load(path)
    except Exception as exc:                            # noqa: BLE001
        return _check("sequence validates", False, f"unreadable: {exc}"[:200])

    element_ids = _element_ids(paths.xml(sample_id), paths, sample_id)
    problems = seq.validate(element_ids)
    if problems:
        return _check("sequence validates", False,
                      f"{len(problems)} violation(s): {problems[0]}"[:200])
    return _check("sequence validates", True)


def _element_ids(xml_path: Path, paths: CachePaths | None = None,
                 sample_id: str | None = None) -> set[str] | None:
    """Ids a `focus` may name: the XML plus the code.

    Delegates to the planner's own helper rather than re-deriving the set with
    a local regex -- three copies of this had already drifted, and the one
    here missed that `text_node` elements live only in the code.
    """
    from ..planner.parser import referenceable_ids

    code = None
    if paths is not None and sample_id is not None:
        try:
            code = Path(paths.resolve_code(sample_id)).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            code = None
    return referenceable_ids(xml_path, code)


def blocking_for(stage_key: str, sample_id: str, paths: CachePaths, target: str,
                 *, code_for: dict | None = None) -> dict | None:
    """The unmet gate standing in front of `stage_key`, or None if clear.

    `stage_key` is a pipeline stage ("2c"), not a record key ("stage2c").
    """
    required = PREREQUISITE.get(stage_key)
    if not required:
        return None
    status = gate_status(sample_id, required, paths, target, code_for=code_for)
    return None if status["passed"] else status
