"""Annotation-tool test suite: 2c/3a overrides, gating, exporter, isolation.

Offline except where marked LIVE (those hit the running server on 8602, still
no model calls). Every test restores what it touched.
"""
import copy
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path

from img_2_svg_pretraining.pipeline.annotate import app as A, gates, store
from img_2_svg_pretraining.pipeline.cache import CachePaths, write_text
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.schema import AnimationSequence

CFG = "src/img_2_svg_pretraining/pipeline/configs/default.yaml"
GOOD = "CVPR_2025_pipe00041"     # animation exports cleanly
BAD = "m3grounder"               # known \pgffor export failure
SVG_SAMPLE = "CVPR_2025_pipe00002"
BASE = "http://localhost:8602"

A._init(CFG, None, None, ["progressive_reveal"])

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:160]))


def group(title):
    print()
    print("== %s ==" % title)


@contextmanager
def record_restored(sample_id):
    """Snapshot and restore one annotation record."""
    root = A.STATE["paths"].root
    before = copy.deepcopy(store.load(root, sample_id))

    def put(rec):
        rec.clear()
        rec.update(copy.deepcopy(before))
    try:
        yield
    finally:
        store.update(root, sample_id, put)


def http(method, path, body=None, timeout=900):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200].decode(errors="replace")}


# ---------------------------------------------------------------- swaps

group("swap isolation: a crash must not leak a human file")
with A._sample_context(GOOD) as paths:
    target = paths.animation_final(GOOD)
    original = target.read_bytes()
    human = paths.root / "_t_crash.tex"
    human.write_text("HUMAN")
    try:
        with A._swapped(human, target):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("swap", "original restored after an exception inside the block",
          target.read_bytes() == original)
    check("swap", "no .pre_human survives an exception",
          not Path(str(target) + ".pre_human").exists())
    human.unlink()

group("swap: nested swaps do not corrupt each other")
with A._sample_context(GOOD) as paths:
    seq_t, anim_t = paths.sequence_final(GOOD), paths.animation_final(GOOD)
    seq_o, anim_o = seq_t.read_bytes(), anim_t.read_bytes()
    h1 = paths.root / "_t_seq.json"
    h2 = paths.root / "_t_anim.tex"
    h1.write_text('{"nodes":[],"traversal":[]}')
    h2.write_text("ANIM")
    with A._swapped(h1, seq_t), A._swapped(h2, anim_t):
        check("swap", "both swapped simultaneously",
              seq_t.read_text().startswith("{") and anim_t.read_text() == "ANIM")
    check("swap", "both restored", seq_t.read_bytes() == seq_o
          and anim_t.read_bytes() == anim_o)
    h1.unlink()
    h2.unlink()

group("swap: skip flags match the stage that owns each path")
with A._sample_context(GOOD) as paths:
    with A._human_sequence_raw_applied(GOOD, paths, skip=True) as u:
        check("swap", "2c skipped when the sequencer is in the run", not u)
    with A._human_animation_applied(GOOD, paths, skip=True) as u:
        check("swap", "3a skipped when the animation critic is in the run", not u)

# ------------------------------------------------------------- 2c vs 2d

group("2c and 2d are genuinely different slots")
with record_restored(GOOD):
    with A._sample_context(GOOD) as paths:
        raw = paths.sequence(GOOD)
        final = paths.sequence_final(GOOD)
        raw_o, final_o = raw.read_bytes(), final.read_bytes()

        h = paths.root / "_t_2c.json"
        h.write_text(final.read_text())

        def m(rec):
            rec["stage2c"]["human_sequence_path"] = str(h)
        store.update(paths.root, GOOD, m)

        with A._human_sequence_raw_applied(GOOD, paths) as used:
            check("2c/2d", "2c override applies", used)
            check("2c/2d", "2c writes over sequence(), not sequence_final()",
                  raw.read_bytes() != raw_o and final.read_bytes() == final_o)
        check("2c/2d", "sequence() restored", raw.read_bytes() == raw_o)
        h.unlink()

group("2c save route targets its own directory")
with record_restored(GOOD):
    with A.app.test_client() as c:
        with A._sample_context(GOOD) as paths:
            text = paths.sequence_final(GOOD).read_text()
        r = c.post("/api/human-sequence/%s?stage=stage2c" % GOOD,
                   json={"text": text})
        check("2c/2d", "2c save returns ok", r.status_code == 200, r.status_code)
        if r.status_code == 200:
            p = r.get_json()["path"]
            check("2c/2d", "2c lands in sequence_raw_human/", "sequence_raw_human" in p, p)
            Path(p).unlink(missing_ok=True)

        r = c.post("/api/human-sequence/%s?stage=stage2d" % GOOD, json={"text": text})
        if r.status_code == 200:
            p = r.get_json()["path"]
            check("2c/2d", "2d lands in sequence_human/",
                  "sequence_human" in p and "raw" not in p, p)
            Path(p).unlink(missing_ok=True)

        r = c.post("/api/human-sequence/%s?stage=stage9z" % GOOD, json={"text": text})
        check("2c/2d", "unknown stage rejected with 400", r.status_code == 400,
              r.status_code)

# ------------------------------------------------------------ validation

group("save rejects malformed input, keeps salvageable input")
with record_restored(GOOD):
    with A.app.test_client() as c:
        r = c.post("/api/human-sequence/%s" % GOOD, json={"text": "{not json"})
        check("validate", "unparseable JSON rejected (400)", r.status_code == 400,
              r.status_code)

        r = c.post("/api/human-sequence/%s" % GOOD, json={"text": ""})
        check("validate", "empty body rejected (400)", r.status_code == 400,
              r.status_code)

        # Structurally valid but semantically broken: must still SAVE, because
        # repairing violations is the critic's job.
        broken = json.dumps({"style": "progressive_reveal",
                             "traversal_style": "OVERVIEW_FIRST",
                             "nodes": [{"id": "n1", "parent": None, "depth": 2,
                                        "focus": ["does_not_exist"]}],
                             "traversal": ["n1"]})
        r = c.post("/api/human-sequence/%s" % GOOD, json={"text": broken})
        ok = r.status_code == 200
        check("validate", "violating-but-valid sequence is still saved", ok,
              r.status_code)
        if ok:
            d = r.get_json()
            check("validate", "violations reported back to the reviewer",
                  len(d["validation"].get("problems", [])) > 0)
            Path(d["path"]).unlink(missing_ok=True)

group("empty animation code is refused")
with A.app.test_client() as c:
    r = c.post("/api/human-animation/%s" % GOOD, json={"text": "   "})
    check("validate", "empty animation rejected (400)", r.status_code == 400,
          r.status_code)

# ----------------------------------------------------------------- gates

group("gate: approval alone is not enough, and does not freeze a stage")
with record_restored(BAD):
    root = A.STATE["paths"].root

    def approve(rec):
        rec["stage3a"].update(approved=True, approved_at=store.now(),
                              approved_by="test")
    store.update(root, BAD, approve)
    with A._sample_context(BAD) as paths:
        g = gates.gate_status(BAD, "stage3a", paths, A._target_for(BAD))
    check("gates", "approved but non-exporting animation still blocked",
          not g["passed"] and "animation exports" in g["blocking"], g["blocking"])

group("gate: machine half is re-read, so a later edit re-closes it")
with record_restored(GOOD):
    root = A.STATE["paths"].root
    store.update(root, GOOD, lambda r: r["stage2c"].update(
        approved=True, approved_at=store.now(), approved_by="test"))
    with A._sample_context(GOOD) as paths:
        before = gates.gate_status(GOOD, "stage2c", paths, A._target_for(GOOD))
        h = paths.root / "_t_bad_seq.json"
        h.write_text(json.dumps({"style": "progressive_reveal",
                                 "traversal_style": "OVERVIEW_FIRST",
                                 "nodes": [{"id": "x", "parent": None, "depth": 3}],
                                 "traversal": ["x"]}))
        store.update(paths.root, GOOD, lambda r: r["stage2c"].update(
            human_sequence_path=str(h)))
        after = gates.gate_status(GOOD, "stage2c", paths, A._target_for(GOOD))
        check("gates", "a broken edit re-closes an approved gate",
              not after["passed"] and "sequence validates" in after["blocking"],
              "before=%s after=%s" % (before["passed"], after["blocking"]))
        h.unlink()

group("gate: prerequisites chain in pipeline order")
check("gates", "2c requires 2b", gates.PREREQUISITE["2c"] == "stage2b")
check("gates", "2d requires 2c", gates.PREREQUISITE["2d"] == "stage2c")
# 3a hangs off 2c, not 2d: gating the animation behind the sequence *critic*
# made "run without the critic" impossible, which was the point of having it.
check("gates", "3a requires 2c (so a critic-free run is reachable)",
      gates.PREREQUISITE["3a"] == "stage2c", gates.PREREQUISITE["3a"])
check("gates", "narration has no prerequisite at all",
      "2e" not in gates.PREREQUISITE)
check("gates", "3c requires 3a", gates.PREREQUISITE["3c"] == "stage3a")
check("gates", "narration is never a prerequisite",
      "stage2e" not in gates.PREREQUISITE.values())
check("gates", "narration has no gate", "stage2e" not in gates.STAGE_LABELS)
try:
    gates.gate_status(GOOD, "stage9z", A.STATE["paths"], "tikz")
    check("gates", "unknown stage raises rather than silently passing", False,
          "returned instead of raising")
except ValueError:
    check("gates", "unknown stage raises rather than silently passing", True)

group("gate: refusal names the blocking stage, not a generic failure")
r = A._gate_refusal(GOOD, ["3c"], False)
check("gates", "3c blocked by 3a", r and r["gate"]["stage"] == "stage3a",
      r and r["gate"]["stage"])
check("gates", "refusal carries actionable blocking list",
      r and isinstance(r["gate"]["blocking"], list) and r["gate"]["blocking"])
check("gates", "override bypasses every gate",
      A._gate_refusal(GOOD, ["3c"], True) is None)

group("gate: missing artifacts fail closed")
with record_restored(GOOD):
    with A._sample_context(GOOD) as paths:
        missing = CachePaths.from_config(load_config(Path(CFG)))
    ok, detail = gates.animation_renders(None, "tikz", missing.compile_cache())
    check("gates", "no code is a failure, not a pass", not ok and detail == "no code")
    ok, _ = gates._renders("", "tikz", missing.compile_cache())
    check("gates", "empty string is a failure", not ok)

# ------------------------------------------------------------- exporter

group("exporter: frames-only produces no video artifacts")
from img_2_svg_pretraining.pipeline.animator import exporter
from img_2_svg_pretraining.pipeline.runner import AgentContext
from img_2_svg_pretraining.pipeline.samples import discover_samples

cfg = load_config(Path(CFG))
cfg.raw["animation_style"] = "progressive_reveal"
cfg.style = "progressive_reveal"
ctx = AgentContext(cfg, "exporter")
samples = {s.id: s for s in discover_samples(cfg.dataset_root)}
out = ctx.paths.exports(GOOD)
shutil.rmtree(out, ignore_errors=True)
_, detail = exporter.export_sample(ctx, samples[GOOD], ["frames"], 2, 150)
frames = sorted((out / "frames").glob("*.png"))
check("exporter", "frames are produced", len(frames) > 0, detail)
for junk in ("animation.mp4", "animation.gif", "animation.pptx"):
    check("exporter", "no %s" % junk, not (out / junk).exists())
check("exporter", "pdf still produced (frames derive from it)",
      (out / "animation.pdf").exists())

group("exporter: mp4 path is untouched by the frames token")
check("exporter", "mp4 still in needs_frames set",
      "mp4" in {"frames", "mp4", "gif", "narrated_mp4"})
for conf, want in (("default.yaml", "animation.pdf"), ("svg.yaml", "animation.mp4")):
    c2 = load_config(Path("src/img_2_svg_pretraining/pipeline/configs/" + conf))
    f = list(c2.agents["exporter"].option("outputs", ["pdf", "mp4"]))
    marker = ("frames" if not ({"mp4", "gif", "pptx", "narrated_mp4"} & set(f))
              else ("animation.pdf" if c2.target == "tikz" else "animation.mp4"))
    check("exporter", "%s keeps marker %s" % (conf, want), marker == want, marker)

group("exporter: a failing sample raises rather than reporting success")
try:
    shutil.rmtree(ctx.paths.exports(BAD), ignore_errors=True)
    exporter.export_sample(ctx, samples[BAD], ["frames"], 2, 150)
    check("exporter", "m3grounder export fails loudly", False, "no exception")
except Exception as e:
    check("exporter", "m3grounder export fails loudly",
          "pgffor" in str(e) or "latexmk" in str(e), type(e).__name__)

# ----------------------------------------------------------------- store

group("store: migration is additive and idempotent")
p = Path("src/img_2_svg_pretraining/pipeline/cache/test_benchmark/annotations/"
         "%s.json" % SVG_SAMPLE)
raw = json.loads(p.read_text())
once = store._migrate(copy.deepcopy(raw), SVG_SAMPLE)
twice = store._migrate(copy.deepcopy(once), SVG_SAMPLE)
check("store", "migration is idempotent", once == twice)
check("store", "approval fields on every approvable stage",
      all("approved" in once[s] for s in store.APPROVABLE))
check("store", "narration block is not approvable", "stage2e" not in store.APPROVABLE)

group("store: a corrupt record degrades to empty rather than crashing")
tmp = Path(tempfile.mkdtemp())
(tmp / "annotations").mkdir()
(tmp / "annotations" / "x.json").write_text("{{{ not json")
rec = store.load(tmp, "x")
check("store", "unparseable record yields a fresh one",
      rec["sample_id"] == "x" and rec["stage3a"]["status"] == "unreviewed")
shutil.rmtree(tmp, ignore_errors=True)

group("bench conversion preserves step hierarchy")
BENCH = {"metadata": {"animation_style": "progressive_reveal",
                      "traversal_order": "OVERVIEW_FIRST"},
         "sequence": [
             {"timestamp": 1, "to_be_animated": {"blocks": [{"id": "a", "depth": 1}]}},
             {"timestamp": 2, "to_be_animated": {"nodes": [{"id": "a1", "depth": 2}]}},
             {"timestamp": 3, "to_be_animated": {"nodes": [{"id": "a2", "depth": 2}]}},
             {"timestamp": 4, "to_be_animated": {"nodes": [{"id": "deep", "depth": 3}]}},
             {"timestamp": 5, "to_be_animated": {"blocks": [{"id": "b", "depth": 1}]}},
             {"timestamp": 6, "to_be_animated": {"nodes": [{"id": "b1", "depth": 2}]}}]}
conv = AnimationSequence.from_dict(BENCH)
by = {n.id: n for n in conv.nodes}
# The bug: every step came back parent=None while keeping the element depth,
# so a nested step read as "a root at depth 2" and the gate refused it.
check("bench", "nested steps get a parent",
      by["t2"].parent == "t1" and by["t3"].parent == "t1",
      [(n.id, n.parent, n.depth) for n in conv.nodes])
check("bench", "deeper step nests under the most recent shallower one",
      by["t4"].parent == "t3", by["t4"].parent)
check("bench", "a new depth-1 step starts a new root",
      by["t5"].parent is None and by["t6"].parent == "t5")
check("bench", "hierarchy is preserved, not flattened",
      max(n.depth for n in conv.nodes) == 3,
      max(n.depth for n in conv.nodes))
check("bench", "converted sequence validates", not conv.validate(),
      conv.validate()[:2])

group("relink repairs an already-damaged sequence in place")
damaged = AnimationSequence.from_dict({
    "style": "progressive_reveal",
    "nodes": [{"id": "t1", "parent": None, "depth": 1},
              {"id": "t2", "parent": None, "depth": 2},
              {"id": "t3", "parent": None, "depth": 2}],
    "traversal": ["t1", "t2", "t3"]})
before = len(damaged.validate())
changed = damaged.relink_orphan_depths()
check("bench", "repair reports what it changed", changed == 2, changed)
check("bench", "repair clears the violations",
      before == 2 and not damaged.validate(), (before, damaged.validate()))
check("bench", "repair keeps the depths", damaged.nodes[1].depth == 2)
check("bench", "a healthy sequence is left alone",
      AnimationSequence.from_dict({
          "style": "s",
          "nodes": [{"id": "n1", "parent": None, "depth": 1}],
          "traversal": ["n1"]}).relink_orphan_depths() == 0)

group("focus validation spans the XML and the code")
from img_2_svg_pretraining.pipeline.planner.parser import (
    code_element_ids, load_element_ids, referenceable_ids)

with A._sample_context(GOOD) as paths:
    xml_p = paths.xml(GOOD)
    src = Path(paths.resolve_code(GOOD)).read_text(encoding="utf-8")
    xml_only = load_element_ids(xml_p)
    both = referenceable_ids(xml_p, src)

# The parser is told to exclude `text_node`; the sequencer is told the code is
# the exclusive source for them. Validating against the XML alone therefore
# flagged correct text references as absent from the diagram.
check("focus", "code contributes ids the XML omits", len(both) > len(xml_only),
      "%d vs %d" % (len(both), len(xml_only)))
check("focus", "the union is a superset of the XML", xml_only <= both)
# Not asserted by name prefix: samples name their labels `text_*`, `lbl_*` or
# `title_*` as the model saw fit. The property that matters is that ids marked
# `text_node` in the code -- which the parser is told to exclude -- are still
# recoverable from it.
_text_ids = set(re.findall(r"xml id=([A-Za-z0-9_]+)[^\n]*text_node", src))
check("focus", "text_node ids are recovered from the code",
      _text_ids and _text_ids <= code_element_ids(src),
      sorted(_text_ids)[:4])
check("focus", "and the XML genuinely lacks them",
      _text_ids and not (_text_ids & xml_only), sorted(_text_ids & xml_only)[:4])
check("focus", "no artifacts -> None, meaning 'cannot check'",
      referenceable_ids(Path("/nonexistent.xml"), None) is None)

with A._sample_context(GOOD) as paths:
    live = AnimationSequence.load(paths.sequence(GOOD))
check("focus", "a real sequence validates against the union",
      not live.validate(both), live.validate(both)[:1])
# Permissiveness would be its own bug: the check still has to catch bad ids.
live.nodes[0].focus.append("definitely_not_an_element")
bad = live.validate(both)
check("focus", "a fabricated id is still caught",
      any("definitely_not_an_element" in str(p) for p in bad), bad[:1])

# ------------------------------------------------------------ copy-block

group("copy-block: every agent, and what each carries")
with A.app.test_client() as c:
    for agent, must, mustnot in (
            ("sequencer", ["STRUCTURE XML", "DIAGRAM CODE"], []),
            ("designer", ["ANIMATION SEQUENCE"], []),
            ("code_converter", [], []),
            ("parser", [], [])):
        r = c.get("/api/copy-block/%s?agent=%s" % (GOOD, agent))
        ok = r.status_code == 200
        check("copy", "%s block builds" % agent, ok, r.status_code)
        if ok:
            t = r.get_json()["prompt"]
            check("copy", "%s prompt is substantial" % agent, len(t) > 500, len(t))
            for m in must:
                check("copy", "%s carries %s" % (agent, m), m in t)

    r = c.get("/api/copy-block/%s?agent=nonsense" % GOOD)
    check("copy", "unknown agent rejected (400)", r.status_code == 400, r.status_code)

group("copy-block: human edits are reflected in the copied prompt")
with record_restored(GOOD):
    with A.app.test_client() as c:
        base = c.get("/api/copy-block/%s?agent=sequencer" % GOOD).get_json()["prompt"]
        with A._sample_context(GOOD) as paths:
            h = paths.root / "_t_xml.xml"
            # Insert before the closing root tag, whatever it is called -- an
            # earlier version hardcoded "</diagram>" and the real tag is
            # "</Diagram>", so the edit silently never happened and the test
            # failed against correct code.
            src = paths.xml(GOOD).read_text().rstrip()
            close = src.splitlines()[-1]
            assert close.startswith("</"), close
            h.write_text(src[:src.rfind(close)]
                         + '<node id="SENTINEL_NODE" class="node" depth="1"/>\n'
                         + close)
            store.update(paths.root, GOOD,
                         lambda r: r["stage2b"].update(human_xml_path=str(h)))
        after = c.get("/api/copy-block/%s?agent=sequencer" % GOOD).get_json()["prompt"]
        check("copy", "a human XML edit changes the sequencer prompt", base != after)
        check("copy", "the edit itself appears in the prompt",
              "SENTINEL_NODE" in after)
        h.unlink()

group("stage 3: the designer reads the narration 2e wrote")
from img_2_svg_pretraining.pipeline.animator import designer as _designer
from img_2_svg_pretraining.pipeline.runner import AgentContext as _Ctx
from img_2_svg_pretraining.pipeline.samples import discover_samples as _discover

_samples = {x.id: x for x in _discover(A.STATE["cfg"].dataset_root)}
with A._sample_context(GOOD):
    _ctx = _Ctx(A.STATE["cfg"], "designer")
    _msgs = _designer.build_request(_ctx, _samples[GOOD])
    _text = "\n".join(m.text() for m in _msgs if m.text())
    _narrated = _ctx.paths.sequence_narrated(GOOD)
_in_prompt = len(re.findall(r'"narrative":\s*"[^"]', _text))
_on_disk = (sum(1 for n in json.loads(_narrated.read_text())["nodes"]
                if n.get("narrative")) if _narrated.is_file() else 0)
# Narration used to be written at 2e and then never read: the designer went
# straight to sequence_final, so the stage that turns a sequence into motion
# never saw the script written for it.
check("stage3", "designer receives every narrative on disk",
      _on_disk > 0 and _in_prompt == _on_disk, "%d in prompt vs %d on disk"
      % (_in_prompt, _on_disk))

group("stage 3: the screen shows what the run sends")
with A.app.test_client() as c:
    _st = c.get("/api/animation-state/%s" % GOOD).get_json()
check("stage3", "state names the file it read",
      _st.get("sequence_origin") == "sequence_narrated", _st.get("sequence_origin"))
check("stage3", "and that sequence carries the narration",
      sum(1 for n in json.loads(_st["sequence"])["nodes"]
          if n.get("narrative")) == _on_disk)
check("stage3", "diagram code is copyable", len(_st.get("diagram_code") or "") > 100)
check("stage3", "style block is copyable",
      "ANIMATION STYLE" in (_st.get("style_block") or ""))
# The copy block and the pipeline must not diverge -- a screen showing a
# different file than the run consumes is the bug this screen exists to avoid.
with A.app.test_client() as c:
    _blk = c.get("/api/copy-block/%s?agent=designer" % GOOD).get_json()
check("stage3", "copied prompt carries the same narration count",
      len(re.findall(r'"narrative":\s*"[^"]', _blk.get("prompt", ""))) == _on_disk)

with A.app.test_client() as c:
    check("stage3", "the page is served", c.get("/stage3/%s" % GOOD).status_code == 200)

group("a superseded critique is not planned against")
import os as _os
with A._sample_context(GOOD) as paths:
    _rev, _fin = paths.code_reviewed(GOOD), paths.code_final(GOOD)
    if _rev.is_file() and _fin.is_file():
        _rev_t, _fin_t = _rev.stat().st_mtime, _fin.stat().st_mtime
        # Critique newer than the splice -> the critique is the right input.
        _os.utime(_rev, (_fin_t + 60, _fin_t + 60))
        check("resolve", "a current critique is used",
              Path(paths.resolve_code(GOOD)).parent.parent.name == "code_reviewed",
              Path(paths.resolve_code(GOOD)).parent.parent.name)
        # Splice newer than the critique -> the critique describes a diagram
        # that no longer exists, and using it drops freshly spliced rasters.
        _os.utime(_rev, (_fin_t - 60, _fin_t - 60))
        check("resolve", "a superseded critique is skipped for the splice",
              Path(paths.resolve_code(GOOD)).parent.parent.name == "code_final",
              Path(paths.resolve_code(GOOD)).parent.parent.name)
        _os.utime(_rev, (_rev_t, _rev_t))       # restore
        check("resolve", "restored", abs(_rev.stat().st_mtime - _rev_t) < 1)
    else:
        check("resolve", "sample lacks both artifacts (skipped)", True)

group("the diagram critic is told never to un-splice a raster")
from img_2_svg_pretraining.pipeline.prompts import load_prompt as _lp
for _key in ("diagnose", "fix", "compile_fix"):
    _t = _lp("diagram_transmuter/critic.yaml#%s" % _key)
    if _key == "diagnose":
        # A crop looks different from the region it replaces; reporting that
        # led the fix stage to swap the image for a text label.
        check("resolve", "diagnose will not report a spliced raster",
              "Never report a spliced raster" in _t)
    else:
        # The rule wraps across lines in the YAML block scalar, so match on
        # collapsed whitespace rather than the literal one-line phrase -- an
        # earlier version of this failed against a prompt that was correct.
        _flat = " ".join(_t.split())
        check("resolve", "%s forbids replacing the command, not just the path" % _key,
              "the command, not just the path" in _flat
              and "Never replace an `\\includegraphics`" in _flat,
              _flat[_flat.find("Preserve every `\\includegraphics"):][:120])

group("the raster screen parses what the splice targeted")
from img_2_svg_pretraining.pipeline.transmuter.rasters import find_placeholders as _find

# The screen matches detections to placeholders by id. If it parses the raw
# 1a while the detections were computed against the spliced version, every id
# misses and the canvas draws nothing -- 7 ids vs 25, zero overlap, observed
# on pipe00011 after 1a was re-run post-splice.
_drew = _missed = 0
for _s in A.STATE["samples"]:
    with A._sample_context(_s.id) as _p:
        _det = _p.rasters(_s.id) / "detections.json"
        if not _det.is_file():
            continue
        _code, _ = A._effective_1a(_s.id)
        _tgt = A._target_for(_s.id)
    if not _code:
        continue
    _ids = {r["xml_id"] for r in json.loads(_det.read_text()).get("regions") or []}
    _ph = {x.xml_id for x in _find(_code, _tgt)}
    if _ids and _ph:
        if _ids & _ph:
            _drew += 1
        else:
            _missed += 1
# pipe00002's detections predate a converter rename (raster_node_N -> img_*),
# which no path choice can reconcile -- it needs re-detection, so one sample
# is allowed to miss.
check("rasters", "detections match the parsed placeholders",
      _drew >= 5 and _missed <= 1, "%d matched, %d with zero overlap" % (_drew, _missed))

with A.app.test_client() as c:
    _r = c.get("/api/rasters/CVPR_2025_pipe00011").get_json()
_boxes = _r.get("boxes") or []
check("rasters", "pipe00011 draws every box",
      len(_boxes) > 0 and all(b.get("human_bbox") for b in _boxes),
      "%d box(es), %d drawn" % (len(_boxes), sum(1 for b in _boxes if b.get("human_bbox"))))

group("the sample list reports completion")
with record_restored(GOOD):
    _root = A.STATE["paths"].root
    for _st in store.APPROVABLE:
        store.update(_root, GOOD, lambda r, st=_st: r[st].update(approved=False))
    with A.app.test_client() as c:
        _row = next(x for x in c.get("/api/samples").get_json()["samples"]
                    if x["id"] == GOOD)
    check("list", "an unapproved sample is not done", _row.get("all_approved") is False,
          _row.get("all_approved"))

    # Every gate signed off -> the sidebar colours the sample green.
    for _st in store.APPROVABLE:
        store.update(_root, GOOD, lambda r, st=_st: r[st].update(approved=True))
    with A.app.test_client() as c:
        _row = next(x for x in c.get("/api/samples").get_json()["samples"]
                    if x["id"] == GOOD)
    check("list", "all six approved -> done", _row.get("all_approved") is True)

    # One withdrawal is enough to un-finish it: partial approval is not done.
    store.update(_root, GOOD, lambda r: r["stage3a"].update(approved=False))
    with A.app.test_client() as c:
        _row = next(x for x in c.get("/api/samples").get_json()["samples"]
                    if x["id"] == GOOD)
    check("list", "withdrawing one approval clears done",
          _row.get("all_approved") is False)

group("approving takes a sample back off the scrap heap")
with record_restored(GOOD):
    store.update(A.STATE["paths"].root, GOOD,
                 lambda r: r.update(discarded={"by": "t", "at": "now", "reason": "test"}))
    with A.app.test_client() as c:
        _row = next(x for x in c.get("/api/samples").get_json()["samples"]
                    if x["id"] == GOOD)
        check("list", "a discarded sample says so", _row.get("discarded") is True)
        # Otherwise a re-reviewed sample reads as discarded *and* complete,
        # and the list has to arbitrate between two contradictory flags.
        c.post("/api/approve/%s" % GOOD,
               json={"stage": "stage1a", "approved": True, "author": "t"})
        _row = next(x for x in c.get("/api/samples").get_json()["samples"]
                    if x["id"] == GOOD)
    check("list", "approving clears the discard", _row.get("discarded") is False,
          _row.get("discarded"))

# ---------------------------------------------------------------- LIVE

group("LIVE: pages and endpoints over HTTP")
for path in ("/", "/sequence/%s" % GOOD, "/animation/%s" % GOOD, "/xml/%s" % GOOD):
    try:
        with urllib.request.urlopen(BASE + path, timeout=120) as r:
            check("live", "GET %s" % path, r.status == 200, r.status)
    except Exception as e:
        check("live", "GET %s" % path, False, e)

status, body = http("GET", "/api/workflow-state/%s" % GOOD)
check("live", "workflow-state returns every gate",
      status == 200 and set(body.get("gates", {})) == set(gates.STAGE_LABELS), status)
check("live", "workflow-state names the next open stage",
      status == 200 and "next_stage" in body)

status, body = http("POST", "/api/run-stage2/%s" % BAD, {"stages": ["3c"]})
check("live", "blocked run refused with 409", status == 409, status)
check("live", "409 body explains which gate", status == 409 and "error" in body,
      body.get("error"))

status, body = http("GET", "/api/next-sample")
check("live", "next-sample answers", status == 200 and "id" in body, status)

group("LIVE: approve round-trip leaves the record as it started")
before = store.load(A.STATE["paths"].root, GOOD)["stage2b"]["approved"]
http("POST", "/api/approve/%s" % GOOD, {"stage": "stage2b", "approved": True,
                                        "author": "test"})
s1, b1 = http("GET", "/api/workflow-state/%s" % GOOD)
check("live", "approve flips the gate's human half",
      b1["gates"]["stage2b"]["checks"][-1]["passed"])
http("POST", "/api/approve/%s" % GOOD, {"stage": "stage2b", "approved": bool(before)})
s2, b2 = http("GET", "/api/workflow-state/%s" % GOOD)
check("live", "un-approve restores prior state",
      b2["gates"]["stage2b"]["checks"][-1]["passed"] == bool(before))

status, body = http("POST", "/api/approve/%s" % GOOD, {"stage": "stage2e"})
check("live", "narration cannot be approved (400)", status == 400, status)

# --------------------------------------------------------------- hygiene

group("file hygiene")
root = A.STATE["paths"].root
strays = list(root.rglob("*.pre_human"))
check("hygiene", "no .pre_human anywhere", not strays, strays[:3])
leftovers = [p.name for p in root.glob("_t_*")]
check("hygiene", "no test scratch files left", not leftovers, leftovers)
subtrees = sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name.endswith("_human"))
check("hygiene", "human subtrees are lineage-keyed dirs",
      all(any(x.is_dir() for x in (root / s).iterdir())
          for s in subtrees if any((root / s).iterdir())), subtrees)

# ---------------------------------------------------------------- report

print()
print("=" * 62)
groups = {}
for g, _n, ok_, _d in _results:
    a, b = groups.get(g, (0, 0))
    groups[g] = (a + int(ok_), b + 1)
for g, (a, b) in groups.items():
    print("  %-10s %d/%d" % (g, a, b))
passed = sum(1 for _g, _n, ok_, _d in _results if ok_)
total = len(_results)
print("=" * 62)
print("RESULT: %d passed, %d failed (of %d)" % (passed, total - passed, total))
for g, n, ok_, d in _results:
    if not ok_:
        print("  FAILED [%s] %s   %s" % (g, n, str(d)[:200]))
sys.exit(0 if passed == total else 1)
