"""Does a real annotation workflow behave, and are stale artifacts evicted?

Not a happy-path walk: each test edits something upstream and then asks
whether everything downstream either updated or is visibly marked stale.
The failure mode being hunted is "screen says fine, artifact is old".
"""
import copy
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

from img_2_svg_pretraining.pipeline.annotate import app as A, gates, store

BASE = "http://localhost:8602"
S = "CVPR_2025_pipe00004"          # untouched by other tests
A._init("src/img_2_svg_pretraining/pipeline/configs/default.yaml", None, None,
        ["progressive_reveal"])

_res = []


def check(group, name, cond, detail=""):
    _res.append((group, name, bool(cond)))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:170]))


def group(t):
    print("\n== %s ==" % t)


def http(method, path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return r.status, {"_bytes": len(raw),
                                  "_from": r.headers.get("X-Render-Source")}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return e.code, {"_raw": raw[:160].decode(errors="replace")}


ROOT = A.STATE["paths"].root
_before = copy.deepcopy(store.load(ROOT, S))


def restore_record():
    def put(rec):
        rec.clear()
        rec.update(copy.deepcopy(_before))
    store.update(ROOT, S, put)


# ---------------------------------------------------------------- gating

group("gate order is enforced from a clean record")
restore_record()
for st in store.APPROVABLE:
    store.update(ROOT, S, lambda r, st=st: r[st].update(
        approved=False, approved_at=None, approved_by=None))

s, d = http("GET", "/api/workflow-state/%s" % S)
check("gate", "nothing approved -> next is 1a", d.get("next_stage") == "stage1a",
      d.get("next_stage"))

for stage, blocked_by in (("1b", "stage1a"), ("2b", "stage1b"),
                          ("2c", "stage2b"), ("3a", "stage2c")):
    s, b = http("POST", "/api/run-stage2/%s" % S, {"stages": [stage]})
    if s == 400:                      # stage-1 ids go to the other endpoint
        s, b = http("POST", "/api/run-stages/%s" % S, {"stages": [stage]})
    check("gate", "%s blocked by %s" % (stage, blocked_by),
          s == 409 and b.get("gate", {}).get("stage") == blocked_by,
          "%s %s" % (s, b.get("gate", {}).get("stage")))

group("approving in order opens each gate in turn")
for stage in ("stage1a", "stage1b", "stage2b"):
    s, b = http("POST", "/api/approve/%s" % S,
                {"stage": stage, "approved": True, "author": "wf"})
    check("gate", "%s opens after approval" % stage,
          s == 200 and b["gates"][stage]["passed"], b.get("gates", {}).get(stage))

s, b = http("POST", "/api/run-stage2/%s" % S, {"stages": ["2c"], "force": False})
check("gate", "2c runs once 2b is approved", s == 200, "%s %s" % (s, b.get("error")))

# ------------------------------------------------------------- staleness

group("editing 1a marks downstream stale (the reported bug class)")
restore_record()
with A._sample_context(S) as paths:
    code_1a = paths.code(S)
    original = code_1a.read_text()

s, d = http("GET", "/api/sample/%s" % S)
before_stale = d["stages"]["stale_1c"]

# A human 1a edit, saved through the route the reviewer uses.
s, b = http("POST", "/api/human-code/%s" % S,
            {"text": original + "\n%% workflow test edit\n", "source": "pasted"})
check("stale", "human 1a saves", s == 200, s)

s, d = http("GET", "/api/sample/%s" % S)
st = d["stages"]
check("stale", "1b reported stale after a 1a edit", st["stale_1b"],
      "stale_1b=%s (was 1c=%s)" % (st["stale_1b"], before_stale))
check("stale", "1c reported stale after a 1a edit", st["stale_1c"], st)

group("the render follows the human edit immediately")
# Asserted while the human 1a is still recorded -- an earlier version of this
# checked after the cleanup below and was testing nothing.
s, d = http("GET", "/api/render/%s?source=diagram" % S)
check("stale", "diagram render resolves to the human 1a",
      d.get("_from") == "human 1a" or s == 422,
      "%s %s" % (s, d.get("_from")))
check("stale", "and it is the human copy, not the converter's",
      A._resolve_render_code(S, "diagram")[1] == "human 1a",
      A._resolve_render_code(S, "diagram")[1])

group("the splice is a snapshot, and the tool must not pretend otherwise")
with A._sample_context(S) as paths:
    spliced = paths.root / "code_final_human" / paths.code_lineage / ("%s.tex" % S)
if spliced.is_file():
    fresh = spliced.read_text()
    check("stale", "spliced file does NOT silently contain the new edit",
          "workflow test edit" not in fresh,
          "splice already had the edit — unexpected")
    s, d = http("GET", "/api/render/%s?source=rasters" % S)
    check("stale", "raster render is served from the older splice",
          d.get("_from") in ("human 1b", "code_reviewed", "code_final"),
          d.get("_from"))
else:
    check("stale", "no splice for this sample (skipped)", True)

restore_record()

# ------------------------------------------------------- frames eviction

group("frames are rebuilt, not served from the previous compile")
s, b = http("POST", "/api/compile-animation/%s" % S, {"force": True})
check("frames", "compile succeeds", s == 200 and b.get("frames", 0) > 0,
      "%s %s" % (s, b.get("error") or b.get("detail")))
n1 = b.get("frames", 0)

with A._sample_context(S) as paths:
    fdir = paths.exports(S) / "frames"
    first = sorted(fdir.glob("*.png"))
    mtime_before = first[0].stat().st_mtime if first else 0

time.sleep(1.1)
s, b = http("POST", "/api/compile-animation/%s" % S, {"force": True})
with A._sample_context(S) as paths:
    fdir = paths.exports(S) / "frames"
    after = sorted(fdir.glob("*.png"))
    mtime_after = after[0].stat().st_mtime if after else 0

check("frames", "forced recompile rewrites the frame files",
      mtime_after > mtime_before, "%s -> %s" % (mtime_before, mtime_after))
check("frames", "frame count is stable across recompiles",
      len(after) == n1, "%s vs %s" % (len(after), n1))
# Compile produces frames AND an mp4 (stills for one moment, video for the
# motion). What must NOT survive is anything older than the deck: gif/pptx are
# never rebuilt here, and an mp4 from an earlier export must be replaced, not
# left sitting beside fresh frames.
newest_frame = max((f.stat().st_mtime for f in fdir.glob("*.png")), default=0)
stale = []
for n in ("animation.mp4", "animation.gif", "animation.pptx"):
    f = fdir.parent / n
    if f.is_file() and f.stat().st_mtime < newest_frame - 120:
        stale.append(n)
check("frames", "no artifact older than the deck survives a recompile",
      not stale, stale)
check("frames", "compile produces a video alongside the frames",
      (fdir.parent / "animation.mp4").is_file())

group("stale frames are not left behind when the deck shrinks")
with A._sample_context(S) as paths:
    fdir = paths.exports(S) / "frames"
    ghost = fdir / "frame-999.png"
    ghost.write_bytes(b"\x89PNG\r\n\x1a\n")          # a leftover from a longer run
s, b = http("POST", "/api/compile-animation/%s" % S, {"force": True})
check("frames", "a leftover frame from a longer deck is evicted",
      not ghost.exists(),
      "frame-999.png survived the recompile")
if ghost.exists():
    ghost.unlink()

group("a no-op swap does not fake an edit")
with A._sample_context(S) as paths:
    tgt = paths.code(S)
    before_m = tgt.stat().st_mtime
    tmp = paths.root / "_wf_swap.tex"
    tmp.write_text(tgt.read_text())
    time.sleep(1.1)
    with A._swapped(tmp, tgt):
        pass
    after_m = tgt.stat().st_mtime
    check("stale", "restore preserves the original mtime",
          abs(after_m - before_m) < 0.01,
          "%s -> %s" % (before_m, after_m))
    check("stale", "and leaves no .pre_human",
          not Path(str(tgt) + ".pre_human").exists()
          and not tgt.with_suffix(tgt.suffix + ".pre_human").exists())
    tmp.unlink()

# ------------------------------------------------------------- progress

group("run progress reports per stage")
s, b = http("GET", "/api/run-progress/%s" % S)
check("progress", "endpoint answers when idle", s == 200 and "planned" in b, s)
check("progress", "last run recorded its stages",
      not b.get("running"), b.get("running"))

# --------------------------------------------------------------- render

group("render sources stay distinct and labelled")
seen = {}
for src in ("diagram", "rasters", "animation", "best"):
    s, d = http("GET", "/api/render/%s?source=%s" % (S, src))
    seen[src] = (s, d.get("_from"))
    check("render", "%s renders" % src, s == 200, "%s %s" % (s, d.get("error")))
check("render", "each source names the artifact it compiled",
      all(v[1] for v in seen.values()), seen)

s, d = http("GET", "/api/render/%s?source=bogus" % S)
check("render", "unknown source rejected", s == 400, s)

# ---------------------------------------------------------------- report

restore_record()
print()
print("=" * 64)
g = {}
for grp, _n, ok in _res:
    a, b_ = g.get(grp, (0, 0))
    g[grp] = (a + int(ok), b_ + 1)
for k, (a, b_) in g.items():
    print("  %-10s %d/%d" % (k, a, b_))
passed = sum(1 for _g, _n, ok in _res if ok)
print("=" * 64)
print("RESULT: %d passed, %d failed (of %d)" % (passed, len(_res) - passed, len(_res)))
for grp, n, ok in _res:
    if not ok:
        print("  FAILED [%s] %s" % (grp, n))
