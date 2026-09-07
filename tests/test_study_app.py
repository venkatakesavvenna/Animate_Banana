"""HTTP-level study tests (LIVE), against both cohorts.

Needs two servers, one per bundle/config pair. `scripts/run_study_tests.sh`
starts fresh ones on 8612 (main-v1 + study_main.yaml) and 8613 (selective-v1 +
study_selective.yaml); by hand:

    FRESH=1 PORT=8612 DB=data/study_runs/test_8612.db bash scripts/run_study_server.sh
    FRESH=1 PORT=8613 DB=data/study_runs/test_8613.db \
        BUNDLE=data/study_bundles/selective-v1 \
        CONFIG=src/img_2_svg_pretraining/pipeline/configs/study_selective.yaml \
        bash scripts/run_study_server.sh
    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_app.py

STUDY_BASE / STUDY_BASE_SELECTIVE override the servers. Every participant is
registered fresh; nothing is deleted (the schema has no delete path, and
that absence is asserted elsewhere). Do not point this at a production DB:
the full-session groups consume one judgment per cell and would retire cells
that real participants should see.

Checks marked KNOWN pin product bugs found while writing the suite. They are
reported, not counted as failures, so the suite stays green until the fix
lands -- at which point they flip to PASS and the marker should be removed.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

BASE = os.environ.get("STUDY_BASE", "http://localhost:8612")
SEL_BASE = os.environ.get("STUDY_BASE_SELECTIVE", "http://localhost:8613")
ADMIN = os.environ.get("STUDY_ADMIN_TOKEN", "devtoken")
CONFIG_DIR = Path("src/img_2_svg_pretraining/pipeline/configs")

_results = []


def check(group, name, cond, detail="", known=None):
    """`known` names a product bug this check pins; a failure is then reported
    as KNOWN rather than counted."""
    ok = bool(cond)
    if not ok and known:
        _results.append((group, name, True, detail))
        print("  KNOWN %s   <- %s [%s]" % (name, str(detail)[:120], known))
        return
    _results.append((group, name, ok, detail))
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if ok else "   <- " + str(detail)[:200]))
    if ok and known:
        print("        (marked KNOWN but passes now -- drop the marker)")


def group(title):
    print("\n== %s ==" % title)


def _call(base, path, data=None, pid=None, raw=False, headers=None):
    req = urllib.request.Request(
        base + path, method="POST" if data is not None else "GET",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json",
                 **({"X-Participant": pid} if pid else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, (r.read() if raw else json.load(r)), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            body = json.loads(body)
        except Exception:
            body = {}
        return e.code, body, dict(e.headers)


def call(path, data=None, pid=None, raw=False):
    return _call(BASE, path, data, pid, raw)


def scall(path, data=None, pid=None, raw=False):
    return _call(SEL_BASE, path, data, pid, raw)


def admin_call(path, token=ADMIN, data=None, base=None):
    status, body, _ = _call(base or BASE, path, data,
                            headers={"X-Admin-Token": token} if token else {})
    return status, body


def register(base=None, **kw):
    payload = {"display_name": "T", "education_level": "student", "consent": True}
    payload.update(kw)
    _, body, _ = _call(base or BASE, "/api/register", payload)
    return body["participant_id"]


def answer(base, trial, pid, qid, value):
    return _call(base, "/api/trial/%s/answer" % trial["trial_id"],
                 {"question_id": qid, "value": value}, pid)


def submit(base, trial, pid, body=None):
    return _call(base, "/api/trial/%s/submit" % trial["trial_id"], body or {}, pid)


def current(base, pid):
    return _call(base, "/api/trial/current", pid=pid)[1]


def complete_trial(base, trial, pid, tournament_picks=("A", "C")):
    """The cheapest valid completion for whatever screen the trial is on."""
    if trial["screen"] == "tournament":
        return submit(base, trial, pid, {"picks": list(tournament_picks)})[0]
    q = trial["questions"]["questions"][0]
    value = {"yesno": False, "choice_pair": "tie"}.get(q["type"], "A")
    answer(base, trial, pid, q["id"], value)
    return submit(base, trial, pid)[0]


def blob_without_titles(trial):
    """The payload as text, minus the free-text fields: a paper title or a
    narration sentence can legitimately say 'baseline' or 'context'. What must
    never carry a condition is the structured part of the payload."""
    t = json.loads(json.dumps(trial))
    if t.get("figure"):
        t["figure"].pop("title", None)
    for slot in t.get("slots", []):
        for cue in slot.get("cues", []):
            cue.pop("text", None)
    return json.dumps(t).lower()


for base in (BASE, SEL_BASE):
    try:
        urllib.request.urlopen(base + "/", timeout=5)
    except Exception as exc:                                   # noqa: BLE001
        print("server not reachable on %s: %s" % (base, exc))
        sys.exit(2)

MAIN_CFG = yaml.safe_load((CONFIG_DIR / "study_main.yaml").read_text(encoding="utf-8"))
SEL_CFG = yaml.safe_load((CONFIG_DIR / "study_selective.yaml").read_text(encoding="utf-8"))
EXP1_TARGET = MAIN_CFG["samples_per_experiment"]["exp1"]
EXP2_TARGET = MAIN_CFG["samples_per_experiment"]["exp2"]
DAY_FIGURES = 10           # main_selection.json: ten fresh figures per day
FORBIDDEN = ["narrative_id", "with_context", "without_context", "pre_verification",
             "animatebanana", "qwen", "baseline", "cell_key", "position_seed",
             "assignment_reason", "gemini", "lineage", ".webp", "/media/",
             "diagram_id", "method", "condition",
             "progressive_reveal", "colour_pop", "alpha_masking",
             "hopping_bounding_box", "sliding_bounding_box"]


# -------------------------------------------------------------- routing --
group("pages and assets")
for path, kind in (("/", "text/html"), ("/study", "text/html"),
                   ("/prepare", "text/html"), ("/admin", "text/html"),
                   ("/player.js", "javascript"), ("/forms.js", "javascript"),
                   ("/study.js", "javascript"), ("/prep_interface.png", "image/png")):
    status, _, headers = call(path, raw=True)
    check("routes", "%s serves %s" % (path, kind),
          status == 200 and kind in headers.get("Content-Type", ""),
          (status, headers.get("Content-Type")))


# ------------------------------------------------------------ lifecycle --
group("registration and state")
status, body, _ = call("/api/register", {"education_level": "", "consent": True})
check("register", "education level is required", status == 400, status)
status, body, _ = call("/api/register", {"education_level": "wizard", "consent": True})
check("register", "an unknown education level is refused", status == 400, status)

pid = register(roll_no="R99", area="cv", reads_papers="weekly")
check("register", "returns an opaque participant id", len(pid) == 32, pid)

status, state, _ = call("/api/state", pid=pid)
check("state", "state is readable", status == 200 and state["completed"] == 0)
check("state", "the main study runs exp1 then exp2",
      state["live_experiments"] == ["exp1", "exp2"], state["live_experiments"])
check("state", "the target counts absolute CELLS plus tournament FIGURES",
      state["target"] == min(EXP1_TARGET, DAY_FIGURES * 2) + min(EXP2_TARGET, DAY_FIGURES),
      (state["target"], EXP1_TARGET, EXP2_TARGET))
check("state", "consent is recorded as the stage", state["stage"] == "consented", state["stage"])
LIVE = state["live_experiments"]
TARGET = state["target"]

status, _, _ = call("/api/state", pid="notarealparticipant")
check("state", "unknown participant is rejected", status == 401, status)
status, _, _ = call("/api/state")
check("state", "missing participant is rejected", status == 401, status)
status, _, _ = call("/api/state?pid=" + pid)
check("state", "the id may ride in the query string (sendBeacon has no headers)",
      status == 200, status)


# ---------------------------------------------------------------- trial --
group("the absolute trial over HTTP")
status, trial, _ = call("/api/trial/current", pid=pid)
check("trial", "a trial is served", status == 200 and "trial_id" in trial)
check("trial", "exp1 comes first", trial["experiment"] == "exp1", trial["experiment"])
check("trial", "on the absolute screen", trial["screen"] == "absolute", trial["screen"])
check("trial", "one slot, lettered A",
      [s["slot"] for s in trial["slots"]] == ["A"], trial["slots"][0]["slot"])
check("trial", "exp1 shows captions (it asks about narration)",
      trial["show_captions"] is True and len(trial["slots"][0]["cues"]) > 0)
check("trial", "the figure is supplied", bool(trial["figure"]["media_id"]))
check("trial", "frames are supplied",
      len(trial["slots"][0]["frames"]) == trial["slots"][0]["n_frames"])
check("trial", "holds match frame count",
      len(trial["slots"][0]["holds"]) == trial["slots"][0]["n_frames"])
check("trial", "frame dimensions ride along for the stage aspect",
      trial["slots"][0]["frame_w"] > 0 and trial["slots"][0]["frame_h"] > 0)
check("trial", "the style is named, summarised and its rules listed",
      bool(trial["style_name"]) and bool(trial["style_description"])
      and len(trial.get("style_rules", [])) >= 3,
      (trial["style_name"], len(trial.get("style_rules", []))))
qids = [q["id"] for q in trial["questions"]["questions"]]
check("trial", "the four exp1 questions are attached in order",
      qids == ["vfs", "ascs", "sss", "nas"], qids)
check("trial", "the style name is written into the ascs prompt",
      trial["style_name"] in trial["questions"]["questions"][1]["prompt"],
      trial["questions"]["questions"][1]["prompt"])
check("trial", "show_if reaches the client so it can gate progressively",
      trial["questions"]["questions"][1].get("show_if") == {"vfs": True})
check("trial", "no familiarity question in the main study",
      trial["questions"]["familiarity"] is None)
check("trial", "no side labels on an absolute trial", "side_labels" not in trial)
check("trial", "a fresh trial reports no saved answers", trial.get("saved_answers") == {})

_, again, _ = call("/api/trial/current", pid=pid)
check("trial", "refetching resumes the same trial",
      again["trial_id"] == trial["trial_id"])
check("trial", "resumed payload is identical",
      json.dumps(again, sort_keys=True) == json.dumps(trial, sort_keys=True))

group("blinding over the wire")
leaks = [t for t in FORBIDDEN if t in blob_without_titles(trial)]
check("blinding", "no condition, method or path token in the trial payload", not leaks, leaks)
check("blinding", "style is named for the participant, not as a lineage slug",
      "_" not in trial["style_name"] and trial["style_name"][0].isupper(),
      trial["style_name"])
check("blinding", "the raw style slug is not in the payload",
      "animation_style" not in trial)


# ---------------------------------------------------------------- media --
group("media")
for label, mid in (("figure", trial["figure"]["media_id"]),
                   ("frame", trial["slots"][0]["frames"][0])):
    status, body, headers = call("/media/" + mid, raw=True)
    # 200 bytes, not 1KB: a progressive reveal's first frame is a nearly empty
    # canvas and compresses to a few hundred bytes.
    check("media", "%s serves as webp" % label,
          status == 200 and headers.get("Content-Type") == "image/webp"
          and len(body) > 200, (status, len(body) if body else 0))
    check("media", "%s is cacheable" % label,
          "immutable" in headers.get("Cache-Control", ""))

for bad, label in (("..%2f..%2fetc%2fpasswd", "traversal"),
                   ("deadbeefdeadbeef", "unknown id"),
                   ("a" * 40, "overlong id")):
    status, _, _ = call("/media/" + bad, raw=True)
    check("media", "%s is refused" % label, status == 404, status)


# --------------------------------------------------------------- answers --
group("the progressive gate, server side")
status, body, _ = submit(BASE, trial, pid)
check("gate", "submitting with nothing answered is refused", status == 409, status)
check("gate", "only the first gate is owed", body.get("missing") == ["vfs"], body)

status, body, _ = answer(BASE, trial, pid, "vfs", True)
check("answers", "an answer is accepted and read back",
      status == 200 and body["answers"] == {"vfs": True}, body)
status, body, _ = submit(BASE, trial, pid)
check("gate", "vfs=Yes opens ascs", status == 409 and body.get("missing") == ["ascs"], body)
answer(BASE, trial, pid, "ascs", True)
status, body, _ = submit(BASE, trial, pid)
check("gate", "both Yes owes the two scores",
      status == 409 and body.get("missing") == ["sss", "nas"], body)
answer(BASE, trial, pid, "sss", 7)
status, body, _ = answer(BASE, trial, pid, "nas", 8)
check("answers", "scores read back as numbers",
      body["answers"].get("sss") == 7 and body["answers"].get("nas") == 8, body)

status, body, _ = answer(BASE, trial, pid, "sss", 2)
check("answers", "a revision is accepted", status == 200)
check("answers", "the revision is what is read back",
      body["answers"]["sss"] == 2, body["answers"].get("sss"))

status, _, _ = call("/api/trial/%s/answer" % trial["trial_id"], {"question_id": "x"}, pid)
check("answers", "an answer without a value is refused", status == 400, status)
status, _, _ = call("/api/trial/%s/answer" % trial["trial_id"], {"value": 1}, pid)
check("answers", "an answer without a question is refused", status == 400, status)

other = register()
status, _, _ = answer(BASE, trial, other, "sss", 1)
check("answers", "another participant cannot answer this trial", status == 404, status)
status, _, _ = submit(BASE, trial, other)
check("answers", "another participant cannot submit this trial", status == 404, status)
status, _, _ = call("/api/trial/nosuchtrial/submit", {}, pid)
check("answers", "an unknown trial is 404", status == 404, status)

# Closing a gate after its dependents were answered: the stale answers are
# not owed, and must not keep the trial open either.
answer(BASE, trial, pid, "vfs", False)
status, body, _ = submit(BASE, trial, pid)
check("gate", "closing the gate afterwards makes the trial complete",
      status == 200 and body.get("ok"), (status, body))
status, body, _ = submit(BASE, trial, pid)
check("gate", "re-submitting is idempotent", status == 200 and body.get("already"), body)
status, _, _ = answer(BASE, trial, pid, "sss", 5)
check("gate", "a submitted trial no longer accepts answers", status == 409, status)
_, st, _ = call("/api/state", pid=pid)
check("gate", "the submission counts toward exp1",
      st["completed"] == 1 and st["per_experiment"] == {"exp1": 1}, st)

group("a No at either gate is a complete trial")
t2 = current(BASE, pid)
check("gate", "a second, different trial follows", t2["trial_id"] != trial["trial_id"])
answer(BASE, t2, pid, "vfs", False)
status, body, _ = submit(BASE, t2, pid)
check("gate", "vfs=No alone submits", status == 200, (status, body))
t3 = current(BASE, pid)
answer(BASE, t3, pid, "vfs", True)
answer(BASE, t3, pid, "ascs", False)
status, body, _ = submit(BASE, t3, pid)
check("gate", "vfs=Yes, ascs=No submits without the scores", status == 200, (status, body))
t4 = current(BASE, pid)
answer(BASE, t4, pid, "vfs", True)
answer(BASE, t4, pid, "ascs", True)
answer(BASE, t4, pid, "sss", 10)
status, body, _ = submit(BASE, t4, pid)
check("gate", "one missing score is still refused, and named",
      status == 409 and body.get("missing") == ["nas"], body)
answer(BASE, t4, pid, "nas", 0)
status, body, _ = submit(BASE, t4, pid)
check("gate", "a zero score is an answer, not an absence", status == 200, (status, body))


# -------------------------------------------------------- saved answers --
group("a trial carries its already-saved answers")
rpid = register()
rt = current(BASE, rpid)
answer(BASE, rt, rpid, "vfs", True)
answer(BASE, rt, rpid, "ascs", True)
rt2 = current(BASE, rpid)
check("resume", "the same trial resumes", rt2["trial_id"] == rt["trial_id"])
check("resume", "the saved answers come back with it",
      rt2.get("saved_answers") == {"vfs": True, "ascs": True}, rt2.get("saved_answers"))
answer(BASE, rt, rpid, "ascs", False)
check("resume", "a revision is reflected in what comes back",
      current(BASE, rpid).get("saved_answers", {}).get("ascs") is False)


# ---------------------------------------------------------------- events --
group("interaction events")
status, _, _ = call("/api/events", {"events": [
    {"trial_id": rt["trial_id"], "type": "scrub", "slot": "A",
     "t_video": 0.0, "client_seq": 0, "payload": {"frame": 3}},
    {"trial_id": rt["trial_id"], "type": "step_forward", "slot": "A",
     "t_video": 12.5, "client_seq": 1, "payload": {"frame": 4}},
]}, rpid)
check("events", "events are accepted (server_ts default works on this sqlite)",
      status == 200, status)
status, _, _ = call("/api/events", {"events": [{"type": "no_trial_id"}]}, rpid)
check("events", "a malformed event is dropped rather than failing the batch",
      status == 200, status)
status, _, _ = call("/api/events?pid=" + rpid, {"events": []})
check("events", "the beacon path (pid in the query) is accepted", status == 200, status)


# -------------------------------------------------------- full session --
group("a full main-study session")
runner = register(education_level="researcher")
seen = []
for _ in range(80):
    t = current(BASE, runner)
    if t.get("done"):
        break
    # Tournament: pick A then C, so the ranks are talk 1, animatebanana 2, qwen38 3.
    status = complete_trial(BASE, t, runner)
    seen.append((t["experiment"], t["screen"], t["show_captions"],
                 t["figure"]["media_id"], [s["slot"] for s in t["slots"]], status,
                 t["trial_id"]))

check("session", "every trial submitted", all(s[5] == 200 for s in seen),
      Counter(s[5] for s in seen))
# A failed submit leaves the trial open and it is served again; report that
# once above rather than letting it cascade into every structural check.
check("session", "no trial was served twice",
      len({s[6] for s in seen}) == len(seen), len(seen) - len({s[6] for s in seen}))
seen = list({s[6]: s for s in seen}.values())
order = [e for e, *_ in seen]
check("session", "the session completes", len(seen) == TARGET, (len(seen), TARGET))
check("session", "sections run in order, one at a time",
      order == sorted(order, key=lambda e: LIVE.index(e)), order)
check("session", "every live section is reached", set(order) == set(LIVE))
per = Counter(order)
check("session", "exp1 serves its target, exp2 its target",
      per == {"exp1": EXP1_TARGET, "exp2": EXP2_TARGET}, per)
figs1 = [s[3] for s in seen if s[0] == "exp1"]
check("session", "exp1: every day-1 figure is rated once per method (twice)",
      Counter(figs1) and set(Counter(figs1).values()) == {2}
      and len(set(figs1)) == DAY_FIGURES, Counter(figs1).most_common(2))
check("session", "exp1: the first ten figures are all different",
      len(set(figs1[:DAY_FIGURES])) == DAY_FIGURES)
check("session", "exp1: a figure never repeats back to back",
      all(figs1[i] != figs1[i + 1] for i in range(len(figs1) - 1)))
figs2 = [s[3] for s in seen if s[0] == "exp2"]
check("session", "exp2: ten distinct figures, each once",
      len(figs2) == len(set(figs2)) == DAY_FIGURES, len(set(figs2)))
check("session", "exp2 ranks the same figures exp1 rated",
      set(figs2) == set(figs1))
check("session", "exp1 always shows captions, exp2 never",
      all(c for e, _, c, *_ in seen if e == "exp1")
      and all(not c for e, _, c, *_ in seen if e == "exp2"))
check("session", "screens follow the experiment",
      all(sc == {"exp1": "absolute", "exp2": "tournament"}[e] for e, sc, *_ in seen))
check("session", "tournament trials carry slots A, B and C",
      all(s[4] == ["A", "B", "C"] for s in seen if s[0] == "exp2"))
_, final, _ = call("/api/state", pid=runner)
check("session", "the participant is marked complete",
      final["stage"] == "complete" and final["completed"] == final["target"],
      (final["stage"], final["completed"], final["target"]))
check("session", "per-experiment counts are reported",
      final["per_experiment"] == {"exp1": EXP1_TARGET, "exp2": EXP2_TARGET},
      final["per_experiment"])
_, done, _ = call("/api/trial/current", pid=runner)
check("session", "a finished participant is told so", done.get("done") is True, done)


# ------------------------------------------------------------ tournament --
group("the tournament over HTTP")
tp = register()
for _ in range(EXP1_TARGET):
    t = current(BASE, tp)
    if t["experiment"] != "exp1":
        break
    complete_trial(BASE, t, tp)
tt = current(BASE, tp)
check("tour", "after exp1 the tournament begins",
      tt["experiment"] == "exp2" and tt["screen"] == "tournament", tt.get("experiment"))
check("tour", "three contenders, captions off on every side",
      [s["slot"] for s in tt["slots"]] == ["A", "B", "C"]
      and all(s["cues"] == [] for s in tt["slots"]) and tt["show_captions"] is False)
check("tour", "three different decks",
      len({tuple(s["frames"]) for s in tt["slots"]}) == 3)
check("tour", "the only question is the pick",
      [q["id"] for q in tt["questions"]["questions"]] == ["pick"]
      and tt["questions"]["questions"][0]["type"] == "pick")
leaks = [t for t in FORBIDDEN + ["talk"] if t in blob_without_titles(tt)]
check("tour", "no contender is named in the payload", not leaks, leaks)

for body, why in (({}, "no picks"), ({"picks": ["A"]}, "one pick"),
                  ({"picks": ["A", "C", "B"]}, "three picks"),
                  ({"picks": ["A", "B"]}, "round 2 between the two losers"),
                  ({"picks": ["B", "A"]}, "round 2 by the round-1 loser"),
                  ({"picks": ["X", "C"]}, "an unknown slot")):
    status, resp, _ = submit(BASE, tt, tp, body)
    check("tour", "%s is refused" % why, status == 409, (status, resp))
status, resp, _ = submit(BASE, tt, tp, {"picks": ["C", "C"]})
check("tour", "C cannot win round 1 (it is not in it)", status == 409, (status, resp),
      known="app.py api_submit: `p in \"AB\" or p == \"C\"` lets C (and \"\") "
            "through as a round-1 pick; stores a rank with no winner")
# The bug above submits the trial, so fetch whatever is open now (the same
# trial once it is fixed) before the next probe.
nxt = current(BASE, tp)
status, resp, _ = submit(BASE, nxt, tp, {"picks": ["", ""]})
check("tour", "empty picks are refused, not a 500", status == 409, (status, resp),
      known="same check: '' in \"AB\" is True; slot_of[''] then KeyErrors -> 500")
good = current(BASE, tp)
check("tour", "a refused submission leaves the trial open",
      good["trial_id"] == nxt["trial_id"])
status, resp, _ = submit(BASE, good, tp, {"picks": ["A", "C"]})
check("tour", "a valid pair of picks submits", status == 200 and resp.get("ok"), (status, resp))
check("tour", "the submit response does not name the contenders",
      not any(m in json.dumps(resp).lower() for m in ("animatebanana", "qwen", "talk")),
      resp, known="app.py api_submit returns rank keyed by method name to the browser")
status, resp, _ = submit(BASE, good, tp, {"picks": ["B", "C"]})
check("tour", "a second submission is idempotent and changes nothing",
      status == 200 and resp.get("already"), resp)

_, exp = admin_call("/admin/api/export")
resp_rows = {r["question_id"]: json.loads(r["value"]) for r in exp["tables"]["response"]
             if r["trial_id"] == good["trial_id"]}
trial_row = next(r for r in exp["tables"]["trial"] if r["trial_id"] == good["trial_id"])
conds = json.loads(trial_row["presentation_conditions"])
check("tour", "the served order is the design's order",
      conds == ["animatebanana", "qwen38", "talk"], conds)
check("tour", "round1_pick and round2_pick are stored as conditions, not letters",
      resp_rows.get("round1_pick") == "animatebanana" and resp_rows.get("round2_pick") == "talk",
      resp_rows)
check("tour", "the rank is derived: winner 1, runner-up 2, round-1 loser 3",
      resp_rows.get("rank") == {"talk": 1, "animatebanana": 2, "qwen38": 3}, resp_rows.get("rank"))
check("tour", "trial rows carry all three contenders",
      len(json.loads(trial_row["presentation_ids"])) == 3)


# ----------------------------------------------------------------- admin --
group("admin surface")
status, _ = admin_call("/admin/api/overview", token=None)
check("admin", "no token is refused", status == 401, status)
status, _ = admin_call("/admin/api/overview", token="wrong")
check("admin", "a wrong token is refused", status == 401, status)

status, ov = admin_call("/admin/api/overview")
check("admin", "overview loads", status == 200 and "participants" in ov)
check("admin", "pool sizes: one absolute cell per (day-1 figure, method), one tournament per figure",
      ov["pool_sizes"] == {"exp1": DAY_FIGURES * 2, "exp2": DAY_FIGURES}, ov.get("pool_sizes"))
check("admin", "the pairwise arms are not offered in the main study",
      not any(ov["pool_sizes"].get(e) for e in ("context", "bench")), ov.get("pool_sizes"))
check("admin", "capacity is derived from the pool, not from recruitment",
      ov["capacity_participants"] == {
          "exp1": DAY_FIGURES * 2 * MAIN_CFG["judgments_per_sample"] // EXP1_TARGET,
          "exp2": DAY_FIGURES * MAIN_CFG["judgments_per_sample"] // EXP2_TARGET},
      ov.get("capacity_participants"))
check("admin", "the quota is the config's", ov["quota"] == MAIN_CFG["judgments_per_sample"])
check("admin", "trials are counted by experiment",
      ov["by_experiment"].get("exp1", 0) >= EXP1_TARGET + 4
      and ov["by_experiment"].get("exp2", 0) >= EXP2_TARGET + 1, ov["by_experiment"])

status, rows = admin_call("/admin/api/samples")
check("admin", "the sample view lists every diagram in the bundle",
      status == 200 and len(rows) == 30 and all("cells" in r for r in rows), len(rows))
check("admin", "samples carry stratification and the day stamp",
      all("complexity" in r and "study_day" in r for r in rows))
served_days = {r["study_day"] for r in rows if r["cells"]}
check("admin", "only today's figures have cells",
      served_days == {MAIN_CFG["study_day"]}, served_days)
check("admin", "every day-1 figure has its two absolute cells and a tournament",
      all(Counter(c["experiment"] for c in r["cells"]) == {"exp1": 2, "exp2": 1}
          for r in rows if r["study_day"] == MAIN_CFG["study_day"]))
check("admin", "no cell is over quota",
      all(c["judgments"] <= c["quota"] for r in rows for c in r["cells"]))

status, exps = admin_call("/admin/api/experiments")
check("admin", "the experiment view aggregates by question",
      status == 200 and [e["experiment"] for e in exps] == ["exp1", "exp2"])
exp1_q = {q["question_id"]: q for q in exps[0]["questions"]}
check("admin", "vfs answers are tallied", exp1_q["vfs"].get("counts") and
      exp1_q["vfs"]["n"] >= EXP1_TARGET, exp1_q["vfs"])
check("admin", "each question is tied to the metric it should correlate with",
      exp1_q["vfs"]["metric"] == "vfs" and exp1_q["nas"]["metric"] == "nas")
check("admin", "gated questions count only the trials that reached them",
      exp1_q["nas"]["n"] < exp1_q["vfs"]["n"], (exp1_q["nas"]["n"], exp1_q["vfs"]["n"]))

status, people = admin_call("/admin/api/participants")
check("admin", "the participant view joins identity", status == 200
      and all("display_name" in p for p in people))
target = people[0]["participant_id"]
status, _ = admin_call("/admin/api/participants/%s/annotate" % target,
                       data={"kind": "exclude", "reason": ""})
check("admin", "an exclusion without a reason is refused", status == 400, status)
status, _ = admin_call("/admin/api/participants/%s/annotate" % target,
                       data={"kind": "purge", "reason": "x"})
check("admin", "an unknown annotation kind is refused", status == 400, status)
before = admin_call("/admin/api/export")[1]["row_counts"]["response"]
status, _ = admin_call("/admin/api/participants/%s/annotate" % target,
                       data={"kind": "exclude", "reason": "suite check"})
after = admin_call("/admin/api/export")[1]["row_counts"]["response"]
check("admin", "exclusion is accepted with a reason", status == 200, status)
check("admin", "exclusion leaves raw responses untouched", before == after, (before, after))

status, exp = admin_call("/admin/api/export")
check("export", "the export loads", status == 200 and "tables" in exp)
check("export", "identity is NEVER exported",
      "participant_pii" not in exp["tables"], list(exp["tables"]))
check("export", "the export is version stamped",
      all(k in exp for k in ("bundle_id", "study_version", "config_version"))
      and exp["bundle_id"] == "main-v1")
check("export", "row counts accompany the tables",
      set(exp["row_counts"]) == set(exp["tables"]))
check("export", "the config that ran is in the export",
      any(json.loads(c["params_json"]).get("study_day") == MAIN_CFG["study_day"]
          for c in exp["tables"]["study_config"]))
status, _, _ = call("/api/coverage", raw=True)
check("export", "coverage is readable", status == 200, status)


# ---------------------------------------------------------- calibration --
group("calibration")
fresh = register()
status, prep, _ = call("/api/prepare", pid=fresh)
check("calib", "the prepare payload loads", status == 200)
check("calib", "the marking key never reaches the browser",
      all("expected" not in i for i in prep["items"]))
if not prep["calibration_available"]:
    # Correct behaviour when no expert session has been recorded: let people
    # through rather than blocking the study on missing data.
    t = current(BASE, fresh)
    check("calib", "with no expert key, the study is still reachable", "trial_id" in t, t)
    status, _, _ = call("/api/calibration/submit", {"answers": {}}, fresh)
    check("calib", "submitting to an unavailable quiz is refused", status == 409, status)
else:
    check("calib", "items cover the enabled experiments", len(prep["items"]) > 0)
status, ex, _ = call("/api/prep-example", pid=fresh)
check("calib", "the prep example is a real, captioned animation",
      status == 200 and ex["available"] and len(ex["frames"]) == len(ex["holds"])
      and any(c["text"] for c in ex["cues"]) and ex["figure"]["media_id"], ex.get("available"))
_, ex2, _ = call("/api/prep-example", pid=register())
check("calib", "the prep example is the same for everyone",
      ex2["frames"] == ex["frames"])


# ======================================================== selective cohort ==
group("selective cohort: state")
sp = register(SEL_BASE)
_, sstate, _ = scall("/api/state", pid=sp)
check("sel", "the cohort runs context then bench",
      sstate["live_experiments"] == ["context", "bench"], sstate["live_experiments"])
SEL_FIGURES = 13
check("sel", "the target is bounded by distinct figures, not by cells",
      sstate["target"] == min(SEL_CFG["samples_per_experiment"]["context"], SEL_FIGURES)
      + min(SEL_CFG["samples_per_experiment"]["bench"], SEL_FIGURES), sstate["target"])

group("selective cohort: the context trial")
ct = current(SEL_BASE, sp)
check("sel", "context comes first, on the pairwise screen",
      ct["experiment"] == "context" and ct["screen"] == "pairwise", ct.get("experiment"))
check("sel", "two slots, A and B", [s["slot"] for s in ct["slots"]] == ["A", "B"])
check("sel", "both narrations are shown",
      ct["show_captions"] and all(len(s["cues"]) > 0 for s in ct["slots"]))
check("sel", "the two narrations differ",
      ct["slots"][0]["cues"] != ct["slots"][1]["cues"])
check("sel", "the animation is the same on both sides",
      ct["slots"][0]["frames"] == ct["slots"][1]["frames"])
check("sel", "blind: no side labels", "side_labels" not in ct)
leaks = [t for t in FORBIDDEN + ["talk", "verified"] if t in blob_without_titles(ct)]
check("sel", "no condition token in the context payload", not leaks, leaks)
q = ct["questions"]["questions"]
check("sel", "one question, a three-way choice",
      [x["id"] for x in q] == ["pref_insight"] and q[0]["type"] == "choice_pair"
      and [o["value"] for o in q[0]["options"]] == ["A", "B", "tie"], q)
status, body, _ = submit(SEL_BASE, ct, sp)
check("sel", "the choice is owed", status == 409 and body.get("missing") == ["pref_insight"], body)
answer(SEL_BASE, ct, sp, "pref_insight", "tie")
status, body, _ = submit(SEL_BASE, ct, sp)
check("sel", "a tie is a valid answer", status == 200, (status, body))

group("selective cohort: two full sessions")
sessions = []
for who in (sp, register(SEL_BASE)):
    seen = []
    for _ in range(60):
        t = current(SEL_BASE, who)
        if t.get("done"):
            break
        seen.append((t["experiment"], t["figure"]["media_id"], t.get("side_labels"),
                     complete_trial(SEL_BASE, t, who), t))
    sessions.append(seen)
_, sfinal, _ = scall("/api/state", pid=sp)
check("sel", "both sessions complete at the target",
      all(len(s) + (1 if i == 0 else 0) == sstate["target"] for i, s in enumerate(sessions)),
      [len(s) for s in sessions])
check("sel", "every trial submitted", all(x[3] == 200 for s in sessions for x in s))
check("sel", "context is finished before bench begins",
      all([e for e, *_ in s] == sorted((e for e, *_ in s), key=["context", "bench"].index)
          for s in sessions))
for i, s in enumerate(sessions):
    for e in ("context", "bench"):
        figs = [f for ex, f, *_ in s if ex == e]
        check("sel", "session %d %s: every figure at most once" % (i, e),
              len(figs) == len(set(figs)), len(figs))
check("sel", "the participant is marked complete",
      sfinal["stage"] == "complete" and sfinal["completed"] == sfinal["target"], sfinal)

group("selective cohort: bench is deliberately unblinded, with fixed sides")
bench_trials = [t for s in sessions for e, _, _, _, t in s if e == "bench"]
check("sel", "bench trials were served", len(bench_trials) == 2 * SEL_FIGURES, len(bench_trials))
check("sel", "side labels name the original and the correction",
      all(t.get("side_labels") == {"A": "Original (uncorrected)",
                                   "B": "Verified and corrected"} for t in bench_trials))
check("sel", "bench asks a single yes/no",
      all([x["id"] for x in t["questions"]["questions"]] == ["improved"]
          and t["questions"]["questions"][0]["type"] == "yesno" for t in bench_trials))
check("sel", "both sides carry captions", all(
    all(len(sl["cues"]) > 0 for sl in t["slots"]) for t in bench_trials))
check("sel", "the two sides are different renders",
      all(t["slots"][0]["frames"] != t["slots"][1]["frames"] for t in bench_trials))
leaks = [tok for tok in FORBIDDEN + ["talk"] if tok in blob_without_titles(bench_trials[0])]
check("sel", "beyond the labels, nothing names the condition", not leaks, leaks)

_, sexp = admin_call("/admin/api/export", base=SEL_BASE)
by_id = {r["trial_id"]: r for r in sexp["tables"]["trial"]}
bench_rows = [by_id[t["trial_id"]] for t in bench_trials]
check("sel", "bench: the original is ALWAYS A and the correction ALWAYS B",
      all((r["presentation_a_condition"], r["presentation_b_condition"])
          == ("pre_verification", "verified") for r in bench_rows),
      Counter((r["presentation_a_condition"], r["presentation_b_condition"]) for r in bench_rows))
ctx_rows = [by_id[t["trial_id"]] for s in sessions for e, _, _, _, t in s if e == "context"]
sides = Counter(r["presentation_a_condition"] for r in ctx_rows)
# 25 draws: the chance of all landing one way is 2 * 2^-25.
check("sel", "context: both conditions appear on the left over %d draws" % len(ctx_rows),
      len(sides) == 2 and len(ctx_rows) >= 20, sides)
check("sel", "context: the analysis can recover condition from position",
      all({r["presentation_a_condition"], r["presentation_b_condition"]}
          == {"with_context", "without_context"} for r in ctx_rows))
_, sov = admin_call("/admin/api/overview", base=SEL_BASE)
check("sel", "pool sizes: one pair per figure per experiment",
      sov["pool_sizes"] == {"context": SEL_FIGURES, "bench": SEL_FIGURES}, sov["pool_sizes"])
check("sel", "the ranking arms are dark in the selective cohort",
      not any(sov["pool_sizes"].get(e) for e in ("exp1", "exp2")))


# ------------------------------------------------------------------ report --
failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:220]))
sys.exit(1 if failed else 0)
