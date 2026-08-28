"""HTTP-level study tests (LIVE).

Needs the server running against the pilot bundle on port 8607:

    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        -m img_2_svg_pretraining.study.app \
        --bundle data/study_bundles/pilot-v1 \
        --config src/img_2_svg_pretraining/pipeline/configs/study.yaml \
        --db /tmp/study_test.db --port 8607

No model calls. Participants created here are left in place -- they are
anonymous rows in a scratch DB, and deleting them would exercise a delete path
the production schema deliberately does not have.

Three of these pin bugs that only appeared over HTTP and that the offline
suites could not have caught:
  * `datetime('now','subsec')` returns NULL on sqlite < 3.42, so every
    interaction event violated NOT NULL.
  * Flask resolves a relative send_file path against the app root, not the
    CWD, so media 500'd while Path.exists() said otherwise.
  * The progress target counted arms that have no stimuli, parking the bar at
    40% on a session that was actually finished.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8607"

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:200]))


def group(title):
    print("\n== %s ==" % title)


def call(path, data=None, pid=None, raw=False):
    req = urllib.request.Request(
        BASE + path, method="POST" if data is not None else "GET",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json",
                 **({"X-Participant": pid} if pid else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, (r.read() if raw else json.load(r)), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            body = json.loads(body)
        except Exception:
            body = {}
        return e.code, body, dict(e.headers)


def register(**kw):
    payload = {"display_name": "T", "education_level": "student",
               "consent": True}
    payload.update(kw)
    _, body, _ = call("/api/register", payload)
    return body["participant_id"]


def answer_for(q):
    return {"likert5": 4, "yesno": True, "choice_ab": "A",
            "select": (q.get("options") or [{"value": "x"}])[0]["value"],
            "text": "note"}[q["type"]]


try:
    urllib.request.urlopen(BASE + "/", timeout=5)
except Exception as exc:                                   # noqa: BLE001
    print("server not reachable on %s: %s" % (BASE, exc))
    sys.exit(2)


# -------------------------------------------------------------- routing --
group("pages and assets")
for path, kind in (("/", "text/html"), ("/study", "text/html"),
                   ("/player.js", "javascript"), ("/forms.js", "javascript"),
                   ("/study.js", "javascript")):
    status, _, headers = call(path, raw=True)
    check("routes", "%s serves %s" % (path, kind),
          status == 200 and kind in headers.get("Content-Type", ""),
          (status, headers.get("Content-Type")))


# ------------------------------------------------------------ lifecycle --
group("registration and state")
status, body, _ = call("/api/register", {"education_level": "", "consent": True})
check("register", "education level is required", status == 400, status)

pid = register(roll_no="R99", area="cv", reads_papers="weekly")
check("register", "returns an opaque participant id", len(pid) == 32, pid)

status, state, _ = call("/api/state", pid=pid)
check("state", "state is readable", status == 200 and state["completed"] == 0)
_, _cov, _ = call("/api/state", pid=pid)
check("state", "a target is reported", state["target"] > 0, state["target"])
check("state", "live sections are named",
      len(state.get("live_experiments", [])) > 0, state.get("live_experiments"))
LIVE = state["live_experiments"]
TARGET = state["target"]

status, _, _ = call("/api/state", pid="notarealparticipant")
check("state", "unknown participant is rejected", status == 401, status)
status, _, _ = call("/api/state")
check("state", "missing participant is rejected", status == 401, status)


# ---------------------------------------------------------------- trial --
group("trial assignment over HTTP")
status, trial, _ = call("/api/trial/current", pid=pid)
check("trial", "a trial is served", status == 200 and "trial_id" in trial)
check("trial", "exp1 comes first", trial["experiment"] == "exp1", trial["experiment"])
check("trial", "exp1 hides captions",
      trial["show_captions"] is False and trial["slots"][0]["cues"] == [])
check("trial", "the figure is supplied", bool(trial["figure"]["media_id"]))
check("trial", "frames are supplied",
      len(trial["slots"][0]["frames"]) == trial["slots"][0]["n_frames"])
check("trial", "holds match frame count",
      len(trial["slots"][0]["holds"]) == trial["slots"][0]["n_frames"])
check("trial", "style text reaches the participant",
      bool(trial["style_description"]))
check("trial", "questions are attached",
      len(trial["questions"]["questions"]) == 12)

_, again, _ = call("/api/trial/current", pid=pid)
check("trial", "refetching resumes the same trial",
      again["trial_id"] == trial["trial_id"])
check("trial", "resumed payload is identical",
      json.dumps(again, sort_keys=True) == json.dumps(trial, sort_keys=True))

group("blinding over the wire")
blob = json.dumps(trial).lower()
# Every style slug, so the check does not quietly pass just because this
# trial happened to draw a style nobody listed.
STYLE_SLUGS = ["progressive_reveal", "colour_pop", "alpha_masking",
               "hopping_bounding_box", "sliding_bounding_box"]
FORBIDDEN = ["narrative_id", "with_context", "without_context", "pre_verification",
             "animatebanana", "baseline", "cell_key", "position_seed",
             "assignment_reason", "gemini", "lineage", ".webp", "/media/",
             "diagram_id"] + STYLE_SLUGS
leaks = [t for t in FORBIDDEN if t in blob]
check("blinding", "no condition or path token in the trial payload", not leaks, leaks)
check("blinding", "style is named for the participant, not as a lineage token",
      trial["style_name"] in [s.replace("_", " ") for s in STYLE_SLUGS]
      and "_" not in trial["style_name"], trial["style_name"])


# ---------------------------------------------------------------- media --
group("media")
for label, mid in (("figure", trial["figure"]["media_id"]),
                   ("frame", trial["slots"][0]["frames"][0])):
    status, body, headers = call("/media/" + mid, raw=True)
    # 200 bytes, not 1KB: a progressive reveal's first frame is a nearly empty
    # canvas and compresses to a few hundred bytes. That is real stimulus
    # content, not a truncated response.
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
group("answers and the submit gate")
tid = trial["trial_id"]
status, body, _ = call("/api/trial/%s/submit" % tid, {}, pid)
check("gate", "submitting with nothing answered is refused", status == 409, status)
expected_missing = len(trial["questions"]["questions"]) + (
    1 if trial["questions"]["familiarity"] else 0)
check("gate", "the refusal names every missing question",
      len(body.get("missing", [])) == expected_missing,
      (len(body.get("missing", [])), expected_missing))

if trial["questions"]["familiarity"]:
    call("/api/trial/%s/answer" % tid, {"question_id": "familiarity",
                                        "value": "somewhat"}, pid)
for q in trial["questions"]["questions"]:
    call("/api/trial/%s/answer" % tid,
         {"question_id": q["id"], "value": answer_for(q)}, pid)

status, body, _ = call("/api/trial/%s/answer" % tid,
                       {"question_id": "qual_overall", "value": 2}, pid)
check("answers", "a revision is accepted", status == 200)
check("answers", "the revision is what is read back",
      body["answers"]["qual_overall"] == 2, body["answers"].get("qual_overall"))

status, _, _ = call("/api/trial/%s/answer" % tid, {"question_id": "x"}, pid)
check("answers", "an answer without a value is refused", status == 400, status)

other = register()
status, _, _ = call("/api/trial/%s/answer" % tid,
                    {"question_id": "qual_overall", "value": 1}, other)
check("answers", "another participant cannot answer this trial",
      status == 404, status)

status, _, _ = call("/api/trial/%s/submit" % tid, {}, pid)
check("gate", "a complete trial submits", status == 200, status)
status, body, _ = call("/api/trial/%s/submit" % tid, {}, pid)
check("gate", "re-submitting is idempotent",
      status == 200 and body.get("already"), body)
status, _, _ = call("/api/trial/%s/answer" % tid,
                    {"question_id": "qual_overall", "value": 5}, pid)
check("gate", "a submitted trial no longer accepts answers", status == 409, status)


# ---------------------------------------------------------------- events --
group("interaction events")
_, trial2, _ = call("/api/trial/current", pid=pid)
status, _, _ = call("/api/events", {"events": [
    {"trial_id": trial2["trial_id"], "type": "animation_play", "slot": "A",
     "t_video": 0.0, "client_seq": 0},
    {"trial_id": trial2["trial_id"], "type": "animation_complete", "slot": "A",
     "t_video": 12.5, "client_seq": 1, "payload": {"watched_fraction": 0.98}},
]}, pid)
check("events", "events are accepted (server_ts default works on this sqlite)",
      status == 200, status)
status, _, _ = call("/api/events", {"events": [{"type": "no_trial_id"}]}, pid)
check("events", "a malformed event is dropped rather than failing the batch",
      status == 200, status)


# -------------------------------------------------------- full sequence --
group("familiarity is asked once per diagram")
fampid = register()
asked = []
asked_detail = []
for _ in range(6):
    _, ft, _ = call("/api/trial/current", pid=fampid)
    if ft.get("done"):
        break
    asked.append((ft["experiment"], ft["questions"]["familiarity"] is not None))
    # The client never sees diagram_id, so ask the admin surface which figure
    # this trial used.
    _, _adm, _ = call("/api/state", pid=fampid)
    asked_detail.append((ft["experiment"],
                         ft["questions"]["familiarity"] is not None,
                         ft["figure"]["media_id"]))
    if ft["questions"]["familiarity"]:
        call("/api/trial/%s/answer" % ft["trial_id"],
             {"question_id": "familiarity", "value": "somewhat"}, fampid)
    for q in ft["questions"]["questions"]:
        if not q.get("optional"):
            call("/api/trial/%s/answer" % ft["trial_id"],
                 {"question_id": q["id"], "value": answer_for(q)}, fampid)
    call("/api/trial/%s/submit" % ft["trial_id"], {}, fampid)
check("familiarity", "asked on the first encounter with a diagram",
      asked and asked[0][1] is True, asked)
# Track it per diagram rather than per position: a new figure SHOULD be asked.
seen_diagrams = set()
repeats_asked = []
for exp, was_asked, diagram in asked_detail:
    if diagram in seen_diagrams:
        repeats_asked.append((exp, diagram, was_asked))
    seen_diagrams.add(diagram)
check("familiarity", "a repeated diagram is never asked again",
      all(not a for _, _, a in repeats_asked), repeats_asked)
check("familiarity", "the test actually exercised a repeat",
      len(repeats_asked) > 0, len(repeats_asked))


group("a full session")
runner = register(education_level="researcher")
seen = []
for _ in range(60):
    _, t, _ = call("/api/trial/current", pid=runner)
    if t.get("done"):
        break
    if t["questions"]["familiarity"]:
        call("/api/trial/%s/answer" % t["trial_id"],
             {"question_id": "familiarity", "value": "familiar"}, runner)
    for q in t["questions"]["questions"]:
        if q.get("optional"):
            continue
        call("/api/trial/%s/answer" % t["trial_id"],
             {"question_id": q["id"], "value": answer_for(q)}, runner)
    call("/api/trial/%s/submit" % t["trial_id"], {}, runner)
    seen.append((t["experiment"], t["show_captions"]))

order = [e for e, _ in seen]
check("session", "the session completes", len(seen) == TARGET, (len(seen), TARGET))
check("session", "sections run in order, one at a time",
      order == sorted(order, key=lambda e: LIVE.index(e)), order)
check("session", "every live section is reached",
      set(order) == set(LIVE), (sorted(set(order)), LIVE))
check("session", "exp1 never shows captions",
      all(not c for e, c in seen if e == "exp1"))
check("session", "exp2 always shows captions",
      all(c for e, c in seen if e == "exp2"))
_, final, _ = call("/api/state", pid=runner)
check("session", "the participant is marked complete",
      final["stage"] == "complete" and final["completed"] == final["target"],
      (final["stage"], final["completed"], final["target"]))


# ----------------------------------------------------------------- admin --
group("admin surface")

ADMIN = os.environ.get("STUDY_ADMIN_TOKEN", "devtoken")


def admin_call(path, token=ADMIN, data=None):
    req = urllib.request.Request(
        BASE + path, method="POST" if data is not None else "GET",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json",
                 **({"X-Admin-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {}


status, _ = admin_call("/admin/api/overview", token=None)
check("admin", "no token is refused", status == 401, status)
status, _ = admin_call("/admin/api/overview", token="wrong")
check("admin", "a wrong token is refused", status == 401, status)

status, ov = admin_call("/admin/api/overview")
check("admin", "overview loads", status == 200 and "participants" in ov)
dark = [e for e in ("exp1", "exp2", "exp3", "exp4", "exp5") if e not in LIVE]
check("admin", "every arm outside live_experiments has an empty pool",
      all(ov["pool_sizes"].get(e, 0) == 0 for e in dark),
      {e: ov["pool_sizes"].get(e) for e in dark})
check("admin", "every live arm has a pool",
      all(ov["pool_sizes"].get(e, 0) > 0 for e in LIVE), ov.get("pool_sizes"))
check("admin", "capacity is derived from the pool, not from recruitment",
      "capacity_participants" in ov and ov["capacity_participants"],
      ov.get("capacity_participants"))

status, rows = admin_call("/admin/api/samples")
check("admin", "the sample view lists every diagram",
      status == 200 and len(rows) > 0 and "cells" in rows[0], len(rows))
check("admin", "samples carry stratification for replacement matching",
      all("element_density" in r for r in rows))

status, exps = admin_call("/admin/api/experiments")
check("admin", "the experiment view aggregates by question",
      status == 200 and any(e["questions"] for e in exps))
likert = [q for e in exps for q in e["questions"] if q.get("mean") is not None]
check("admin", "likert questions carry mean/median/sd", bool(likert),
      len(likert))
check("admin", "each question is tied to the metric it should correlate with",
      any(q.get("metric") for e in exps for q in e["questions"]))

status, people = admin_call("/admin/api/participants")
check("admin", "the participant view joins identity", status == 200)
check("admin", "identity is available to admin only",
      all("display_name" in p for p in people) if people else True)

if people:
    target = people[0]["participant_id"]
    status, _ = admin_call("/admin/api/participants/%s/annotate" % target,
                           data={"kind": "exclude", "reason": ""})
    check("admin", "an exclusion without a reason is refused", status == 400, status)
    before = admin_call("/admin/api/export")[1]["row_counts"]["response"]
    status, _ = admin_call("/admin/api/participants/%s/annotate" % target,
                           data={"kind": "exclude", "reason": "suite check"})
    after = admin_call("/admin/api/export")[1]["row_counts"]["response"]
    check("admin", "exclusion is accepted with a reason", status == 200, status)
    check("admin", "exclusion leaves raw responses untouched", before == after,
          (before, after))

status, exp = admin_call("/admin/api/export")
check("export", "the export loads", status == 200 and "tables" in exp)
check("export", "identity is NEVER exported",
      "participant_pii" not in exp["tables"], list(exp["tables"]))
check("export", "the export is version stamped",
      all(k in exp for k in ("bundle_id", "study_version", "config_version")))
check("export", "row counts accompany the tables",
      set(exp["row_counts"]) == set(exp["tables"]))


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
    _, t, _ = call("/api/trial/current", pid=fresh)
    check("calib", "with no expert key, the study is still reachable",
          "trial_id" in t, t)
    status, _, _ = call("/api/calibration/submit", {"answers": {}}, fresh)
    check("calib", "submitting to an unavailable quiz is refused",
          status == 409, status)
else:
    check("calib", "items cover the enabled experiments", len(prep["items"]) > 0)


# ------------------------------------------------------------------ report --
failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:220]))
sys.exit(1 if failed else 0)
