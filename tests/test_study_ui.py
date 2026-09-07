"""Browser tests for the study UI (LIVE, headless Chromium), both cohorts.

Needs the two test servers (see tests/test_study_app.py) and the study venv's
playwright:

    LD_LIBRARY_PATH=/fsxvision_new/venkat.kesav/environments/study/syslibs/usr/lib/x86_64-linux-gnu \
    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_ui.py

Chromium's shared libraries are not installed system-wide (no root on this
box); they were fetched as .debs and extracted under the study venv, which is
why LD_LIBRARY_PATH is required. `scripts/run_study_tests.sh` sets it.

What is rendered here and nowhere else: the one-page progressive form (a "No"
closes the questions behind it, a revised gate drops its dependents), the
unlock bar that holds the questions back until every on-stage slider has been
run to the end, the section interstitial, the two-round tournament with its
fade-out / slide-in, the pairwise screen that needs BOTH sliders, and the
bench screen's deliberately unblinded side labels. The offline and HTTP suites
both passed once while the trial screen was unusable, so the layout checks
are load-bearing rather than decorative.
"""
import json
import os
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("STUDY_BASE", "http://localhost:8612")
SEL_BASE = os.environ.get("STUDY_BASE_SELECTIVE", "http://localhost:8613")
ADMIN = os.environ.get("STUDY_ADMIN_TOKEN", "devtoken")

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:200]))


def group(title):
    print("\n== %s ==" % title)


def api(base, path, data=None, pid=None, headers=None):
    req = urllib.request.Request(
        base + path, method="POST" if data is not None else "GET",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json",
                 **({"X-Participant": pid} if pid else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def new_participant(base=BASE, name="UI"):
    return api(base, "/api/register", {"display_name": name, "education_level": "student",
                                       "consent": True})[1]["participant_id"]


def state(base, pid):
    return api(base, "/api/state", pid=pid)[1]


def drive_by_http(base, pid, until_experiment):
    """Complete trials over HTTP until the participant reaches `until_experiment`."""
    for _ in range(80):
        _, t = api(base, "/api/trial/current", pid=pid)
        if t.get("done") or t["experiment"] == until_experiment:
            return t
        if t["screen"] == "tournament":
            api(base, "/api/trial/%s/submit" % t["trial_id"], {"picks": ["A", "C"]}, pid)
        else:
            q = t["questions"]["questions"][0]
            api(base, "/api/trial/%s/answer" % t["trial_id"],
                {"question_id": q["id"], "value": False if q["type"] == "yesno" else "tie"}, pid)
            api(base, "/api/trial/%s/submit" % t["trial_id"], {}, pid)
    return None


def open_study(browser, base, pid, width=1600, height=1000):
    """Load /study as `pid`; return (page, interstitial title or None)."""
    page = browser.new_page(viewport={"width": width, "height": height})
    page.add_init_script("localStorage.setItem('animatebanana_study_pid','%s')" % pid)
    page.goto(base + "/study", wait_until="domcontentloaded")
    page.wait_for_selector("#between, .pl-stage, #done", state="visible", timeout=60000)
    title = None
    if page.is_visible("#between"):
        title = page.inner_text("#btTitle")
        page.click("#btGo")
    if not page.is_visible("#done"):
        page.wait_for_selector(".pl-stage", timeout=60000)
        # The figure is a real image fetch; measuring it before it has decoded
        # reports a 0-wide box and says nothing about the layout.
        page.wait_for_function("document.querySelector('#figimg').naturalWidth > 0",
                               timeout=60000)
        page.wait_for_timeout(400)
    return page, title


def run_slider_to_end(page, index=None):
    """What a participant does with the keyboard, done in one go: put the
    slider(s) at max and fire the events the player listens for."""
    page.evaluate("""(idx) => {
        document.querySelectorAll('.pl-seek').forEach((s, i) => {
            if (idx !== null && i !== idx) return;
            s.value = s.max;
            s.dispatchEvent(new Event('input'));
            s.dispatchEvent(new Event('change'));
        });
    }""", index)
    page.wait_for_timeout(350)


def visible_holders(page):
    return page.evaluate("""() => [...document.querySelectorAll('#players > div')]
        .filter((d) => d.style.display !== 'none')
        .map((d) => d.querySelector('.pl-label').textContent)""")


for base in (BASE, SEL_BASE):
    try:
        urllib.request.urlopen(base + "/", timeout=5)
    except Exception as exc:                                       # noqa: BLE001
        print("server not reachable on %s: %s" % (base, exc))
        sys.exit(2)


with sync_playwright() as pw:
    browser = pw.chromium.launch()

    # ------------------------------------------------------------ landing --
    group("landing page")
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(BASE + "/", wait_until="networkidle")
    check("landing", "the begin button starts disabled", page.is_disabled("#go"))
    page.fill("#name", "Test Person")
    page.select_option("#level", "student")
    page.select_option("#papers", "weekly")
    check("landing", "consent alone still gates the button", page.is_disabled("#go"))
    page.check("#consent")
    check("landing", "a complete form enables the button", page.is_enabled("#go"))
    page.close()

    # -------------------------------------------------------------- trial --
    group("first entry: interstitial, layout, style bar")
    pid = new_participant()
    page, title = open_study(browser, BASE, pid)
    check("between", "a section intro is shown on first entry", title is not None, title)
    check("between", "it names the section in the participant's terms",
          title and "Part 1" in title and "quality" in title.lower(), title)

    m = page.evaluate("""() => {
        const r = (s) => { const e = document.querySelector(s);
                           return e ? e.getBoundingClientRect() : null; };
        const bs = getComputedStyle(document.body);
        const hs = getComputedStyle(document.documentElement);
        return {docH: document.documentElement.scrollHeight,
                winH: window.innerHeight,
                overflowHidden: bs.overflowY === 'hidden' || hs.overflowY === 'hidden',
                scrollable: document.documentElement.scrollHeight > window.innerHeight,
                bodyOverflowX: document.documentElement.scrollWidth > window.innerWidth,
                fig: r('#figimg'), stage: r('.pl-stage'), unlock: r('#unlockbar'),
                locked: r('.qf-locked'), submit: r('.qf-submit'), style: r('#stylebar')};
    }""")
    check("layout", "the document is not height-locked", not m["overflowHidden"])
    check("layout", "all content is reachable",
          m["docH"] <= m["winH"] or m["scrollable"], (m["docH"], m["winH"]))
    check("layout", "no horizontal overflow", not m["bodyOverflowX"])
    check("layout", "the figure is rendered at a readable size",
          m["fig"]["width"] > 250, m["fig"]["width"])
    check("layout", "the player does not dominate the viewport",
          m["stage"]["height"] < m["winH"] * 0.42, m["stage"]["height"])
    check("layout", "the unlock bar is at the top of the page",
          m["unlock"] and m["unlock"]["top"] < 120, m["unlock"])
    check("layout", "the style bar sits above the animation",
          m["style"] and m["style"]["bottom"] <= m["stage"]["top"] + 1,
          (m["style"], m["stage"]))
    check("layout", "the locked question panel is above the fold",
          m["locked"] is not None and 0 < m["locked"]["top"] < m["winH"], m["locked"])
    check("layout", "the submit control exists", m["submit"] is not None)

    _, payload = api(BASE, "/api/trial/current", pid=pid)
    check("stylebar", "the style name is shown in bold over the animation",
          page.is_visible("#stylebar") and page.inner_text("#sbName") == payload["style_name"],
          (page.inner_text("#sbName"), payload["style_name"]))
    check("stylebar", "with its one-line summary",
          page.inner_text("#sbSum").strip() == payload["style_description"].strip())
    check("stylebar", "and the judge's rules behind a disclosure",
          page.locator("#sbRules li").count() == len(payload["style_rules"]))
    check("top", "progress reads 1 of N", page.inner_text("#prog").startswith("1 of "),
          page.inner_text("#prog"))
    check("top", "the section title is the question set's",
          page.inner_text("#expname") == payload["questions"]["title"])
    check("top", "captions are on, so no 'visuals only' tag",
          page.inner_text("#stylename").strip() == "")

    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    page.wait_for_timeout(300)
    after = page.evaluate("() => document.querySelector('#figimg').getBoundingClientRect().top")
    check("layout", "the figure stays pinned when scrolled to the bottom", after > -50, after)
    page.evaluate("window.scrollTo(0, 0)")

    # ------------------------------------------------------------- player --
    group("player")
    check("player", "no timed playback controls exist",
          page.locator(".pl-play").count() == 0 and page.locator(".pl-tab").count() == 0)
    check("player", "the slider is usable straight away", not page.is_disabled(".pl-seek"))
    check("player", "the caption bar shows the first cue",
          page.inner_text(".pl-cue").strip() == payload["slots"][0]["cues"][0]["text"].strip(),
          page.inner_text(".pl-cue")[:60])
    check("player", "it reports not viewed yet", "not viewed" in page.inner_text(".pl-badge"))
    check("player", "progress toward the end is shown",
          "%" in page.inner_text('[data-role="progress"]'))
    check("player", "the step counter starts at 1",
          page.inner_text(".pl-step").startswith("1 /"), page.inner_text(".pl-step"))

    group("questions stay hidden until the stimulus has been seen")
    check("locked", "no question is shown yet", page.locator(".qf-q").count() == 0)
    check("locked", "the reason is stated", page.locator(".qf-locked").count() == 1)
    check("locked", "submit is disabled", page.is_disabled(".qf-submit"))
    check("locked", "and the note says why",
          "slider" in page.inner_text('[data-role="note"]').lower())
    check("locked", "the unlock bar names the single slider",
          "the slider" in page.inner_text("#unlocktext").lower()
          and "both" not in page.inner_text("#unlocktext").lower())

    group("arrow keys")
    page.click(".pl-seek")            # focuses the slider (and jumps to where it was clicked)
    at = int(page.evaluate("Number(document.querySelector('.pl-seek').value)"))
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(80)
    check("keys", "the right arrow lights up",
          page.locator('[data-role="kright"].pl-key-on').count() == 1)
    page.wait_for_timeout(250)
    check("keys", "the indicator fades again", page.locator(".pl-key-on").count() == 0)
    check("keys", "the step counter advanced by one",
          page.inner_text(".pl-step").startswith("%d /" % (at + 2)),
          (at, page.inner_text(".pl-step")))
    check("keys", "the hint clears once the slider is used", not page.is_visible(".pl-hint"))
    check("keys", "the unlock bar tracks progress",
          page.inner_text("#unlockprog").strip() not in ("", "0%"), page.inner_text("#unlockprog"))

    group("reaching the end unlocks the form")
    run_slider_to_end(page)
    check("unlock", "the badge flips to viewed",
          page.inner_text(".pl-badge").strip() == "viewed", page.inner_text(".pl-badge"))
    check("unlock", "the unlock bar disappears", not page.is_visible("#unlockbar"))
    check("unlock", "the body is marked unlocked",
          page.evaluate("document.body.classList.contains('unlocked')"))
    check("unlock", "only the first question is shown", page.locator(".qf-q").count() == 1)
    check("unlock", "it is the VFS gate, as a horizontal yes/no",
          page.locator(".qf-q >> nth=0 >> .qf-yn .qf-yes").count() == 1
          and page.locator(".qf-q >> nth=0 >> .qf-yn .qf-no").count() == 1)
    check("unlock", "numbered Q1", page.inner_text(".qf-num").strip() == "Q1.")
    check("unlock", "submit still disabled, one question left",
          page.is_disabled(".qf-submit") and "1 question left" in page.inner_text('[data-role="note"]'),
          page.inner_text('[data-role="note"]'))

    group("media stays on screen while answering")
    page.evaluate("window.scrollTo(0, 600)")
    page.wait_for_timeout(300)
    vis = page.evaluate("""() => {
        const s = document.querySelector('.pl-stage').getBoundingClientRect();
        const f = document.querySelector('#figimg').getBoundingClientRect();
        return {stage: s.top, fig: f.top, h: window.innerHeight};
    }""")
    check("parallax", "the animation stays visible when scrolled", -20 < vis["stage"] < vis["h"])
    check("parallax", "the figure stays visible when scrolled", -20 < vis["fig"] < vis["h"])
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)

    group("full screen")
    page.click(".pl-stage")
    page.wait_for_timeout(400)
    check("fs", "clicking the animation opens full screen", page.is_visible("#fsview"))
    check("fs", "the slider is still usable inside",
          page.locator('[data-role="fsseek"]').count() == 1)
    before = page.inner_text('[data-role="fsstep"]')
    page.click('[data-role="fsseek"]')
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    check("fs", "arrow keys step inside full screen",
          page.inner_text('[data-role="fsstep"]') != before)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    check("fs", "esc returns to the questions",
          not page.is_visible("#fsview") and page.locator(".qf-q").count() == 1)
    page.click("#figimg")
    page.wait_for_timeout(400)
    check("fs", "the source figure opens full screen too", page.is_visible("#fsview"))
    check("fs", "a still image hides the frame controls",
          "fs-still" in (page.get_attribute("#fsview", "class") or ""))
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)

    # ---------------------------------------------------------- questions --
    group("the progressive form")
    page.click(".qf-q >> nth=0 >> .qf-no")
    page.wait_for_timeout(350)
    check("form", "No closes the form: no further question appears",
          page.locator(".qf-q").count() == 1)
    check("form", "the gate-closed text is shown", page.locator(".qf-closed").count() == 1
          and "submit" in page.inner_text(".qf-closed").lower())
    check("form", "the No is marked selected",
          page.locator(".qf-q >> nth=0 >> .qf-no.qf-sel").count() == 1)
    check("form", "submit is enabled after a No alone", page.is_enabled(".qf-submit"))
    check("form", "the answer reached the server",
          api(BASE, "/api/trial/current", pid=pid)[1].get("saved_answers") == {"vfs": False})

    page.click(".qf-q >> nth=0 >> .qf-yes")
    page.wait_for_timeout(350)
    check("form", "Yes opens the style question", page.locator(".qf-q").count() == 2
          and "style" in page.inner_text(".qf-q >> nth=1 >> .qf-prompt").lower())
    check("form", "the gate-closed text goes away", page.locator(".qf-closed").count() == 0)
    check("form", "the style help is the style summary",
          page.inner_text(".qf-q >> nth=1 >> .qf-help").strip() == payload["style_description"].strip())
    check("form", "submit is disabled again", page.is_disabled(".qf-submit"))

    page.click(".qf-q >> nth=1 >> .qf-yes")
    page.wait_for_timeout(350)
    check("form", "both Yes: all four questions on one page", page.locator(".qf-q").count() == 4)
    check("form", "the two scores are dropdowns", page.locator("select.qf-dd").count() == 2)
    check("form", "with their anchors", page.locator(".qf-anchor").count() == 4
          and "0 =" in page.inner_text(".qf-anchor >> nth=0"))
    check("form", "the dropdowns start unanswered",
          page.evaluate("[...document.querySelectorAll('select.qf-dd')].every((s) => s.value === '')"))
    check("form", "two questions left", "2 questions left" in page.inner_text('[data-role="note"]'))
    page.select_option("select.qf-dd >> nth=0", "7")
    page.wait_for_timeout(300)
    check("form", "one score in: one question left",
          page.is_disabled(".qf-submit") and "1 question left" in page.inner_text('[data-role="note"]'))
    page.select_option("select.qf-dd >> nth=1", "0")
    page.wait_for_timeout(300)
    check("form", "a score of 0 counts as answered", page.is_enabled(".qf-submit"))
    check("form", "everything reached the server",
          api(BASE, "/api/trial/current", pid=pid)[1].get("saved_answers")
          == {"vfs": True, "ascs": True, "sss": 7, "nas": 0})

    group("revising a gate drops what hung off it")
    page.click(".qf-q >> nth=1 >> .qf-no")
    page.wait_for_timeout(350)
    check("revise", "ascs=No hides the scores", page.locator(".qf-q").count() == 2
          and page.locator("select.qf-dd").count() == 0)
    check("revise", "and shows the gate-closed text", page.locator(".qf-closed").count() == 1)
    check("revise", "submit is enabled", page.is_enabled(".qf-submit"))
    page.click(".qf-q >> nth=1 >> .qf-yes")
    page.wait_for_timeout(350)
    check("revise", "reopening brings the scores back UNanswered",
          page.locator("select.qf-dd").count() == 2 and page.is_disabled(".qf-submit")
          and page.evaluate("[...document.querySelectorAll('select.qf-dd')].every((s) => s.value === '')"))
    check("revise", "the answer trail is kept server-side (revisions append)",
          api(BASE, "/api/trial/current", pid=pid)[1].get("saved_answers").get("ascs") is True)

    group("a dropped answer is repaired on submit")
    # The client retries a failed POST three times on its own, so a single
    # dropped request lands anyway. Black-hole the whole retry burst for one
    # answer: the server then genuinely lacks it, and only the repair-on-submit
    # path can make the trial complete.
    dropped = {"n": 0}

    def drop_all(route):
        dropped["n"] += 1
        route.abort()

    page.route("**/answer", drop_all)
    page.select_option("select.qf-dd >> nth=0", "5")      # every attempt is dropped
    page.wait_for_timeout(2500)                           # 400 + 800 ms between retries
    page.unroute("**/answer")
    page.select_option("select.qf-dd >> nth=1", "6")      # this one goes through
    page.wait_for_timeout(600)
    check("repair", "the answer's POST and its retries were all dropped",
          dropped["n"] >= 2, dropped["n"])
    check("repair", "the server is missing that answer",
          "sss" not in api(BASE, "/api/trial/current", pid=pid)[1].get("saved_answers", {}))
    check("repair", "the form still considers itself complete", page.is_enabled(".qf-submit"))
    page.click(".qf-submit")
    page.wait_for_selector(".qf-locked", timeout=60000)
    page.wait_for_timeout(500)
    st = state(BASE, pid)
    check("repair", "the trial submitted despite the dropped write",
          st["completed"] == 1 and st["per_experiment"] == {"exp1": 1}, st)
    check("repair", "the next trial is on screen, locked again",
          page.inner_text("#prog").startswith("2 of ") and page.locator(".qf-locked").count() == 1
          and page.is_visible("#unlockbar"), page.inner_text("#prog"))
    check("between", "no interstitial within a section", not page.is_visible("#between"))

    group("submit and continue")
    run_slider_to_end(page)
    page.click(".qf-q >> nth=0 >> .qf-no")
    page.wait_for_timeout(300)
    page.click(".qf-submit")
    page.wait_for_selector(".qf-locked", timeout=60000)
    check("next", "a No-only trial submits and the third loads",
          page.inner_text("#prog").startswith("3 of ") and state(BASE, pid)["completed"] == 2)
    page.close()

    # --------------------------------------------------------- tournament --
    group("tournament: round 1")
    tp = new_participant(name="Tour")
    drive_by_http(BASE, tp, "exp2")
    page, title = open_study(browser, BASE, tp)
    check("tour", "the section interstitial introduces the ranking",
          title and "Part 2" in title, title)
    check("tour", "the tournament panel replaces the form",
          page.is_visible("#tour") and page.evaluate("document.querySelector('#form').innerHTML") == "")
    check("tour", "round 1 of 2", "round 1 of 2" in page.inner_text(".tour-round").lower(),
          page.inner_text(".tour-round"))
    check("tour", "the question is the pick prompt",
          "use" in page.inner_text(".tour-q").lower())
    check("tour", "two pick buttons, both disabled until watched",
          page.locator(".tour-pick button").count() == 2
          and all(page.locator(".tour-pick button").nth(i).is_disabled() for i in range(2)))
    check("tour", "A and B are on stage, C waits offstage",
          visible_holders(page) == ["Left (A)", "Right (B)"]
          and page.locator("#players > div").count() == 3, visible_holders(page))
    check("tour", "visuals only: the caption bars are hidden",
          page.inner_text("#stylename").strip() == "· visuals only"
          and page.evaluate("[...document.querySelectorAll('.pl-cue')].every((e) => e.style.display === 'none')"))
    check("tour", "the style bar is still shown", page.is_visible("#stylebar"))
    check("tour", "the unlock bar asks for BOTH sliders",
          "both" in page.inner_text("#unlocktext").lower() and "0 of 2" in page.inner_text("#unlockprog"))
    run_slider_to_end(page, 0)
    check("tour", "one slider done is not enough",
          page.locator(".tour-pick button").nth(0).is_disabled()
          and "1 of 2" in page.inner_text("#unlockprog"), page.inner_text("#unlockprog"))
    run_slider_to_end(page, 1)
    check("tour", "both watched enables the picks",
          all(page.locator(".tour-pick button").nth(i).is_enabled() for i in range(2))
          and not page.is_visible("#unlockbar"))

    group("tournament: the loser leaves, C arrives")
    page.click(".tour-pick button >> nth=0")        # pick A
    page.wait_for_timeout(150)
    check("tour", "the loser fades", page.locator("#players > div.tour-out").count() == 1)
    page.wait_for_timeout(900)
    check("tour", "the loser is then hidden and C takes its place",
          visible_holders(page) == ["Left (A)", "Right (C)"], visible_holders(page))
    check("tour", "C slides in", page.locator("#players > div.tour-in").count() == 1)
    check("tour", "round 2 of 2", "round 2 of 2" in page.inner_text(".tour-round").lower())
    check("tour", "the buttons name the round-2 pair",
          [page.locator(".tour-pick button").nth(i).inner_text() for i in range(2)]
          == ["Left (A)", "Right (C)"])
    check("tour", "the picks lock again until C has been watched",
          all(page.locator(".tour-pick button").nth(i).is_disabled() for i in range(2))
          and page.is_visible("#unlockbar") and "1 of 2" in page.inner_text("#unlockprog"))
    run_slider_to_end(page)
    check("tour", "watching C unlocks the second pick",
          all(page.locator(".tour-pick button").nth(i).is_enabled() for i in range(2)))
    before_prog = page.inner_text("#prog")
    page.click(".tour-pick button >> nth=1")        # pick C
    page.wait_for_function(
        "(p) => document.querySelector('#prog').textContent !== p", arg=before_prog, timeout=60000)
    page.wait_for_timeout(400)
    st = state(BASE, tp)
    check("tour", "the second pick submits the ranking",
          st["per_experiment"].get("exp2") == 1, st["per_experiment"])
    check("tour", "and the next tournament loads",
          page.is_visible("#tour") and "round 1 of 2" in page.inner_text(".tour-round").lower()
          and visible_holders(page) == ["Left (A)", "Right (B)"], visible_holders(page))
    _, exp = api(BASE, "/admin/api/export", headers={"X-Admin-Token": ADMIN})
    ranks = [json.loads(r["value"]) for r in exp["tables"]["response"]
             if r["participant_id"] == tp and r["question_id"] == "rank"]
    check("tour", "the server derived the rank from the two picks",
          ranks == [{"talk": 1, "animatebanana": 2, "qwen38": 3}], ranks)
    page.close()

    # -------------------------------------------------------- narrow view --
    group("narrow viewport")
    page, _ = open_study(browser, BASE, new_participant(name="Narrow"), width=1024, height=768)
    n = page.evaluate("""() => ({
        overflowX: document.documentElement.scrollWidth > window.innerWidth + 2,
        stacked: getComputedStyle(document.querySelector('#main')).flexDirection === 'column',
        questionArea: document.querySelector('.qf-locked, .qf-q') !== null})""")
    check("narrow", "the layout stacks below 1100px", n["stacked"])
    check("narrow", "still no horizontal overflow", not n["overflowX"])
    check("narrow", "the question area still renders", n["questionArea"])
    page.close()

    # ---------------------------------------------------------- selective --
    group("pairwise: the context screen needs both sliders")
    sp = new_participant(SEL_BASE, name="Sel")
    page, title = open_study(browser, SEL_BASE, sp)
    check("pair", "the interstitial introduces narration", title and "Narration" in title, title)
    check("pair", "two players, lettered left and right",
          visible_holders(page) == ["Left (A)", "Right (B)"], visible_holders(page))
    check("pair", "no tournament panel", not page.is_visible("#tour"))
    check("pair", "both narrations are on screen",
          page.evaluate("[...document.querySelectorAll('.pl-cue')].every((e) => e.textContent.trim().length > 0)"))
    check("pair", "the unlock bar asks for BOTH sliders",
          "both" in page.inner_text("#unlocktext").lower())
    run_slider_to_end(page, 0)
    check("pair", "one side watched keeps the form locked",
          page.locator(".qf-q").count() == 0 and page.is_visible("#unlockbar"))
    run_slider_to_end(page, 1)
    check("pair", "both watched shows the single choice",
          page.locator(".qf-q").count() == 1 and not page.is_visible("#unlockbar"))
    opts = page.evaluate("[...document.querySelectorAll('.qf-q .qf-opt')].map((e) => e.textContent.trim())")
    check("pair", "three options: first, second, tie",
          opts == ["First", "Second", "Both are equally useful"], opts)
    page.click(".qf-q .qf-opt >> nth=2")
    page.wait_for_timeout(300)
    check("pair", "choosing enables submit", page.is_enabled(".qf-submit")
          and page.locator(".qf-opt.qf-sel").count() == 1)
    check("pair", "the tie reached the server",
          api(SEL_BASE, "/api/trial/current", pid=sp)[1].get("saved_answers") == {"pref_insight": "tie"})
    page.click(".qf-submit")
    page.wait_for_selector(".qf-locked", timeout=60000)
    check("pair", "the next pair loads", page.inner_text("#prog").startswith("2 of ")
          and state(SEL_BASE, sp)["completed"] == 1)
    page.close()

    group("bench: unblinded labels, fixed sides")
    drive_by_http(SEL_BASE, sp, "bench")
    page, title = open_study(browser, SEL_BASE, sp)
    check("bench", "the interstitial introduces corrections", title and "Corrections" in title, title)
    check("bench", "the players are labelled original and corrected",
          visible_holders(page) == ["Original (uncorrected)", "Verified and corrected"],
          visible_holders(page))
    run_slider_to_end(page)
    check("bench", "one yes/no question", page.locator(".qf-q").count() == 1
          and page.locator(".qf-yn .qf-yes").count() == 1
          and "improvement" in page.inner_text(".qf-prompt").lower())
    page.click(".qf-yn .qf-yes")
    page.wait_for_timeout(300)
    check("bench", "answering enables submit", page.is_enabled(".qf-submit"))
    page.click(".qf-submit")
    page.wait_for_selector(".qf-locked, #done", timeout=60000)
    st = state(SEL_BASE, sp)
    check("bench", "the answer is recorded under bench",
          st["per_experiment"].get("bench") == 1, st["per_experiment"])
    page.close()

    browser.close()


failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:220]))
sys.exit(1 if failed else 0)
