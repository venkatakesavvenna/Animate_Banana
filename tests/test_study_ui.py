"""Browser tests for the study UI (LIVE, headless Chromium).

Needs the server on 8607 and the study venv's playwright:

    LD_LIBRARY_PATH=/fsxvision_new/venkat.kesav/environments/study/syslibs/usr/lib/x86_64-linux-gnu \
    PYTHONPATH=src /fsxvision_new/venkat.kesav/environments/study/bin/python \
        tests/test_study_ui.py

Chromium's shared libraries are not installed system-wide (no root on this
box); they were fetched as .debs and extracted under the study venv, which is
why LD_LIBRARY_PATH is required. `scripts/run_study_ui_tests.sh` sets it.

These exist because the offline and HTTP suites both passed while the trial
screen was unusable: it was a fixed 100vh shell with overflow:hidden, so the
players ate the viewport and the questions were unreachable. Nothing short of
rendering it catches that, so the layout assertions here are load-bearing
rather than decorative.
"""
import json
import os
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("STUDY_BASE", "http://localhost:8607")

_results = []


def check(group, name, cond, detail=""):
    _results.append((group, name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          "" if cond else "   <- " + str(detail)[:200]))


def group(title):
    print("\n== %s ==" % title)


def new_participant(name="UI"):
    req = urllib.request.Request(
        BASE + "/api/register",
        data=json.dumps({"display_name": name, "education_level": "student",
                         "consent": True}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["participant_id"]


try:
    urllib.request.urlopen(BASE + "/", timeout=5)
except Exception as exc:                                       # noqa: BLE001
    print("server not reachable on %s: %s" % (BASE, exc))
    sys.exit(2)


with sync_playwright() as pw:
    browser = pw.chromium.launch()

    # ------------------------------------------------------------ landing --
    group("landing page")
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(BASE + "/", wait_until="networkidle")
    check("landing", "the begin button starts disabled",
          page.is_disabled("#go"))
    page.fill("#name", "Test Person")
    page.select_option("#level", "student")
    page.select_option("#papers", "weekly")
    check("landing", "consent alone still gates the button", page.is_disabled("#go"))
    page.check("#consent")
    check("landing", "a complete form enables the button", page.is_enabled("#go"))
    page.close()

    # -------------------------------------------------------------- trial --
    group("trial layout")
    pid = new_participant()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.add_init_script(
        "localStorage.setItem('animatebanana_study_pid','%s')" % pid)
    page.goto(BASE + "/study", wait_until="networkidle")
    # The section interstitial covers the trial on first entry to each section.
    page.wait_for_timeout(900)
    if page.is_visible("#between"):
        check("between", "a section intro is shown on first entry", True)
        check("between", "it names the section",
              bool(page.inner_text("#btTitle").strip()))
        page.click("#btGo")
    page.wait_for_selector(".pl-stage", timeout=15000)
    page.wait_for_timeout(800)

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
                fig: r('#figimg'), stage: r('.pl-stage'),
                locked: r('.qf-locked'), submit: r('.qf-submit')};
    }""")
    # The bug this guards against was a trapped 100vh shell: content existed
    # below the fold with no way to reach it. The property is "everything is
    # reachable", which is satisfied either by fitting or by scrolling -- not
    # by the page happening to be tall.
    check("layout", "the document is not height-locked",
          not m["overflowHidden"], m["overflowHidden"])
    check("layout", "all content is reachable",
          m["docH"] <= m["winH"] or m["scrollable"], (m["docH"], m["winH"]))
    check("layout", "no horizontal overflow", not m["bodyOverflowX"])
    check("layout", "the figure is rendered at a readable size",
          m["fig"]["width"] > 250, m["fig"]["width"])
    check("layout", "the player does not dominate the viewport",
          m["stage"]["height"] < m["winH"] * 0.42, m["stage"]["height"])
    check("layout", "the locked question panel is above the fold",
          m["locked"] is not None and 0 < m["locked"]["top"] < m["winH"],
          m["locked"])
    check("layout", "the submit control exists", m["submit"] is not None)

    # The figure has to stay visible while answering -- that is the protocol,
    # not a nicety.
    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    page.wait_for_timeout(400)
    after = page.evaluate(
        "() => document.querySelector('#figimg').getBoundingClientRect().top")
    check("layout", "the figure stays pinned when scrolled to the bottom",
          after > -50, after)
    page.evaluate("window.scrollTo(0, 0)")

    # ------------------------------------------------------------- player --
    group("player")
    check("player", "timed playback is gone entirely",
          page.locator('[data-mode="video"]').count() == 0
          and page.locator(".pl-play").count() == 0)
    check("player", "the slider is usable straight away",
          not page.is_disabled(".pl-seek"))
    check("player", "the unlock instruction is at the top of the page",
          page.is_visible("#unlockbar")
          and page.evaluate("() => document.querySelector('#unlockbar')"
                            ".getBoundingClientRect().top") < 120)
    check("player", "the hint tells you how to move",
          page.is_visible(".pl-hint"))
    check("player", "it reports not viewed yet",
          "not viewed" in page.inner_text(".pl-badge"),
          page.inner_text(".pl-badge"))
    check("player", "progress toward the end is shown",
          "%" in page.inner_text('[data-role="progress"]'),
          page.inner_text('[data-role="progress"]'))

    group("questions stay hidden until the stimulus has been seen")
    check("locked", "no question is shown yet", page.locator(".qf-prompt").count() == 0)
    check("locked", "the reason is stated", page.locator(".qf-locked").count() == 1)
    check("locked", "submit is disabled", page.is_disabled(".qf-submit"))
    check("locked", "and it says why",
          "slider" in page.inner_text('[data-role="note"]').lower(),
          page.inner_text('[data-role="note"]'))

    group("arrow keys")
    page.click(".pl-seek")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(80)
    check("keys", "the right arrow lights up",
          page.locator('[data-role="kright"].pl-key-on').count() == 1)
    page.wait_for_timeout(250)
    check("keys", "the indicator fades again",
          page.locator(".pl-key-on").count() == 0)
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(80)
    check("keys", "the left arrow lights up",
          page.locator('[data-role="kleft"].pl-key-on').count() == 1)
    page.wait_for_timeout(250)
    check("keys", "the hint clears once the slider is used",
          not page.is_visible(".pl-hint"))

    group("reaching the end unlocks the questions")
    steps = page.evaluate("() => Number(document.querySelector('.pl-seek').max)")
    for _ in range(steps + 2):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(700)
    check("unlock", "the badge flips to viewed",
          "viewed" in page.inner_text(".pl-badge")
          and "not viewed" not in page.inner_text(".pl-badge"),
          page.inner_text(".pl-badge"))
    check("unlock", "the questions appear", page.locator(".qf-prompt").count() == 1)
    check("unlock", "the unlock bar disappears", not page.is_visible("#unlockbar"))

    group("media stays on screen while answering (parallax)")
    page.evaluate("window.scrollTo(0, 600)")
    page.wait_for_timeout(400)
    vis = page.evaluate("""() => {
        const s = document.querySelector('.pl-stage').getBoundingClientRect();
        const f = document.querySelector('#figimg').getBoundingClientRect();
        return {stage: s.top, fig: f.top, h: window.innerHeight};
    }""")
    check("parallax", "the animation stays visible when scrolled",
          -20 < vis["stage"] < vis["h"], vis["stage"])
    check("parallax", "the figure stays visible when scrolled",
          -20 < vis["fig"] < vis["h"], vis["fig"])
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)

    group("full screen")
    page.click(".pl-stage")
    page.wait_for_timeout(400)
    check("fs", "clicking the animation opens full screen", page.is_visible("#fsview"))
    check("fs", "the slider is still usable inside",
          page.locator('[data-role="fsseek"]').count() == 1)
    before = page.inner_text('[data-role="fsstep"]')
    page.click('[data-role="fsseek"]')      # focus inside the overlay
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)
    check("fs", "arrow keys step inside full screen",
          page.inner_text('[data-role="fsstep"]') != before)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    check("fs", "esc returns to the questions",
          not page.is_visible("#fsview") and page.locator(".qf-prompt").count() == 1)
    page.click("#figimg")
    page.wait_for_timeout(400)
    check("fs", "the source figure opens full screen too", page.is_visible("#fsview"))
    check("fs", "a still image hides the frame controls",
          "fs-still" in (page.get_attribute("#fsview", "class") or ""))
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    check("unlock", "familiarity is asked first",
          "familiar" in page.inner_text(".qf-prompt").lower(),
          page.inner_text(".qf-prompt"))

    # ---------------------------------------------------------- questions --
    group("question flow")
    page.click("text=Somewhat familiar")
    page.wait_for_timeout(400)
    check("questions", "answering advances to the next question",
          "familiar" not in page.inner_text(".qf-prompt").lower(),
          page.inner_text(".qf-prompt"))
    check("questions", "the answered question becomes a chip",
          page.locator(".qf-chip").count() == 1)

    first_prompt = page.inner_text(".qf-prompt")
    page.click(".qf-scale .qf-opt >> nth=3")
    page.wait_for_timeout(400)
    check("questions", "a Likert answer is recorded as a chip",
          page.locator(".qf-chip").count() == 2)
    second_prompt = page.inner_text(".qf-prompt")

    # Revision used to strand the participant: `record` stays put on a revision
    # so the change is visible, and with every later question also answered
    # there was no way forward and no control offering one.
    page.click(".qf-chip >> nth=1")
    page.wait_for_timeout(300)
    check("questions", "clicking a chip reopens that question",
          page.inner_text(".qf-prompt") == first_prompt)
    check("questions", "the previous answer is shown as selected",
          page.locator(".qf-scale .qf-opt.qf-sel").count() == 1)
    page.click(".qf-scale .qf-opt >> nth=0")
    page.wait_for_timeout(500)
    check("questions", "a revision stays on the same question",
          page.inner_text(".qf-prompt") == first_prompt)
    check("questions", "the chip shows the revised value",
          "1" in page.inner_text(".qf-chip >> nth=1"),
          page.inner_text(".qf-chip >> nth=1"))

    group("navigation out of a revision")
    check("nav", "Next is offered after revising", page.is_enabled('[data-role="next"]'))
    page.click('[data-role="next"]')
    page.wait_for_timeout(300)
    check("nav", "Next moves forward again",
          page.inner_text(".qf-prompt") == second_prompt,
          page.inner_text(".qf-prompt"))
    page.click('[data-role="prev"]')
    page.wait_for_timeout(300)
    check("nav", "Previous moves back",
          page.inner_text(".qf-prompt") == first_prompt)
    page.click('[data-role="next"]')
    page.wait_for_timeout(200)
    check("nav", "forward still works after going back",
          page.inner_text(".qf-prompt") == second_prompt)

    stored = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/api/state", headers={"X-Participant": pid})))
    check("questions", "the session is still in progress on the server",
          stored["completed"] == 0)
    page.close()

    # -------------------------------------------------------- narrow view --
    group("narrow viewport")
    page = browser.new_page(viewport={"width": 1024, "height": 768})
    page.add_init_script(
        "localStorage.setItem('animatebanana_study_pid','%s')" % new_participant("Narrow"))
    page.goto(BASE + "/study", wait_until="networkidle")
    page.wait_for_timeout(900)
    if page.is_visible("#between"):
        page.click("#btGo")
    page.wait_for_selector(".pl-stage", timeout=15000)
    page.wait_for_timeout(600)
    n = page.evaluate("""() => ({
        overflowX: document.documentElement.scrollWidth > window.innerWidth + 2,
        stacked: getComputedStyle(document.querySelector('#main')).flexDirection === 'column',
        questionArea: document.querySelector('.qf-locked, .qf-prompt') !== null})""")
    check("narrow", "the layout stacks below 1100px", n["stacked"])
    check("narrow", "still no horizontal overflow", not n["overflowX"])
    check("narrow", "the question area still renders", n["questionArea"])
    page.close()

    browser.close()


failed = [r for r in _results if not r[2]]
print("\n%d checks, %d passed, %d failed" %
      (len(_results), len(_results) - len(failed), len(failed)))
for g, name, _, detail in failed:
    print("  FAIL [%s] %s   %s" % (g, name, str(detail)[:220]))
sys.exit(1 if failed else 0)
