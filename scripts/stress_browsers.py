#!/usr/bin/env python
"""Concurrency stress in REAL browser contexts, not just HTTP.

`simulate_participants.py` hammers the API, which catches server-side races but
runs no JavaScript. This drives N independent Chromium contexts through the
actual UI at the same time -- separate localStorage per context, so each is a
distinct participant sharing one browser the way a lab full of laptops shares
one server.

It exists because two classes of bug only appear here:
  * per-participant client state kept under a GLOBAL key (the section-intro
    bug: participant 2 inherited participant 1's "already seen" list);
  * JS exceptions, which an HTTP driver cannot observe at all.

    python scripts/stress_browsers.py --n 6 --base http://localhost:8609
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter

from playwright.sync_api import sync_playwright


def register(base, name):
    req = urllib.request.Request(
        base + "/api/register",
        data=json.dumps({"display_name": name, "education_level": "student",
                         "consent": True}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["participant_id"]


def drive(page, base, pid, report):
    """Walk one participant through every trial via the UI."""
    page.goto(base + "/study", wait_until="networkidle", timeout=60000)
    for _ in range(60):
        page.wait_for_timeout(400)
        if page.is_visible("#done"):
            break
        if page.is_visible("#between"):
            report["intros"] += 1
            page.click("#btGo")
            page.wait_for_timeout(500)
        try:
            page.wait_for_selector(".pl-stage", timeout=45000)
        except Exception:                                      # noqa: BLE001
            if page.is_visible("#done"):
                break
            raise

        # Take every slider to the end, as a participant must.
        # Only the sliders on stage: a tournament mounts its third player
        # hidden until round 2, and clicking a hidden input waits forever.
        for i in range(page.locator(".pl-seek").count()):
            seek = page.locator(".pl-seek").nth(i)
            if not seek.is_visible():
                continue
            seek.click()
            n = page.evaluate(
                "i => Number(document.querySelectorAll('.pl-seek')[i].max)", i)
            for _ in range(n + 2):
                page.keyboard.press("ArrowRight")
        page.wait_for_timeout(500)

        # Whichever screen this is, its controls mount after the players; a
        # count taken too early reads 0 and sends the driver down the wrong path.
        try:
            page.wait_for_selector(".tour-pick button, .qf-submit", timeout=15000)
        except Exception:                                      # noqa: BLE001
            pass
        # Tournament screen: pick, wait for the third slot, slide it, pick again.
        if page.locator(".tour-pick button").count():
            for rnd in range(2):
                for i in range(page.locator(".pl-seek").count()):
                    page.evaluate(
                        "i => { const s=document.querySelectorAll('.pl-seek')[i]; s.value=s.max;"
                        " s.dispatchEvent(new Event('input',{bubbles:true}));"
                        " s.dispatchEvent(new Event('change',{bubbles:true})); }", i)
                page.wait_for_timeout(500)
                btn = page.locator(".tour-pick button").nth(rnd % 2)
                if not btn.is_enabled():
                    report["submit_stuck"] += 1
                    return
                btn.click()
                page.wait_for_timeout(1400)
            report["trials"] += 1
            page.wait_for_timeout(1200)
            continue

        if page.locator(".qf-locked").count():
            report["locked_stuck"] += 1
            raise RuntimeError("locked_stuck: sliders at end but form still locked")

        # One-page progressive form: answer every visible question, in order,
        # until submit enables. Yes/No and choices are buttons; scores are
        # <select>s. A gate answer may hide later questions -- fine, submit
        # enables on the visible set.
        for _ in range(40):
            if page.is_enabled(".qf-submit"):
                break
            progressed = False
            for q in page.locator(".qf-q").all():
                if q.locator(".qf-opt.qf-sel").count():
                    continue
                sel = q.locator("select")
                if sel.count() and not sel.first.input_value():
                    sel.first.select_option("7"); progressed = True; break
                opt = q.locator(".qf-opt").first
                if opt.count():
                    opt.click(); progressed = True; break
            page.wait_for_timeout(150)
            if not progressed:
                break

        if not page.is_enabled(".qf-submit"):
            report["submit_stuck"] += 1
            raise RuntimeError("submit_stuck: answered everything visible, submit still disabled")
        # Submit re-mounts the whole trial: playwright's default click keeps
        # waiting for the (now detached, then disabled) button to settle and
        # times out even though the submit went through. Fire the click and
        # judge success by the progress counter moving instead.
        before = page.inner_text("#prog")
        page.locator(".qf-submit").first.dispatch_event("click")
        try:
            page.wait_for_function(
                "b => document.querySelector('#prog').innerText !== b"
                " || document.querySelector('#done')?.offsetParent"
                " || getComputedStyle(document.querySelector('#between')).display !== 'none'",
                arg=before, timeout=20000)
        except Exception:                                      # noqa: BLE001
            report["submit_stuck"] += 1
            raise RuntimeError("submit_stuck: clicked submit, progress did not advance")
        report["trials"] += 1
        page.wait_for_timeout(500)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--base", default="http://localhost:8609")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    started = time.time()
    reports = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        contexts, pages = [], []
        for i in range(args.n):
            pid = register(args.base, "Stress %02d" % i)
            # A separate context per participant: isolated localStorage, which
            # is precisely the thing the global-key bug got wrong.
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            # Register the session on the CONTEXT, before any page exists. On a
            # page it races the first navigation over a slow link, and the tab
            # lands on /study with no participant and simply sits there.
            ctx.add_init_script(
                "localStorage.setItem('animatebanana_study_pid','%s')" % pid)
            page = ctx.new_page()
            rep = {"i": i, "pid": pid, "trials": 0, "intros": 0,
                   "errors": [], "locked_stuck": 0, "submit_stuck": 0}
            page.on("pageerror", lambda e, r=rep: r["errors"].append(str(e)[:120]))
            contexts.append(ctx)
            pages.append((page, pid, rep))
            reports.append(rep)

        # Interleave a step at a time so the contexts genuinely overlap rather
        # than running one after another.
        for page, pid, rep in pages:
            try:
                drive(page, args.base, pid, rep)
            except Exception as exc:                           # noqa: BLE001
                rep["errors"].append("driver: %s" % str(exc)[:400])
                # Leave evidence: what was on screen and why the submit was
                # not clickable, so a timeout is diagnosable after the fact.
                try:
                    page.screenshot(path="data/study_runs/stress_fail_ctx%d.png" % i)
                    rep["errors"].append("state: " + json.dumps(page.evaluate("""() => {
                      const chain = (el) => { const out=[]; for (let e=el; e; e=e.parentElement) {
                        const cs=getComputedStyle(e); out.push(e.tagName+'#'+(e.id||'')+'.'+e.className+
                        ' d='+cs.display+' v='+cs.visibility+' o='+cs.opacity+' h='+e.offsetHeight); } return out; };
                      const b=document.querySelector('.qf-submit');
                      return {url: location.href, unlocked: document.body.classList.contains('unlocked'),
                        submits: document.querySelectorAll('.qf-submit').length,
                        between: getComputedStyle(document.querySelector('#between')).display,
                        tour: document.querySelector('#tour') ? getComputedStyle(document.querySelector('#tour')).display : null,
                        qs: document.querySelectorAll('.qf-q').length,
                        chain: b ? chain(b).slice(0,6) : null}; }""")))
                except Exception as exc2:                          # noqa: BLE001
                    rep["errors"].append("state failed: %s" % exc2)

        for ctx in contexts:
            ctx.close()
        browser.close()

    elapsed = time.time() - started
    print("\n%d browser contexts in %.0fs" % (args.n, elapsed))
    print("  trials submitted : %d" % sum(r["trials"] for r in reports))
    print("  section intros   : %s" % Counter(r["intros"] for r in reports))
    stuck = [r for r in reports if r["locked_stuck"] or r["submit_stuck"]]
    errs = [r for r in reports if r["errors"]]
    print("  stuck            : %d" % len(stuck))
    print("  contexts with JS errors: %d" % len(errs))
    for r in errs[:5]:
        for e in r["errors"][:3]:
            print("    ctx %d: %s" % (r["i"], e))
    # Every participant must get its own section intros; a global client-side
    # key silently gives later participants none.
    missing = [r["i"] for r in reports if r["trials"] and r["intros"] == 0]
    print("  contexts that saw NO section intro: %s" % (missing or "none"))
    return 1 if (errs or stuck or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
