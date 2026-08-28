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
    page.goto(base + "/study", wait_until="networkidle")
    for _ in range(60):
        page.wait_for_timeout(400)
        if page.is_visible("#done"):
            break
        if page.is_visible("#between"):
            report["intros"] += 1
            page.click("#btGo")
            page.wait_for_timeout(500)
        try:
            page.wait_for_selector(".pl-stage", timeout=15000)
        except Exception:                                      # noqa: BLE001
            if page.is_visible("#done"):
                break
            raise

        # Take every slider to the end, as a participant must.
        for i in range(page.locator(".pl-seek").count()):
            page.locator(".pl-seek").nth(i).click()
            n = page.evaluate(
                "i => Number(document.querySelectorAll('.pl-seek')[i].max)", i)
            for _ in range(n + 2):
                page.keyboard.press("ArrowRight")
        page.wait_for_timeout(500)

        if page.locator(".qf-locked").count():
            report["locked_stuck"] += 1
            break

        # Answer whatever is in front of us until submit enables.
        for _ in range(40):
            if page.is_enabled(".qf-submit"):
                break
            opt = page.locator(".qf-current .qf-opt").first
            if opt.count() == 0:
                skip = page.locator('.qf-current [data-role="skip"]')
                if skip.count():
                    skip.click()
                    continue
                break
            opt.click()
            page.wait_for_timeout(120)

        if not page.is_enabled(".qf-submit"):
            report["submit_stuck"] += 1
            break
        page.click(".qf-submit")
        report["trials"] += 1
        page.wait_for_timeout(700)


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
            page = ctx.new_page()
            rep = {"i": i, "pid": pid, "trials": 0, "intros": 0,
                   "errors": [], "locked_stuck": 0, "submit_stuck": 0}
            page.on("pageerror", lambda e, r=rep: r["errors"].append(str(e)[:120]))
            page.add_init_script(
                "localStorage.setItem('animatebanana_study_pid','%s')" % pid)
            contexts.append(ctx)
            pages.append((page, pid, rep))
            reports.append(rep)

        # Interleave a step at a time so the contexts genuinely overlap rather
        # than running one after another.
        for page, pid, rep in pages:
            try:
                drive(page, args.base, pid, rep)
            except Exception as exc:                           # noqa: BLE001
                rep["errors"].append("driver: %s" % str(exc)[:160])

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
        print("    ctx %d: %s" % (r["i"], r["errors"][0]))
    # Every participant must get its own section intros; a global client-side
    # key silently gives later participants none.
    missing = [r["i"] for r in reports if r["trials"] and r["intros"] == 0]
    print("  contexts that saw NO section intro: %s" % (missing or "none"))
    return 1 if (errs or stuck or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
