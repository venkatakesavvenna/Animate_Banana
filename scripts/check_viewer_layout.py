"""Screenshot the viewer's three tabs and report layout facts.

Complements check_viewer_media.py: that one answers "does it load", this one
answers "does it read clearly" -- how many panels sit on a row, whether the
target sub-tabs switch, and whether the scoreboard carries both targets.

    docker exec -u $(id -u):$(id -g) \
      -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
      animatebanana-v4 \
      bash -lc 'cd /code && /environments/img_2_svg_pretraining/bin/python \
        scripts/check_viewer_layout.py <url> <outdir>'
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_URL = "https://identifying-brian-postings-duplicate.trycloudflare.com"


def main() -> None:
    from playwright.sync_api import sync_playwright

    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/viewer_shots")
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.goto(url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(8000)

        # --- Panels: are a model's two targets on the same row? ---
        page.screenshot(path=str(out / "panels.png"), full_page=True)
        tops = page.evaluate(
            """() => Array.from(document.querySelectorAll('#grid .cell')).map(c => ({
                title: (c.querySelector('h3 span') || {}).innerText || '',
                top: Math.round(c.getBoundingClientRect().top)}))""")
        print("=== PANELS: cell -> row offset ===")
        for c in tops:
            print(f"  top={c['top']:5d}  {c['title'][:46]}")
        rows: dict[int, list[str]] = {}
        for c in tops:
            rows.setdefault(c["top"], []).append(c["title"])
        paired = [v for v in rows.values() if len(v) > 1]
        print(f"  -> {len(rows)} row(s); rows with 2+ cells: {len(paired)}")

        # --- Metrics: do the sub-tabs exist and switch? ---
        page.click("#tab-explain")
        page.wait_for_timeout(7000)
        tabs = page.evaluate(
            "() => Array.from(document.querySelectorAll('.xtab')).map(b => b.innerText)")
        print(f"\n=== METRICS sub-tabs: {tabs} ===")
        page.screenshot(path=str(out / "metrics_tikz.png"), full_page=False)
        if len(tabs) > 1:
            page.click(f".xtab:nth-of-type({len(tabs)})")
            page.wait_for_timeout(6000)
            active = page.evaluate(
                "() => (document.querySelector('.xtab.active')||{}).innerText")
            first = page.evaluate(
                """() => (document.querySelector('.xhead .v')||{}).innerText""")
            print(f"  after switch: active={active!r} first value={first!r}")
            page.screenshot(path=str(out / "metrics_svg.png"), full_page=False)

        # --- Scoreboard: both targets in one table? ---
        page.click("#tab-table")
        page.wait_for_timeout(6000)
        hdr = page.evaluate(
            "() => Array.from(document.querySelectorAll('table.score thead tr.grp th')).map(t=>t.innerText)")
        body = page.evaluate(
            """() => Array.from(document.querySelectorAll('table.score tbody tr')).slice(0,4)
                 .map(r => Array.from(r.children).slice(0,4).map(c=>c.innerText).join(' | '))""")
        print(f"\n=== SCOREBOARD header: {hdr} ===")
        for b in body:
            print("  " + b)
        page.screenshot(path=str(out / "scoreboard.png"), full_page=False)

        browser.close()
        print(f"\nscreenshots -> {out}")


if __name__ == "__main__":
    main()
