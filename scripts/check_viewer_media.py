"""Load the viewer in a real browser and report what actually plays.

Written because three rounds of guessing at a "no video with supported format
and MIME type found" error were wrong: the files were fine, the URLs were
fine, and the real cause (a reference video shipped as MPEG-4 Part 2, which
no browser decodes) was only visible from `HTMLMediaElement.error` inside a
browser. Curl cannot see this -- it fetches the bytes happily.

    python scripts/check_viewer_media.py [URL]

Runs inside the container, where playwright's chromium lives:

    docker exec -u $(id -u):$(id -g) \
      -e PLAYWRIGHT_BROWSERS_PATH=/environments/playwright-browsers \
      animatebanana-v4 \
      bash -lc 'cd /code && /environments/img_2_svg_pretraining/bin/python \
        scripts/check_viewer_media.py <url>'

The container is on a bridge network, so a host-run viewer is NOT reachable at
127.0.0.1 from in there -- pass the public tunnel URL.
"""
from __future__ import annotations

import sys

DEFAULT_URL = "https://identifying-brian-postings-duplicate.trycloudflare.com"

VIDEO_PROBE = """() => Array.from(document.querySelectorAll('video')).map(v => ({
    src: (v.currentSrc || v.src).slice(-50),
    ready: v.readyState,
    err: v.error ? (v.error.code + ': ' + v.error.message) : null
}))"""

ROW_PROBE = """() => Array.from(document.querySelectorAll('.xhead'))
    .slice(0, 8).map(h => h.innerText.split('\\n').join(' | '))"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        failures: list[str] = []
        page.on("requestfailed",
                lambda r: failures.append(f"{r.url[-55:]} :: {r.failure}"))

        page.goto(url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(9000)

        print("=== PANELS (videos) ===")
        ok = True
        for v in page.evaluate(VIDEO_PROBE):
            state = "PLAYS" if v["ready"] >= 3 and not v["err"] else "BROKEN"
            ok = ok and state == "PLAYS"
            print(f"  [{state}] {v['src']}")
            if v["err"]:
                print(f"           {v['err']}")

        page.click("#tab-explain")
        page.wait_for_timeout(8000)
        print("\n=== METRICS ROWS (first 8) ===")
        for row in page.evaluate(ROW_PROBE):
            print("  " + row)

        if failures:
            print("\n=== FAILED REQUESTS ===")
            for f in failures[:8]:
                print("  " + f)

        browser.close()
        print("\nall videos playable" if ok else "\nSOME VIDEOS BROKEN")


if __name__ == "__main__":
    main()
