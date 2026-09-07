"""Evict cached responses that can never parse, so a retry can re-sample.

THE TRAP THIS EXISTS FOR. The response cache is content-addressed and
permanent: a bad answer, once stored, is replayed for every subsequent attempt.
A stage that fails to parse it therefore fails IDENTICALLY forever, which reads
as "the model cannot do this" when the truth is "the model answered badly once".
This project has now hit that three times -- Qwen truncated mid-JSON, Gemma
returned LaTeX for an SVG target, and Gemini did both.

Two distinct shapes, both evicted here:

  WRONG FORMAT  a complete LaTeX/TikZ document where SVG was asked for. Ends
                cleanly with \\end{document}, so it is the model ignoring the
                target representation rather than any transport fault.
  TRUNCATED     opens `<svg`/`<Diagram`/`{` and never closes. Measured on a
                designer response cut at 1199 bytes mid-`<marker`, and on an
                animator-critic response cut the same way.

Deliberately conservative: only responses that are structurally unparseable are
removed. A response that parses but is merely poor is a real result and stays.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def is_unusable(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return "empty"
    body = t
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        body = body.rsplit("```", 1)[0].strip()
    if body.startswith("\\documentclass") or "\\begin{tikzpicture}" in body[:400]:
        return "latex-for-svg-target"
    if body.lstrip().startswith("<svg") and "</svg>" not in body:
        return "truncated-svg"
    if body.lstrip().startswith("<Diagram") and "</Diagram>" not in body:
        return "truncated-xml"
    if body.lstrip().startswith("{"):
        try:
            json.loads(body)
        except json.JSONDecodeError:
            return "truncated-json"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_root")
    ap.add_argument("--dataset", default="animatebench_v5")
    ap.add_argument("--fresh-only", action="store_true",
                    help="only evict entries this cache owns (nlink==1), never "
                         "a hardlink shared with the cache it was seeded from")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pat = os.path.join(args.cache_root, args.dataset, "responses", "*", "*.json")
    counts: dict[str, int] = {}
    for f in glob.glob(pat):
        if args.fresh_only and os.stat(f).st_nlink != 1:
            continue
        try:
            text = json.load(open(f)).get("text") or ""
        except (OSError, json.JSONDecodeError):
            text = ""
        why = is_unusable(text)
        if not why:
            continue
        counts[why] = counts.get(why, 0) + 1
        if not args.dry_run:
            os.remove(f)
    verb = "would evict" if args.dry_run else "evicted"
    total = sum(counts.values())
    detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    print(f"  {verb} {total} unusable response(s): {detail}")


if __name__ == "__main__":
    main()
