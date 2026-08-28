"""Correct `ascs_unenforced_rules` on records written by a stale driver.

The long-running sweep imports `animation_quality` once, at start. When
UNENFORCEABLE_RULES changed mid-sweep -- "Mobile Boxes, Static Elements"
became enforceable after its schema field was added -- every record the
driver wrote afterwards kept the old message, even though the judge WAS
answering the rule and the answer is present in `ascs_frame_detail`.

That field is annotation, not measurement: no score derives from it. But it
would tell a reader a CRITICAL rule went unchecked when it did not, so it is
rewritten from the live table rather than left to mislead. Scores are never
touched.

    python scripts/fix_unenforced_rules.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from img_2_svg_pretraining.animatebench.metrics import animation_quality as aq  # noqa: E402

EVALS = (REPO / "data" / "animatebench_v3_cache" / "animatebench_v3"
         / "evals" / "bench_v3_or")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fixed = 0
    for path in sorted(EVALS.glob("*/*/animation.json")):
        style = path.parent.parent.name
        want = aq.UNENFORCEABLE_RULES.get(style, [])
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "ascs_unenforced_rules" not in record:
            continue
        if record["ascs_unenforced_rules"] == want:
            continue
        print(f"{style}/{path.parent.name}: "
              f"{record['ascs_unenforced_rules']} -> {want}")
        if not args.dry_run:
            record["ascs_unenforced_rules"] = want
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        fixed += 1
    print(f"\n{'would fix' if args.dry_run else 'fixed'} {fixed} record(s)")


if __name__ == "__main__":
    main()
