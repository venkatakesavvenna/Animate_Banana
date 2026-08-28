"""Frame judge vs video judge: do they agree, and does either one discriminate?

Compares Qwen's frame-level VFS (in `evals/`) against Gemini's video-level VFS
(in `evals_video_judge/`) over the same cells.

READ THE TIE COUNTS FIRST. 86% of cells sit at exactly VFS 1.0 on the frame
side. A Pearson r against a variable that is constant on 86% of its support is
not a weak correlation, it is an undefined question -- so this script leads with
distribution and discrimination, and reports rank correlation only on the subset
where the frame judge actually varies.

The number that decides the merge question is not the correlation at all. It is
section 3: of the cells the frame judge tied at ceiling, how many does the video
judge separate, and does that separation track `ascs_pass`? If it does, a video
VFS can absorb ASCS. If it does not, ASCS is measuring something neither
fidelity judge sees and must stay a separate gate.

Usage:
    python3 scripts/vfs_video_correlation.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "data/animatebench_v3_cache/animatebench_v3/evals"
VIDEO = REPO / "data/animatebench_v3_cache/animatebench_v3/evals_video_judge"
SKIP_DIRS = {"_stale_prompts_2026-08-19"}


def _load(root: Path) -> dict[tuple, dict]:
    out = {}
    for p in Path(root).glob("*/*/*/animation.json"):
        cfg = p.parent.parent.parent.name
        if cfg in SKIP_DIRS:
            continue
        try:
            out[(cfg, p.parent.parent.name, p.parent.name)] = json.loads(
                p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else None


def _spearman(xs, ys):
    if len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau_b(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0:
                tx += 1; ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


def _auc(scores, labels):
    """P(score of a positive > score of a negative), ties counted as half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0)
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    live, video = _load(LIVE), _load(VIDEO)
    rows = []
    for key, vrec in sorted(video.items()):
        lrec = live.get(key)
        if not lrec or lrec.get("vfs") is None or vrec.get("vfs_video") is None:
            continue
        rows.append({
            "config": key[0], "style": key[1], "sample": key[2],
            "target": "svg" if key[0].endswith("_svg") else "tikz",
            "frames": lrec["vfs"], "video": vrec["vfs_video"],
            "ascs_pass": lrec.get("ascs_pass"),
            "defects": vrec.get("vfs_video_temporal_defects"),
        })

    if not rows:
        raise SystemExit(f"no paired cells. Run scripts/vfs_video_sweep.py first "
                         f"(video root: {VIDEO})")

    fr = [r["frames"] for r in rows]
    vi = [r["video"] for r in rows]
    n = len(rows)

    print("=" * 78)
    print(f"FRAME JUDGE (Qwen, per-frame) vs VIDEO JUDGE (Gemini, whole clip)")
    print(f"{n} paired cell(s)")
    print("=" * 78)

    # -- 1. distribution, before any correlation -------------------------
    print("\n1. DISTRIBUTION -- read this before any correlation")
    ties_f = sum(1 for v in fr if v >= 0.999)
    ties_v = sum(1 for v in vi if v >= 0.999)
    print(f"  frame VFS   mean {st.mean(fr):.3f}  median {st.median(fr):.3f}  "
          f"sd {st.pstdev(fr):.3f}  distinct {len(set(round(v,3) for v in fr))}"
          f"  at 1.0: {ties_f}/{n} ({ties_f/n:.0%})")
    print(f"  video VFS   mean {st.mean(vi):.3f}  median {st.median(vi):.3f}  "
          f"sd {st.pstdev(vi):.3f}  distinct {len(set(round(v,3) for v in vi))}"
          f"  at 1.0: {ties_v}/{n} ({ties_v/n:.0%})")
    if ties_f / n > 0.5:
        print(f"\n  NOTE: the frame judge is at ceiling on {ties_f/n:.0%} of cells.")
        print("  A correlation over the full set is dominated by ties; section 2")
        print("  restricts it to the varying subset and section 3 is the real test.")

    # -- 2. agreement ----------------------------------------------------
    print("\n2. AGREEMENT")
    var = [r for r in rows if r["frames"] < 0.999]
    for label, subset in (("all cells", rows), ("frame VFS < 1.0 only", var)):
        if len(subset) < 3:
            print(f"  {label:<24} n={len(subset)} -- too few to correlate")
            continue
        a = [r["frames"] for r in subset]
        b = [r["video"] for r in subset]
        print(f"  {label:<24} n={len(subset):<3} "
              f"Spearman {_fmt(_spearman(a,b))}  Kendall tau-b {_fmt(_kendall_tau_b(a,b))}"
              f"  Pearson {_fmt(_pearson(a,b))}")

    # Bland-Altman: a systematic offset is more useful than a correlation.
    diffs = [r["video"] - r["frames"] for r in rows]
    md, sd = st.mean(diffs), (st.pstdev(diffs) if len(diffs) > 1 else 0.0)
    print(f"\n  mean difference (video - frames)  {md:+.3f} on the 0-1 scale "
          f"({md*10:+.2f} points of 10)")
    print(f"  limits of agreement               {md-1.96*sd:+.3f} to {md+1.96*sd:+.3f}")
    harsher = sum(1 for d in diffs if d < -0.001)
    print(f"  video judge scored lower on       {harsher}/{n} cells ({harsher/n:.0%})")

    # -- 3. discrimination: THE merge test -------------------------------
    print("\n3. DISCRIMINATION -- does the video judge split the frame judge's ties?")
    tied = [r for r in rows if r["frames"] >= 0.999]
    print(f"  cells the frame judge tied at 1.0   {len(tied)}")
    if tied:
        split = [r for r in tied if r["video"] < 0.999]
        print(f"  of those, video VFS < 1.0           {len(split)} ({len(split)/len(tied):.0%})")
        if split:
            lo = min(r["video"] for r in split)
            print(f"  spread introduced                   {lo:.2f} .. 0.99 "
                  f"(mean {st.mean([r['video'] for r in split]):.3f})")

    labelled = [r for r in rows if isinstance(r["ascs_pass"], bool)]
    if labelled:
        scores = [r["video"] for r in labelled]
        # Label 1 = ASCS FAILED. AUC > 0.5 would mean a HIGHER video score
        # predicts an ASCS failure, which would be backwards; we want the video
        # judge to score ASCS-failing cells LOWER, i.e. AUC well below 0.5.
        fails = [not r["ascs_pass"] for r in labelled]
        auc = _auc(scores, fails)
        n_fail = sum(fails)
        print(f"\n  AUC of video VFS predicting an ASCS FAILURE  {_fmt(auc)}"
              f"   (n_fail={n_fail}, n_pass={len(labelled)-n_fail})")
        if auc is not None:
            print("  0.5 = video VFS says nothing about the style verdict.")
            print("  Well below 0.5 = ASCS-failing cells score LOWER on video fidelity,")
            print("  i.e. the video judge already encodes the style verdict (merge is")
            print("  defensible). Near 0.5 = it does not (keep ASCS separate).")
        for flag in (False, True):
            grp = [r["video"] for r in labelled if bool(r["ascs_pass"]) is flag]
            if grp:
                print(f"    ASCS {'pass' if flag else 'FAIL'}: n={len(grp):<3} "
                      f"mean video VFS {st.mean(grp):.3f}  median {st.median(grp):.3f}")

    # -- 4. per style ----------------------------------------------------
    print("\n4. BY STYLE AND TARGET")
    print(f"  {'style':<24} {'tgt':<5} {'n':>3}  {'frames':>7} {'video':>7} {'diff':>7}")
    print("  " + "-" * 60)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["style"], r["target"])].append(r)
    for key, grp in sorted(buckets.items()):
        f = st.mean([r["frames"] for r in grp])
        v = st.mean([r["video"] for r in grp])
        print(f"  {key[0]:<24} {key[1]:<5} {len(grp):>3}  {f:>7.3f} {v:>7.3f} {v-f:>+7.3f}")

    # -- 5. the qualitative payload --------------------------------------
    print("\n5. LARGEST DISAGREEMENTS, with what the video judge actually saw")
    print("   (a big gap PLUS a named mid-animation defect is evidence the video")
    print("    judge is right, not evidence of noise)")
    for r in sorted(rows, key=lambda r: r["video"] - r["frames"])[:6]:
        print(f"\n  {r['style']}/{r['target']}/{r['sample']}")
        print(f"    frames {r['frames']:.3f} -> video {r['video']:.3f} "
              f"({r['video']-r['frames']:+.3f})  ascs_pass={r['ascs_pass']}")
        if r["defects"]:
            print(textwrap.fill(str(r["defects"]), 92,
                                initial_indent="    defects: ",
                                subsequent_indent="             "))

    # How often does the video judge name a defect at all?
    named = [r for r in rows if r["defects"]
             and "none observed" not in str(r["defects"]).lower()]
    print(f"\n  temporal defects named in {len(named)}/{n} cells "
          f"({len(named)/n:.0%}) -- this field has no frame-judge analogue")

    print("\n" + "=" * 78)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


def _fmt(v):
    return "  n/a" if v is None else f"{v:+.3f}"


if __name__ == "__main__":
    main()
