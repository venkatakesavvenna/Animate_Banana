"""Is ASCS measuring style compliance, or is it measuring its own aggregation?

`ascs_pass` is a strict AND over every judged frame
(`animation_quality.style_compliance`), so one DISCARD in a twenty-frame deck
fails the whole animation. That is a defensible gate only if a single discarded
frame really does mean the animation violated its style. This script tests that,
and it needs no model calls at all: every per-frame verdict and every per-rule
`followed` flag is already stored in `ascs_frame_detail`.

Read the sections in order -- each one is a precondition for trusting the next.

  0  DERIVABILITY   Is the stored `verdict` a fold over the rule flags? If not,
                    leave-one-out is undefined and section 2 is restricted to
                    the subset where it is.
  1  AGGREGATION    Pass rate under tolerant rules. The gap between "0 discards"
                    and "1 discard allowed" is the fragility of the metric: how
                    much of the verdict rests on a single frame.
  2  LEAVE-ONE-OUT  Pass rate with each rule's flags ignored. Answers whether
                    ASCS is one dominant rule wearing a metric's clothing.
  3  POSITION       Discard rate by frame position. Frame 1 gets a structurally
                    different prompt (no previous-frame line, temporal rules
                    force-marked followed), so an anomaly there is a prompt
                    confound rather than a measurement.
  4  RELIABILITY    KR-20 and Spearman-Brown split-half over the per-frame
                    accept vector. This is what "empirically unreliable" should
                    mean: low internal consistency = the gate fires on noise.
  5  REDUNDANCY     The merge test, and it is free. For `alpha_masking` both
                    VFS_POLICY and ASCS_POLICY are "all", so the two nodes
                    scored the IDENTICAL frame list. Joining them per frame
                    gives a direct answer to "does the fidelity judge already
                    see what the style judge discards on" -- which is what
                    decides whether ASCS can fold into VFS.

Nothing here writes to an eval record. It only reads.

Usage:
    python3 scripts/ascs_reliability.py [evals_root]
    python3 scripts/ascs_reliability.py --json out.json
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (REPO / "data/animatebench_v3_cache/animatebench_v3/evals")

# Directories under the evals root that are archives, not live results.
SKIP_DIRS = {"_stale_prompts_2026-08-19"}

TOLERANCES = [0, 1, 2, 3]
FRACTIONS = [0.0, 0.05, 0.10, 0.20]


# ---------------------------------------------------------------- loading

def load_cells(root: Path) -> list[dict]:
    """Every live animation record carrying per-frame ASCS detail."""
    cells = []
    for path in sorted(Path(root).glob("*/*/*/animation.json")):
        config = path.parent.parent.parent.name
        if config in SKIP_DIRS:
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        detail = rec.get("ascs_frame_detail")
        if not isinstance(detail, list) or not detail:
            continue
        cells.append({
            "config": config,
            "style": path.parent.parent.name,
            "sample": path.parent.name,
            "target": "svg" if config.endswith("_svg") else "tikz",
            "record": rec,
            "detail": detail,
        })
    return cells


def _rules(entry: dict) -> dict[str, bool]:
    """Flat {rule_name: followed} across both check blocks of one frame."""
    out = {}
    for block in ("generic_quality_checks", "style_specific_checks"):
        checks = entry.get(block)
        if not isinstance(checks, dict):
            continue
        for name, body in checks.items():
            if isinstance(body, dict) and isinstance(body.get("followed"), bool):
                out[name] = body["followed"]
    return out


def _accepts(cell: dict) -> list[bool]:
    """Per-frame accept vector, frames with an unparseable verdict dropped."""
    return [e.get("verdict") == "ACCEPT" for e in cell["detail"]
            if isinstance(e.get("verdict"), str)]


# ------------------------------------------------------- 0. derivability

def section_derivability(cells: list[dict]) -> dict:
    """Does `verdict == DISCARD` coincide with `any rule followed is False`?

    The verdict is the judge's own string, emitted alongside the flags rather
    than computed from them. If the two disagree, recomputing a verdict with a
    rule removed (section 2) is not a valid operation on those frames.
    """
    agree = disagree = no_flags = 0
    mismatches: list[tuple] = []
    for cell in cells:
        for entry in cell["detail"]:
            verdict = entry.get("verdict")
            if not isinstance(verdict, str):
                continue
            flags = _rules(entry)
            if not flags:
                no_flags += 1
                continue
            derived_discard = not all(flags.values())
            stated_discard = verdict.upper() != "ACCEPT"
            if derived_discard == stated_discard:
                agree += 1
            else:
                disagree += 1
                if len(mismatches) < 8:
                    mismatches.append((cell["style"], cell["sample"],
                                       entry.get("frame"), verdict,
                                       [k for k, v in flags.items() if not v]))
    total = agree + disagree
    rate = agree / total if total else float("nan")

    print("\n" + "=" * 78)
    print("0. DERIVABILITY -- is the verdict a fold over the rule flags?")
    print("=" * 78)
    print(f"  frames with flags     {total}")
    print(f"  verdict == fold       {agree} ({rate:.1%})")
    print(f"  verdict != fold       {disagree}")
    if no_flags:
        print(f"  frames with no flags  {no_flags}  (excluded)")
    if rate < 0.95:
        print("\n  WARNING: below 95%. Section 2's leave-one-out is only valid")
        print("  on the agreeing subset -- read it as indicative, not exact.")
    for style, sample, frame, verdict, broken in mismatches:
        print(f"    {style[:18]:18} {sample[:26]:26} {frame} "
              f"said {verdict:8} broke={broken}")
    return {"agree": agree, "disagree": disagree, "rate": rate}


# -------------------------------------------------------- 1. aggregation

def _pass_under(accepts: list[bool], k: int) -> bool:
    return (len(accepts) - sum(accepts)) <= k


def _pass_frac(accepts: list[bool], theta: float) -> bool:
    if not accepts:
        return False
    return (len(accepts) - sum(accepts)) / len(accepts) <= theta


def section_aggregation(cells: list[dict]) -> dict:
    """Pass rate under every tolerant rule, plus how many cells each flip moves.

    The k=0 -> k=1 flip count is the headline number: cells whose entire
    verdict rests on one frame out of ~20.
    """
    print("\n" + "=" * 78)
    print("1. AGGREGATION -- how much of the verdict rests on a single frame?")
    print("=" * 78)

    vectors = [(c, _accepts(c)) for c in cells]
    vectors = [(c, a) for c, a in vectors if a]
    n = len(vectors)

    print(f"\n  {n} cell(s) with at least one parseable frame verdict\n")
    print(f"  {'rule':<26} {'pass':>10}   {'rate':>6}   {'flips vs k=0':>12}")
    print("  " + "-" * 60)

    base = [_pass_under(a, 0) for _, a in vectors]
    rows = {}
    for k in TOLERANCES:
        got = [_pass_under(a, k) for _, a in vectors]
        flips = sum(1 for g, b in zip(got, base) if g != b)
        label = "allow 0 discards (CURRENT)" if k == 0 else f"allow {k} discard(s)"
        print(f"  {label:<26} {sum(got):>4}/{n:<5} {sum(got)/n:>6.1%}   "
              f"{flips:>12}")
        rows[f"allow_{k}"] = {"pass": sum(got), "n": n, "rate": sum(got) / n,
                              "flips_vs_strict": flips}

    print()
    for theta in FRACTIONS:
        got = [_pass_frac(a, theta) for _, a in vectors]
        flips = sum(1 for g, b in zip(got, base) if g != b)
        print(f"  {'<= ' + f'{theta:.0%}' + ' frames discarded':<26} "
              f"{sum(got):>4}/{n:<5} {sum(got)/n:>6.1%}   {flips:>12}")
        rows[f"frac_{theta}"] = {"pass": sum(got), "n": n, "rate": sum(got) / n,
                                 "flips_vs_strict": flips}

    # The continuous replacement for the binary gate.
    fracs = [sum(a) / len(a) for _, a in vectors]
    print(f"\n  pass_fraction (continuous)  mean {st.mean(fracs):.3f}  "
          f"median {st.median(fracs):.3f}  "
          f"min {min(fracs):.3f}  max {max(fracs):.3f}")
    print(f"  cells at pass_fraction 1.0  {sum(1 for f in fracs if f >= 0.999)}/{n}")

    # Fragility: of the cells that fail strictly, how many fail on 1 or 2 frames?
    failing = [(c, a) for c, a in vectors if not _pass_under(a, 0)]
    if failing:
        by_discards = Counter(len(a) - sum(a) for _, a in failing)
        print(f"\n  of {len(failing)} strictly-failing cell(s), discards per cell:")
        for d in sorted(by_discards):
            share = by_discards[d] / len(failing)
            print(f"    {d:>3} discard(s)  {by_discards[d]:>3} cell(s)  {share:>6.0%}")

    # Sliced by style x target -- a real style signal should not swing wildly.
    print(f"\n  {'style':<24} {'target':<6} {'n':>3}  {'k=0':>6} {'k=1':>6} {'frac':>6}")
    print("  " + "-" * 60)
    buckets = defaultdict(list)
    for c, a in vectors:
        buckets[(c["style"], c["target"])].append(a)
    for (style, target), group in sorted(buckets.items()):
        r0 = sum(_pass_under(a, 0) for a in group) / len(group)
        r1 = sum(_pass_under(a, 1) for a in group) / len(group)
        mf = st.mean([sum(a) / len(a) for a in group])
        print(f"  {style:<24} {target:<6} {len(group):>3}  "
              f"{r0:>6.0%} {r1:>6.0%} {mf:>6.2f}")

    rows["pass_fraction_mean"] = st.mean(fracs)
    return rows


# ----------------------------------------------------- 2. leave-one-out

def section_leave_one_out(cells: list[dict]) -> dict:
    """Recompute every verdict with rule r ignored; report the cell pass rate.

    If one rule's removal moves the rate to near-ceiling, ASCS is that rule.
    """
    print("\n" + "=" * 78)
    print("2. LEAVE-ONE-OUT -- is ASCS one dominant rule wearing a metric's hat?")
    print("=" * 78)

    # Which frames broke which rule.
    broke = Counter()
    for cell in cells:
        for entry in cell["detail"]:
            for name, followed in _rules(entry).items():
                if not followed:
                    broke[name] += 1

    all_rules = sorted({n for c in cells for e in c["detail"] for n in _rules(e)})

    def rate_without(skip: str | None) -> tuple[float, int, int]:
        passed = total = 0
        for cell in cells:
            frames = []
            for entry in cell["detail"]:
                flags = {k: v for k, v in _rules(entry).items() if k != skip}
                if not flags:
                    continue
                frames.append(all(flags.values()))
            if not frames:
                continue
            total += 1
            passed += all(frames)
        return (passed / total if total else float("nan")), passed, total

    base_rate, base_pass, base_n = rate_without(None)
    print(f"\n  baseline (derived, all rules)  {base_pass}/{base_n} = {base_rate:.1%}")
    print(f"\n  {'rule dropped':<32} {'discards':>8}  {'pass rate':>10}  {'delta':>7}")
    print("  " + "-" * 62)

    out = {}
    for rule in sorted(all_rules, key=lambda r: -broke[r]):
        rate, passed, total = rate_without(rule)
        delta = rate - base_rate
        mark = "  <<<" if delta >= 0.15 else ""
        print(f"  {rule:<32} {broke[rule]:>8}  {passed:>3}/{total:<3} {rate:>5.0%}"
              f"  {delta:>+6.1%}{mark}")
        out[rule] = {"discards": broke[rule], "rate": rate, "delta": delta}
    return out


# --------------------------------------------------------- 3. position

def section_position(cells: list[dict]) -> dict:
    """Discard rate by frame position, normalized by frames at that position.

    style_compliance builds a DIFFERENT prompt for position 0: no previous-frame
    reference, renumbered rules, and an instruction to mark temporal rules
    followed. An anomalous rate there is a prompt artefact, not a style finding.
    """
    print("\n" + "=" * 78)
    print("3. POSITION -- is the discard rate uniform across the deck?")
    print("=" * 78)

    at = Counter()
    disc = Counter()
    # Relative position too, since decks vary 6 to 52 frames.
    rel_at = Counter()
    rel_disc = Counter()
    for cell in cells:
        parsed = [e for e in cell["detail"] if isinstance(e.get("verdict"), str)]
        for i, entry in enumerate(parsed):
            at[i] += 1
            bucket = min(int(10 * i / max(len(parsed) - 1, 1)), 9)
            rel_at[bucket] += 1
            if entry["verdict"].upper() != "ACCEPT":
                disc[i] += 1
                rel_disc[bucket] += 1

    overall = sum(disc.values()) / sum(at.values()) if at else float("nan")
    print(f"\n  overall discard rate  {sum(disc.values())}/{sum(at.values())} "
          f"= {overall:.1%}")

    print(f"\n  absolute position (first 12)")
    print(f"  {'pos':>4} {'frames':>7} {'discards':>9} {'rate':>7}")
    print("  " + "-" * 32)
    for i in sorted(at)[:12]:
        rate = disc[i] / at[i]
        flag = "  <-- prompt differs" if i == 0 else ""
        print(f"  {i:>4} {at[i]:>7} {disc[i]:>9} {rate:>7.1%}{flag}")

    print(f"\n  relative position (deciles)")
    print(f"  {'decile':>7} {'frames':>7} {'discards':>9} {'rate':>7}")
    print("  " + "-" * 34)
    for b in sorted(rel_at):
        print(f"  {b*10:>3}-{b*10+9:<3} {rel_at[b]:>7} {rel_disc[b]:>9} "
              f"{rel_disc[b]/rel_at[b]:>7.1%}")

    first = disc[0] / at[0] if at.get(0) else float("nan")
    print(f"\n  frame 0 rate {first:.1%} vs overall {overall:.1%}  "
          f"({'ANOMALOUS' if abs(first - overall) > 0.15 else 'consistent'})")
    return {"overall": overall, "first_frame": first}


# ------------------------------------------------------ 4. reliability

"""KR-20 is deliberately NOT computed here.

It needs the variance of total scores across respondents; within one cell the
total is a scalar, so every within-cell form of it is undefined. An earlier
draft of this script printed a "mean within-cell KR-20" of +1.08 -- impossible
for a coefficient bounded above by 1, and a good reminder that a statistic
which cannot exceed its bound and does is not a weak result but a wrong one.
Split-half below is the valid measure and needs no such contortion.
"""


def _spearman(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) *
                    sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def section_reliability(cells: list[dict]) -> dict:
    """Split-half consistency of the per-frame accept vector.

    If odd frames and even frames of the same animation disagree about whether
    it complies with its style, the frames are not repeated measures of one
    construct and the AND-gate is aggregating noise.
    """
    print("\n" + "=" * 78)
    print("4. RELIABILITY -- are a cell's frames measuring the same thing?")
    print("=" * 78)

    odd_f, even_f = [], []
    for cell in cells:
        acc = _accepts(cell)
        if len(acc) < 4:
            continue
        a = acc[0::2]
        b = acc[1::2]
        odd_f.append(sum(a) / len(a))
        even_f.append(sum(b) / len(b))

    rho = _spearman(odd_f, even_f)
    print(f"\n  cells with >=4 judged frames  {len(odd_f)}")
    if rho is None:
        print("  split-half: not computable")
    else:
        sb = 2 * rho / (1 + rho) if rho > -1 else float("nan")
        print(f"  split-half Spearman rho       {rho:+.3f}")
        print(f"  Spearman-Brown corrected      {sb:+.3f}")
        verdict = ("HIGH -- frames agree; the gate is reading real signal"
                   if sb >= 0.7 else
                   "MODERATE -- partly signal, partly frame-level noise"
                   if sb >= 0.4 else
                   "LOW -- the frames disagree; a strict AND aggregates noise")
        print(f"  -> {verdict}")

    return {"split_half_rho": rho, "n": len(odd_f)}


# ------------------------------------------------------- 5. redundancy

def section_redundancy(cells: list[dict]) -> dict:
    """Does the fidelity judge already score down the frames ASCS discards?

    Only computable where both nodes used policy "all" -- the same frame list
    judged twice, independently, by the same model. That is `alpha_masking`
    (VFS_POLICY["alpha_masking"] == "all"), and it is the closest thing to a
    controlled experiment already sitting on disk.

    A strong negative point-biserial (discarded frames score lower on fidelity)
    means ASCS is largely redundant with VFS and the merge is defensible. Near
    zero means ASCS sees something VFS does not, and it must stay separate no
    matter how noisy its aggregation is.
    """
    print("\n" + "=" * 78)
    print("5. REDUNDANCY -- the merge test (free: both nodes judged the same frames)")
    print("=" * 78)

    paired: list[tuple[bool, float]] = []
    used = []
    for cell in cells:
        rec = cell["record"]
        if rec.get("vfs_frame_policy") != "all" or rec.get("ascs_frame_policy") != "all":
            continue
        vfs_frames = rec.get("vfs_frames")
        if not isinstance(vfs_frames, list):
            continue
        by_label = {f.get("frame"): f.get("score_raw") for f in vfs_frames
                    if isinstance(f, dict)}
        hits = 0
        for entry in cell["detail"]:
            label = entry.get("frame")
            verdict = entry.get("verdict")
            score = by_label.get(label)
            if not isinstance(verdict, str) or not isinstance(score, (int, float)):
                continue
            paired.append((verdict.upper() == "ACCEPT", float(score)))
            hits += 1
        if hits:
            used.append((cell["style"], cell["target"], cell["sample"], hits))

    print(f"\n  {len(used)} cell(s) where both nodes used policy 'all'")
    for style, target, sample, hits in used:
        print(f"    {style:<16} {target:<5} {sample[:34]:34} {hits:>3} paired frames")

    if len(paired) < 10:
        print("\n  too few paired frames for a correlation")
        return {"n": len(paired)}

    acc = [s for ok, s in paired if ok]
    dis = [s for ok, s in paired if not ok]
    print(f"\n  paired frames            {len(paired)}")
    print(f"  ACCEPTed by ASCS         n={len(acc):<4} mean VFS {st.mean(acc):.2f}"
          + (f"  median {st.median(acc):.2f}" if acc else ""))
    if dis:
        print(f"  DISCARDed by ASCS        n={len(dis):<4} mean VFS {st.mean(dis):.2f}"
              f"  median {st.median(dis):.2f}")
    else:
        print("  DISCARDed by ASCS        n=0  -- no discards in this subset")
        return {"n": len(paired), "r": None}

    # Point-biserial == Pearson with the binary coded 0/1.
    xs = [1.0 if ok else 0.0 for ok, _ in paired]
    ys = [s for _, s in paired]
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    r = num / den if den else float("nan")
    diff = st.mean(acc) - st.mean(dis)

    print(f"\n  point-biserial r(accept, VFS)  {r:+.3f}")
    print(f"  mean VFS gap (accept-discard)  {diff:+.2f} points on a 0-10 scale")

    # A point-biserial is only as informative as its smaller group. With a
    # handful of discards the coefficient is dominated by which single frames
    # happened to be discarded, and reporting it as "no relationship" would be
    # a conclusion the data cannot support.
    MIN_MINORITY = 8
    if min(len(acc), len(dis)) < MIN_MINORITY:
        print("\n  READ THIS AS:")
        print(f"    UNDERPOWERED, NOT A NULL RESULT. The smaller group has "
              f"{min(len(acc), len(dis))} frame(s)")
        print(f"    (need >= {MIN_MINORITY}). This subset is the only place both nodes judged")
        print("    the same frames, and it happens to be a subset ASCS almost never")
        print("    discards on. The merge question CANNOT be settled from stored data --")
        print("    it needs the video judge (which is why the video judge is worth running).")
        return {"n": len(paired), "r": r, "gap": diff, "underpowered": True,
                "n_accept": len(acc), "n_discard": len(dis)}

    print("\n  READ THIS AS:")
    if abs(r) >= 0.4:
        print("    STRONG. The fidelity judge already scores down most of what the")
        print("    style judge discards. ASCS is substantially redundant with VFS;")
        print("    merging is defensible on this evidence.")
    elif abs(r) >= 0.2:
        print("    WEAK-TO-MODERATE. Partial overlap. A merge would lose some of")
        print("    what ASCS sees; prefer keeping it and fixing the aggregation.")
    else:
        print("    NEAR ZERO. The two judges are looking at different things. ASCS")
        print("    is NOT redundant -- merging it into VFS would silently drop a")
        print("    signal, regardless of how noisy its gate is. Fix the gate instead.")
    return {"n": len(paired), "r": r, "gap": diff,
            "mean_accept": st.mean(acc), "mean_discard": st.mean(dis)}


# ------------------------------------------------------------------ main

def main() -> None:
    args = [a for a in sys.argv[1:]]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = Path(args[i + 1])
        del args[i:i + 2]
    root = Path(args[0]) if args else DEFAULT_ROOT

    cells = load_cells(root)
    if not cells:
        raise SystemExit(f"no animation records with ascs_frame_detail under {root}")

    print(f"ASCS reliability | root {root}")
    print(f"{len(cells)} cell(s): " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(c["config"] for c in cells).items())))

    out = {
        "root": str(root),
        "n_cells": len(cells),
        "derivability": section_derivability(cells),
        "aggregation": section_aggregation(cells),
        "leave_one_out": section_leave_one_out(cells),
        "position": section_position(cells),
        "reliability": section_reliability(cells),
        "redundancy": section_redundancy(cells),
    }
    print("\n" + "=" * 78)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {json_out}")


if __name__ == "__main__":
    main()
