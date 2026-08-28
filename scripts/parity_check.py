"""Compare one model served LOCALLY against the SAME model on OpenRouter.

THE QUESTION THIS ANSWERS
-------------------------
Our local serving setup has many ways to be subtly wrong -- a mismatched chat
template, a truncating token cap, a different image encoding -- and every one of
them produces plausible output. If any is wrong, everything measured downstream
describes our configuration rather than the model. So: run the same samples
through both routes and check the metrics land in the same place.

    python3 scripts/parity_check.py --stem qwen38_27b --samples A B C

PASS BAR (stage 1, programmatic only -- no judge, so no API cost and no
sampling noise from a judging model):
  * rendering_fidelity within 0.05, and
  * csr (compile rate) identical.
Temperature is 0.2 upstream, so exact string equality is NOT the bar; two
correct runs legitimately differ. Metric-level agreement is.

WHY IT IS SAFE TO COMPARE THESE TWO AT ALL
------------------------------------------
`CachePaths` keys artifacts on the model string alone -- not base_url, not the
backend name. The v5 configs therefore give each route its own `cache_root`
(verified: all 10 paths distinct). Without that the second route would find the
first's artifacts already present, skip every stage, and this script would
compare a run against itself and always pass.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = "src/img_2_svg_pretraining/pipeline/configs"
PY = "/environments/img_2_svg_pretraining/bin/python"
CONTAINER = "animatebanana-v5"

FIDELITY_TOL = 0.05



def structural_fidelity(source_png: Path, render_png: Path) -> float | None:
    """SSIM of a route's render against the SOURCE FIGURE, with no judge.

    WHY NOT `rendering_fidelity`
    ----------------------------
    That metric is JUDGED -- it needs a VLM -- so under `--no-judge` it is never
    written and the record says `judge_skipped`. Running a judge instead would
    add API cost and, worse, judge sampling noise to a comparison whose entire
    purpose is to isolate transport differences.

    SSIM against the source is strictly better for this question: it is
    deterministic, free, and it measures the thing parity actually cares about
    -- did each route reconstruct the same figure. Absolute values are lower
    than a judge's score (a vector redraw is never pixel-identical to a raster
    figure); only the DIFFERENCE between the two routes is interpreted here.

    Reuses keyframes._ssim_map, which is cv2-based on purpose: scikit-image is
    not installed in this container and pulling it risks moving numpy off the
    1.26.4 pin that cv2 and torch are built against.
    """
    try:
        import cv2, numpy as np
        from img_2_svg_pretraining.animatebench.keyframes import _ssim_map
    except Exception:
        return None
    def load_gray(path: Path):
        """Read to grayscale, compositing any alpha onto WHITE first.

        This matters and was found the hard way: an SVG with no explicit
        background renders to transparent RGBA, and cv2's grayscale read
        flattens that to BLACK. One route's figure then measured ink=0.878
        against the source's 0.250 and scored SSIM 0.10 while looking, to the
        eye, like a correct reconstruction of the same diagram. That is a
        rendering artifact being read as a model-quality difference -- exactly
        the false signal this whole check exists to avoid.
        """
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    a, b = load_gray(source_png), load_gray(render_png)
    if a is None or b is None:
        return None
    # Compare on a common canvas. The renders differ in size from the source and
    # from each other, and SSIM needs identical shapes; downscaling both to a
    # fixed box keeps this symmetric rather than privileging either route.
    box = (1024, 1024)
    a = cv2.resize(a, box, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, box, interpolation=cv2.INTER_AREA)
    return float(np.mean(_ssim_map(a, b)))


def dexec(cmd: str, timeout: int = 5400) -> tuple[int, str]:
    """Run inside the container as US -- without -u every artifact lands
    root-owned and there is no passwordless sudo on this node to undo it."""
    import os
    full = ["docker", "exec", "-u", f"{os.getuid()}:{os.getgid()}",
            "-e", f"OPEN_ROUTER_KEY={os.environ.get('OPEN_ROUTER_KEY','')}",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN','')}",
            CONTAINER, "bash", "-lc", f"cd /code && PYTHONPATH=src {cmd}"]
    p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def run_route(stem: str, route: str, samples: list[str], style: str, force: bool):
    cfg = f"{CONFIGS}/bench_v5_{'svg' if route == 'local' else 'or'}_{stem}.yaml"
    ids = " ".join(samples)
    t0 = time.time()
    # Stage 1 only. That is where image->code happens, so it is the stage a
    # broken multimodal path or a truncating cap actually corrupts; later stages
    # are text-to-text and would dilute the signal.
    rc, out = dexec(f"{PY} -u -m img_2_svg_pretraining.pipeline.run_pipeline "
                    f"stage1 --config {cfg} --style {style} --only {ids}"
                    + (" --force" if force else ""))
    gen_s = time.time() - t0
    if rc != 0:
        print(f"  !! {route} generation exit={rc}")
        print("\n".join(out.splitlines()[-15:]))
    # --no-judge: programmatic metrics only (csr, rendering_fidelity), so this
    # costs nothing and introduces no judge sampling noise.
    rc2, out2 = dexec(f"{PY} -u -m img_2_svg_pretraining.animatebench.run_eval "
                      f"stage1 --config {cfg} --style {style} --only {ids} "
                      f"--no-judge" + (" --force" if force else ""))
    if rc2 != 0:
        print(f"  !! {route} eval exit={rc2}")
        print("\n".join(out2.splitlines()[-15:]))
    return gen_s


def read_records(stem: str, route: str, samples: list[str], style: str) -> dict:
    cache = "animatebench_v5_cache" if route == "local" else "animatebench_v5_or_cache"
    name = f"bench_v5_{'svg' if route == 'local' else 'or'}_{stem}"
    base = REPO / "data" / cache / "animatebench_v3" / "evals" / name / style
    out = {}
    for s in samples:
        f = base / s / "stage1.json"
        if f.exists():
            try:
                out[s] = json.loads(f.read_text())
            except json.JSONDecodeError:
                out[s] = {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", required=True, help="e.g. qwen38_27b")
    ap.add_argument("--samples", nargs="+", help="sample ids; default = the 3 in v5_selection.json")
    ap.add_argument("--style", default=None, help="default: each sample's primary style")
    ap.add_argument("--routes", nargs="+", default=["local", "or"])
    ap.add_argument("--force", action="store_true", help="re-generate even if artifacts exist")
    # Scoring needs cv2 (container-only); generation needs `docker` (host-only).
    # So the two halves cannot run in the same place, and --score-only is how
    # the scoring half is re-run inside the container over existing artifacts.
    ap.add_argument("--score-only", action="store_true",
                    help="skip generation; score artifacts already on disk")
    ap.add_argument("--out", help="write the JSON verdict here")
    args = ap.parse_args()

    sel = json.loads((REPO / "data/v5_selection.json").read_text())
    samples = args.samples or sel["parity"]
    # Each sample carries its own reference style; a style with no reference
    # still writes a record with every GT field null, which is indistinguishable
    # from a bad score. So group by the sample's own style rather than forcing one.
    by_style: dict[str, list[str]] = {}
    for s in samples:
        by_style.setdefault(args.style or sel["samples"][s], []).append(s)

    print(f"=== parity: {args.stem} ===")
    print(f"samples: {samples}")
    timings = {}
    if not args.score_only:
        for style, ids in by_style.items():
            for route in args.routes:
                print(f"\n-- {route} / {style} / {len(ids)} sample(s)")
                timings[f"{route}:{style}"] = round(
                    run_route(args.stem, route, ids, style, args.force), 1)

    print(f"\n{'sample':34s} {'style':22s} {'local csr':>9} {'or csr':>7} "
          f"{'local fid':>9} {'or fid':>7} {'d':>7}  verdict")
    rows, fails = [], 0
    for style, ids in by_style.items():
        loc = read_records(args.stem, "local", ids, style)
        rem = read_records(args.stem, "or", ids, style)
        for s in ids:
            l, r = loc.get(s, {}), rem.get(s, {})
            lc, rc = l.get("csr"), r.get("csr")
            # Judged fidelity if a judge ran; otherwise SSIM against the source
            # figure, which needs no judge and is deterministic.
            lf, rf = l.get("rendering_fidelity"), r.get("rendering_fidelity")
            basis = "judged"
            if lf is None or rf is None:
                src = REPO / "data/animatebench_v3" / s / "inputs" / f"{s}.png"
                if not src.exists():
                    cand = sorted((REPO / "data/animatebench_v3" / s).rglob("*.png"))
                    src = cand[0] if cand else None
                lp, rp = l.get("render_path"), r.get("render_path")
                if src and lp and rp:
                    lf = structural_fidelity(src, Path(lp))
                    rf = structural_fidelity(src, Path(rp))
                    basis = "ssim_vs_source"
            # BOTH routes failed to compile. Fidelity is undefined (there is no
            # render to measure), but the routes AGREE -- which is exactly what
            # parity asks. Scoring this MISSING would penalise the transport for
            # a model-quality failure that both sides reproduce identically.
            if lc == rc == 0.0:
                verdict, d, basis = "PASS (both fail to compile)", None, "csr_only"
            elif None in (lc, rc, lf, rf):
                verdict, d = "MISSING", None
            else:
                d = abs(lf - rf)
                ok = (lc == rc) and d <= FIDELITY_TOL
                verdict = "PASS" if ok else "FAIL"
            if not verdict.startswith("PASS"):
                fails += 1
            rows.append({"sample": s, "style": style, "local_csr": lc, "or_csr": rc,
                         "fidelity_basis": basis,
                         "local_fidelity": lf, "or_fidelity": rf,
                         "delta": round(d, 4) if d is not None else None,
                         "verdict": verdict})
            f = lambda v: "  --  " if v is None else f"{v:.3f}"
            print(f"{s:34s} {style:22s} {f(lc):>9} {f(rc):>7} {f(lf):>9} {f(rf):>7} "
                  f"{f(d):>7}  {verdict}")

    ok = fails == 0 and rows
    print(f"\n{'PARITY OK' if ok else 'PARITY FAILED'}: "
          f"{len(rows) - fails}/{len(rows)} within {FIDELITY_TOL} fidelity and equal CSR")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"stem": args.stem, "tolerance": FIDELITY_TOL, "rows": rows,
             "gen_seconds": timings, "pass": ok}, indent=2))
        print(f"wrote {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
