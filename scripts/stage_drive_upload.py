"""Stage every generated animation into the folder layout for Drive upload.

    Original Images/          one source figure per sample (shared)
    Progressive_Reveal/       <sample>__<model>.mp4
    Alpha_Masking/
    Colour_Pop/
    Hopping_Bbox/
    Sliding_Bbox/

Videos are HARD-LINKED, not copied: the export tree and this staging dir sit on
the same filesystem, so the whole set costs no extra disk and is instant. A hard
link also means the staged file IS the artifact -- there is no chance of shipping
a stale copy that drifted from what the pipeline produced. Falls back to a real
copy across filesystems.

The model name is taken from the EXPORT LINEAGE directory, not guessed from a
config: the lineage is what actually produced the bytes, so a file can never be
labelled with a model that did not generate it.
"""
from __future__ import annotations

import argparse, os, re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STYLE_DIR = {
    "progressive_reveal":   "Progressive_Reveal",
    "alpha_masking":        "Alpha_Masking",
    "colour_pop":           "Colour_Pop",
    "hopping_bounding_box": "Hopping_Bbox",
    "sliding_bounding_box": "Sliding_Bbox",
}

# lineage substring -> the name a human expects to read on the file
MODELS = {
    "google-gemini-3.7-flash":          "Gemini-3.7-Flash",
    "google-gemma-4-31B-it":            "Gemma-4-31B",
    "Qwen-Qwen3.8-27B":                 "Qwen3.8-27B",
    "Qwen-Qwen3-VL-235B-A22B-Instruct": "Qwen3-VL-235B",
    "zai-org-GLM-4.6V":                 "GLM-4.6V",
    "moonshotai-Kimi-K2.6":             "Kimi-K2.6",
}


def model_of(lineage: str) -> str | None:
    # Longest first: "Qwen-Qwen3-VL-235B-A22B-Instruct" contains no other key,
    # but matching short-to-long would risk a prefix collision as models are
    # added, and a mislabelled file is worse than an unlabelled one.
    for key in sorted(MODELS, key=len, reverse=True):
        if key in lineage:
            return MODELS[key]
    return None


def style_of(lineage: str) -> str | None:
    for s in STYLE_DIR:
        if re.search(rf"__{re.escape(s)}(__|$)", lineage):
            return s
    return None


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data/drive_upload"))
    ap.add_argument("--dataset", default=str(REPO / "data/animatebench_v5"))
    ap.add_argument("--roots", default="data/animatebench_v5_cache,data/animatebench_v5_or_cache",
                    help="comma-separated cache roots to scan for exports")
    ap.add_argument("--figures", choices=("per-model", "common"), default="per-model",
                    help="'common' puts ONE top-level Original_Images folder "
                         "shared by all models instead of one per model")
    args = ap.parse_args()

    out = Path(args.out)

    # PER-MODEL tree: <Model>/Original_Images + <Model>/<Style>. Each model
    # folder is self-contained, so a reviewer can take one model's directory and
    # have both the source figures and its videos without cross-referencing.
    counts: dict[tuple[str, str], int] = {}
    skipped = 0
    videos: list[tuple[str, str, str, Path]] = []
    for root in args.roots.split(","):
        for mp4 in (REPO / root).rglob("exports/*/*/animation.mp4"):
            lineage, sample = mp4.parent.parent.name, mp4.parent.name
            model, style = model_of(lineage), style_of(lineage)
            if not model or not style:
                skipped += 1
                continue
            videos.append((model, style, sample, mp4))

    models = sorted({m for m, _, _, _ in videos})
    figures = 0
    if args.figures == "common":
        (out / "Original_Images").mkdir(parents=True, exist_ok=True)
        for s in sorted(Path(args.dataset).iterdir()):
            png = s / f"{s.name}.png"
            if png.exists():
                link(png, out / "Original_Images" / png.name)
                figures += 1
    for model in models:
        dirs = list(STYLE_DIR.values())
        if args.figures == "per-model":
            dirs = ["Original_Images", *dirs]
        for d in dirs:
            (out / model / d).mkdir(parents=True, exist_ok=True)
        if args.figures == "per-model":
            # Figures are duplicated into every model folder. They are hard
            # links, so N copies cost the bytes of one.
            for s in sorted(Path(args.dataset).iterdir()):
                png = s / f"{s.name}.png"
                if png.exists():
                    link(png, out / model / "Original_Images" / png.name)
                    figures += 1

    for model, style, sample, mp4 in videos:
        # Model is already the folder name, so the file keeps just the sample id.
        link(mp4, out / model / STYLE_DIR[style] / f"{sample}.mp4")
        counts[(style, model)] = counts.get((style, model), 0) + 1

    print(f"staged -> {out}\n")
    total = 0
    for model in models:
        fig_dir = (out / model / "Original_Images") if args.figures == "per-model" \
            else (out / "Original_Images")
        figs = len(list(fig_dir.glob("*.png")))
        row = []
        for style, folder in STYLE_DIR.items():
            n = len(list((out / model / folder).glob("*.mp4")))
            total += n
            row.append(f"{folder}={n}")
        print(f"  {model:18s} figures={figs:3d}  " + " ".join(row))
    print(f"\n  total videos: {total}" + (f"   (skipped {skipped} unrecognised lineage)" if skipped else ""))
    print(f"  size: {sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
