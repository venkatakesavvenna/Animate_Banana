"""Freeze pipeline artifacts into an immutable study bundle.

Run once, offline. Resolves every lineage through `CachePaths` here so the
running app never has to -- see `study/__init__` for why that boundary exists.

A bundle is immutable. New stimuli (more styles, the -K arm, a baseline, the
verified pairs) produce a NEW bundle version; an existing one is never patched,
because a trial already collected refers to media by content hash and must keep
resolving to exactly the bytes that participant saw.

    python -m img_2_svg_pretraining.study.build.build_bundle \
        --config src/img_2_svg_pretraining/pipeline/configs/bench_v3_or_svg.yaml \
        --out data/study_bundles/pilot-v1

Nothing here is style-specific: the config names the style, and `--config` may
be repeated to fold several styles or context conditions into one bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.samples import PaperSample, discover_samples
from img_2_svg_pretraining.study import BUNDLE_SCHEMA_VERSION
from img_2_svg_pretraining.study.timeline import build_timeline

# Frames ship at 300 dpi (850x1094 for SVG, up to 3280px for TikZ). The study
# renders them in a panel a few hundred pixels wide, so full resolution is
# bandwidth spent on nothing -- but zoom must still be usable on the figure.
FRAME_MAX_WIDTH = 1400
FIGURE_MAX_WIDTH = 2200
WEBP_QUALITY = 88

# Below this a baseline render is a still image rather than an animation.
MIN_BASELINE_FRAMES = 6

_DIGITS = re.compile(r"(\d+)")


def _frame_sort_key(path: Path) -> tuple:
    """Numeric ordering. pdftoppm zero-pads inconsistently, so a plain sort
    gives frame-1, frame-10, frame-2."""
    return tuple(int(p) if p.isdigit() else p for p in _DIGITS.split(path.stem))


def _sorted_frames(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.png"), key=_frame_sort_key)


class MediaStore:
    """Content-addressed media. The filename IS the hash of the bytes.

    This is blinding at the filesystem layer: a leaked directory listing
    carries no style, lineage, method or condition, so a later well-meaning
    edit cannot reintroduce a leak through a "helpful" filename.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.reused = 0

    def add_image(self, source: Path, max_width: int) -> dict:
        with Image.open(source) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                height = round(im.height * max_width / im.width)
                im = im.resize((max_width, height), Image.LANCZOS)
            width, height = im.size
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                temp_path = Path(tmp.name)
            im.save(temp_path, "WEBP", quality=WEBP_QUALITY, method=4)

        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()[:16]
        final = self.root / f"{digest}.webp"
        if final.exists():
            temp_path.unlink()
            self.reused += 1
        else:
            shutil.move(str(temp_path), final)
            self.count += 1
        return {"media_id": digest, "w": width, "h": height}


@dataclass
class BuildLog:
    """What was skipped and why. A silently smaller bundle is how a coverage
    matrix lies, so every omission is recorded rather than dropped."""
    skipped: list[dict] = field(default_factory=list)

    def skip(self, diagram_id: str, style: str, reason: str) -> None:
        self.skipped.append({"diagram_id": diagram_id, "animation_style": style,
                             "reason": reason})
        print(f"  SKIP {diagram_id} [{style}]: {reason}")


def stratification(xml_path: Path) -> dict:
    """Structural attributes used to match a retiring sample to its
    replacement. Derived from the structure XML, which is reproducible; the
    subjective fields (domain, layout_type, context_dependence) are hand
    labelled elsewhere and merged in by `--labels`."""
    if not xml_path.exists():
        return {}
    try:
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    except ET.ParseError:
        return {}

    elements = list(root.iter())[1:]        # everything below <Diagram>
    edges = [e for e in elements if "edge" in e.tag.lower()]
    rasters = [e for e in elements if "raster" in e.tag.lower()]
    nodes = [e for e in elements if e not in edges]
    depths = [int(e.get("depth", 1)) for e in elements if e.get("depth")]

    node_count = max(len(nodes), 1)
    return {
        "element_count": len(elements),
        "edge_count": len(edges),
        "node_count": len(nodes),
        "connectivity": round(len(edges) / node_count, 3),
        "hierarchy_depth": max(depths) if depths else 1,
        "has_raster": bool(rasters),
        "raster_count": len(rasters),
    }


def bucket(value: float, low: float, high: float) -> str:
    return "low" if value < low else ("high" if value >= high else "medium")


def narrative_id(diagram_id: str, style: str, method: str, context: str,
                 verification: str, version: int) -> str:
    raw = f"{diagram_id}|{style}|{method}|{context}|{verification}|v{version}"
    return "n_" + hashlib.blake2b(raw.encode(), digest_size=6).hexdigest()


def effective_context_tier(sample: PaperSample, configured: str | None) -> str:
    """The tier the run ACTUALLY used, not the one the lineage claims.

    `narrative_writer` silently downgrades to the best tier a sample supports,
    but the lineage path is built from the *configured* tier -- so an artifact
    filed under `__full` may have been produced from the image alone. Reading
    the condition off the path would mislabel the RQ3 arm at its root.
    """
    if not configured:
        return "image_only"
    if sample.supports_tier(configured):
        return configured
    available = sample.available_tiers()
    return available[-1] if available else "image_only"


def decode_video_frames(video: Path, out_dir: Path, ffmpeg: str) -> list[Path]:
    """Reference animations ship as mp4 with no source, so Experiment 5's
    "post-correction" side has to be decoded back into a frame deck to play
    through the same component as everything else."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
         str(out_dir / "frame-%03d.png")],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {result.stderr[:300]}")
    return _sorted_frames(out_dir)


def build_reference_narrative(sample: PaperSample, style: str, target: str,
                              media: MediaStore, log: BuildLog,
                              workdir: Path) -> dict | None:
    """The bench's own human-checked animation, as Experiment 5's verified side.

    It ships as an mp4 with no source, so the frames are decoded back out and
    played through the same component as everything else -- otherwise the two
    sides of the comparison would differ in how they are presented, not just in
    what they contain.

    Its narration JSON carries `timestamp` but no `duration`, so this side is
    paced by reading time while ours uses the designer's authored timings. That
    difference is recorded as `timing_source` rather than hidden.
    """
    video = (sample.directory / "reference" / "videos"
             / f"{target}__{style}__full.mp4")
    if not video.exists():
        log.skip(sample.id, style, "no style-matched reference video")
        return None

    narration = (sample.directory / "reference" / "narration"
                 / f"{style}_{sample.id}_{target}.json")
    if not narration.exists():
        log.skip(sample.id, style, "reference video has no narration")
        return None
    steps = json.loads(narration.read_text(encoding="utf-8")).get("sequence", [])
    if not steps:
        log.skip(sample.id, style, "reference narration has no steps")
        return None

    try:
        frames = decode_video_frames(video, workdir / sample.id / style, _ffmpeg())
    except RuntimeError as exc:
        log.skip(sample.id, style, "reference decode failed: %s" % exc)
        return None
    if not frames:
        log.skip(sample.id, style, "reference decoded to zero frames")
        return None

    nodes = [{"narrative": s.get("narrative") or "", "duration": None}
             for s in steps]
    timeline = build_timeline(len(frames), nodes)
    frame_media = [media.add_image(f, FRAME_MAX_WIDTH) for f in frames]
    spoken = sum(1 for n in nodes if n["narrative"].strip())

    return {
        "narrative_id": narrative_id(sample.id, style, "animatebanana",
                                     "not_applicable", "verified", 1),
        "diagram_id": sample.id, "animation_style": style,
        "method": "animatebanana",
        "context_condition": "not_applicable",
        "verification_state": "verified",
        "correction_type": None, "correction_magnitude": None,
        "narrative_version": 1,
        "frames": [m["media_id"] for m in frame_media],
        "frame_w": frame_media[0]["w"], "frame_h": frame_media[0]["h"],
        "n_frames": len(frames), "n_steps": len(nodes),
        "spoken_step_fraction": round(spoken / len(nodes), 3),
        "narration_words": sum(len(n["narrative"].split()) for n in nodes),
        "timeline": timeline.to_dict(),
        "is_attention_check": False,
        "source": "bench_reference",
    }


def build_baseline_narrative(sample: PaperSample, style: str, root: Path,
                             media: MediaStore, log: BuildLog,
                             target_seconds: float | None = None) -> dict | None:
    """The end-to-end model baseline, as Experiment 4's other side.

    Produced by `study.build.baseline_sonnet` -- one model call, figure to
    animated SVG, no pipeline. It carries no narration of its own, so the
    comparison is deliberately visual-only on both sides; Exp4's questions ask
    about preference, visual quality and pacing rather than narration.
    """
    cell = root / f"{sample.id}__{style}"
    frames_dir = cell / "frames"
    if not frames_dir.is_dir():
        log.skip(sample.id, style, "no baseline render")
        return None
    frames = _sorted_frames(frames_dir)
    if not frames:
        log.skip(sample.id, style, "baseline rendered zero frames")
        return None
    # A baseline that barely animates is a real failure of the baseline, but it
    # is not a fair Experiment 4 pair: asking "which animation is better" when
    # one side is effectively a still image measures the wrong thing. Recorded
    # in `skipped` so the generation-success rate can still be reported -- the
    # honest place for this outcome is RQ4's success statistics, not its
    # preference proportions.
    if len(frames) < MIN_BASELINE_FRAMES:
        log.skip(sample.id, style,
                 f"baseline is near-static ({len(frames)} frames) -- excluded "
                 f"from pairing, counts as a generation failure")
        return None

    # Pace the baseline to the same wall-clock as our own animation of the same
    # figure, spread evenly over its frames. A flat 0.5s/frame made it run
    # 13s against our 57s -- and "which is better paced" would then be
    # answering a question about my timing constant rather than about either
    # system. `target_seconds` comes from our narrated sequence for this cell.
    per_frame = (target_seconds / len(frames)) if target_seconds else 0.5
    nodes = [{"narrative": "", "duration": per_frame} for _ in frames]
    timeline = build_timeline(len(frames), nodes)
    frame_media = [media.add_image(f, FRAME_MAX_WIDTH) for f in frames]

    return {
        "narrative_id": narrative_id(sample.id, style, "baseline",
                                     "not_applicable", "not_applicable", 1),
        "diagram_id": sample.id, "animation_style": style,
        "method": "baseline",
        "context_condition": "not_applicable",
        "verification_state": "not_applicable",
        "correction_type": None, "correction_magnitude": None,
        "narrative_version": 1,
        "frames": [m["media_id"] for m in frame_media],
        "frame_w": frame_media[0]["w"], "frame_h": frame_media[0]["h"],
        "n_frames": len(frames), "n_steps": len(nodes),
        "spoken_step_fraction": 0.0, "narration_words": 0,
        "timeline": timeline.to_dict(),
        "is_attention_check": False,
        "source": "baseline_sonnet",
    }


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_narrative(sample: PaperSample, paths: CachePaths, style: str,
                    media: MediaStore, log: BuildLog, *, method: str,
                    context_condition: str, verification_state: str) -> dict | None:
    """One playable stimulus: frames, timeline, and its condition labels."""
    frames_dir = paths.exports(sample.id) / "frames"
    if not frames_dir.is_dir():
        log.skip(sample.id, style, "no frames directory")
        return None
    frames = _sorted_frames(frames_dir)
    if not frames:
        log.skip(sample.id, style, "frames directory is empty")
        return None

    seq_path = paths.sequence_narrated(sample.id)
    if not seq_path.exists():
        log.skip(sample.id, style, "no narrated sequence")
        return None
    nodes = json.loads(seq_path.read_text(encoding="utf-8")).get("nodes", [])
    if not nodes:
        log.skip(sample.id, style, "narrated sequence has no steps")
        return None

    spoken = sum(1 for n in nodes if (n.get("narrative") or "").strip())
    timeline = build_timeline(len(frames), nodes)
    frame_media = [media.add_image(f, FRAME_MAX_WIDTH) for f in frames]

    return {
        "narrative_id": narrative_id(sample.id, style, method, context_condition,
                                     verification_state, 1),
        "diagram_id": sample.id,
        "animation_style": style,
        "method": method,
        "context_condition": context_condition,
        "verification_state": verification_state,
        "correction_type": None,
        "correction_magnitude": None,
        "narrative_version": 1,
        "frames": [m["media_id"] for m in frame_media],
        "frame_w": frame_media[0]["w"],
        "frame_h": frame_media[0]["h"],
        "n_frames": len(frames),
        "n_steps": len(nodes),
        # The arch00389 guard: half its steps are silent in two styles. A
        # narrative that only speaks for half its runtime is not comparable to
        # one that speaks throughout, and must be visible rather than averaged in.
        "spoken_step_fraction": round(spoken / len(nodes), 3),
        "narration_words": sum(len((n.get("narrative") or "").split()) for n in nodes),
        "timeline": timeline.to_dict(),
        "is_attention_check": False,
    }


def build(configs: list[tuple], out: Path, name: str, labels_path: str | None,
          method: str, context_arm: str, verification_state: str,
          reference: bool = False, only: list | None = None,
          baseline_root: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    media = MediaStore(out / "media")
    log = BuildLog()

    labels = {}
    if labels_path and Path(labels_path).exists():
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))

    diagrams: dict[str, dict] = {}
    narratives: list[dict] = []
    refdir = Path(tempfile.mkdtemp(prefix="study_refdecode_"))
    emitted_reference: set = set()
    emitted_baseline: set = set()

    for config_path, style_override, config_arm in configs:
        cfg = load_config(config_path)
        if style_override:
            # The style lives in the sequence lineage, so overriding it here is
            # what lets one config enumerate every style that was generated
            # from it, rather than needing five near-identical config files.
            cfg.style = style_override
            cfg.raw["animation_style"] = style_override
        paths = CachePaths.from_config(cfg)
        style = cfg.style
        samples = discover_samples(cfg.dataset_root)
        writer = cfg.agents.get("narrative_writer")
        configured_tier = writer.option("context_tier", None) if writer else None

        print(f"\n[{Path(config_path).name}] target={cfg.target} style={style} "
              f"samples={len(samples)}")

        for sample in samples:
            if only and sample.id not in only:
                continue
            if not (paths.exports(sample.id) / "frames").is_dir():
                continue    # not an error: most samples simply were not run

            tier = effective_context_tier(sample, configured_tier)
            record = build_narrative(
                sample, paths, style, media, log,
                method=method,
                # `context_condition` is an EXPERIMENT arm, not a per-sample
                # property. Deriving it from each sample's effective tier would
                # label a plain Exp1/Exp2 bundle as a mix of with/without
                # context and read as an RQ3 pair that was never generated.
                # The factual tier is recorded separately, always.
                context_condition=config_arm,
                verification_state=verification_state)
            if record is None:
                continue
            record["effective_context_tier"] = tier
            narratives.append(record)

            # The reference and the baseline belong to the DIAGRAM, not to a
            # config. With +K and -K configs both in the build they were
            # emitted once per config -- same narrative_id twice, double the
            # decode, and any per-narrative count silently doubled.
            emitted_key = (sample.id, style)

            if baseline_root and emitted_key not in emitted_baseline:
                base = build_baseline_narrative(
                    sample, style, Path(baseline_root), media, log,
                    target_seconds=record["timeline"]["duration"])
                if base is not None:
                    narratives.append(base)
                    emitted_baseline.add(emitted_key)

            if reference and emitted_key not in emitted_reference:
                # Only pair a reference in when OUR side exists: an unmatched
                # verified narrative would sit in the bundle forming no pair
                # and inflating the apparent pool.
                ref = build_reference_narrative(sample, style, cfg.target,
                                                media, log, refdir)
                if ref is not None:
                    emitted_reference.add(emitted_key)
                    record["verification_state"] = "pre_verification"
                    record["narrative_id"] = narrative_id(
                        sample.id, style, record["method"],
                        record["context_condition"], "pre_verification", 1)
                    narratives.append(ref)

            if sample.id not in diagrams:
                figure = media.add_image(sample.image_path, FIGURE_MAX_WIDTH)
                strat = stratification(paths.xml(sample.id))
                entry = {
                    "diagram_id": sample.id,
                    "title": sample.title,
                    "figure_media_id": figure["media_id"],
                    "figure_w": figure["w"], "figure_h": figure["h"],
                    "source_collection": sample.id.split("_")[0],
                    "available_context_tiers": sample.available_tiers(),
                    **strat,
                }
                if strat:
                    entry["element_density"] = bucket(strat["element_count"], 15, 40)
                    entry["connectivity_level"] = bucket(strat["connectivity"], 0.5, 1.2)
                entry.update(labels.get(sample.id, {}))
                diagrams[sample.id] = entry

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": name,
        "configs": [{"config": str(c), "style": st, "context_arm": arm}
                    for c, st, arm in configs],
        "diagrams": sorted(diagrams.values(), key=lambda d: d["diagram_id"]),
        "narratives": narratives,
        "attention_checks": [],
        "method": method,
        "context_arm": context_arm,
        "verification_state": verification_state,
        "skipped": log.skipped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.rmtree(refdir, ignore_errors=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                    help="pipeline config; repeat for several conditions. "
                         "Suffix with '=with_context' or '=without_context' to "
                         "label that config's runs as an Experiment 3 arm.")
    ap.add_argument("--style", action="append", default=None,
                    help="animation style to build from each config; repeat to "
                         "sweep several. Defaults to whatever the config names.")
    ap.add_argument("--out", required=True, help="bundle directory to create")
    ap.add_argument("--name", default=None, help="bundle id (default: --out basename)")
    ap.add_argument("--labels", default=None,
                    help="json of hand-labelled per-diagram fields to merge")
    ap.add_argument("--method", default="animatebanana",
                    help="which system produced these (Exp4 baseline arm)")
    ap.add_argument("--context-arm", default="not_applicable",
                    choices=["not_applicable", "with_context", "without_context"],
                    help="Exp3 arm these runs belong to. Left unset for a plain "
                         "Exp1/Exp2 bundle -- it is a property of the RUN, not "
                         "of each sample's available context")
    ap.add_argument("--reference", action="store_true",
                    help="also decode the bench reference animation as the "
                         "verified side of Experiment 5")
    ap.add_argument("--baseline-root", default=None,
                    help="directory of end-to-end baseline renders, added as "
                         "the other side of Experiment 4")
    ap.add_argument("--only", default=None,
                    help="comma-separated diagram ids to restrict the build to")
    ap.add_argument("--verification-state", default="not_applicable",
                    choices=["not_applicable", "pre_verification", "verified"],
                    help="Exp5 side these runs represent")
    args = ap.parse_args()

    out = Path(args.out)
    styles = args.style or [None]
    pairs = []
    for spec in args.config:
        cfg_path, _, arm = spec.partition("=")
        for st in styles:
            pairs.append((cfg_path, st, arm or args.context_arm))
    manifest = build(pairs, out, args.name or out.name, args.labels,
                     args.method, args.context_arm, args.verification_state,
                     reference=args.reference, baseline_root=args.baseline_root,
                     only=[x.strip() for x in args.only.split(",")] if args.only else None)

    print(f"\nbundle: {out}")
    print(f"  diagrams   {len(manifest['diagrams'])}")
    print(f"  narratives {len(manifest['narratives'])}")
    print(f"  skipped    {len(manifest['skipped'])}")


if __name__ == "__main__":
    main()
