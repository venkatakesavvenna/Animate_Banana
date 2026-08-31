"""Stage 3d -- Diagram Exporter.

Turns validated animation code into deliverables: PDF, MP4, GIF, PPTX. This
is the one agent with no model in it -- purely deterministic rendering, so it
is also the cheapest stage to re-run.

The chain for TikZ is: animated source -> unroll `\\multiframe` into one page
per frame -> compile to a multipage PDF -> rasterize every page -> assemble.
"""
from __future__ import annotations

from pathlib import Path

from ..config import PipelineConfig
from ..export.render import (
    DEFAULT_DPI, DEFAULT_FPS, RenderError, compile_pdf, frames_to_gif,
    frames_to_mp4, pdf_to_frames, pdf_to_pptx,
)
from ..export.tikz_source import frame_count, to_multipage_pdf_source
from ..runner import AgentContext, SampleOutcome, StageReport
from ..samples import PaperSample

AGENT = "exporter"


def _export_svg(ctx: AgentContext, sample: PaperSample, code: str, out_dir: Path,
                formats: list[str], fps: int) -> tuple[Path, str]:
    """Export an animated SVG.

    Only frame *production* differs from the TikZ path: the designer emits CSS
    `@keyframes`, which no rasterizer can seek, so frames come from a headless
    browser instead of a multipage PDF. Assembly reuses `frames_to_mp4` /
    `frames_to_gif` unchanged, so both targets produce the same deliverables
    from the same code.

    pdf and pptx are skipped rather than faked -- both go through the TikZ
    PDF chain, and claiming to produce them would be worse than saying so.
    """
    from ..export.svg_frames import render_svg_frames

    # `style` selects the sampling policy: fade styles get one frame per
    # state midpoint, sliding also gets in-transit frames. Without it every
    # style shares one uniform grid, which drops subtitles and makes a slide
    # look like a hop.
    # `steps/` is a SECOND deck with exactly one frame per sequence timestep.
    # The main `frames/` deck samples every animation state (many more frames
    # than steps for a slide), which the video judges want but which breaks the
    # strict n-or-n+1 mapping `step_frames` requires. Emitting both keeps the
    # per-step judges exact without weakening that guard.
    n_steps = None
    for cand in (ctx.paths.sequence_narrated(sample.id),
                 ctx.paths.sequence_final(sample.id),
                 ctx.paths.sequence(sample.id)):
        if cand.exists():
            try:
                from ..schema import AnimationSequence
                n_steps = len(AnimationSequence.load(cand).traversal)
            except Exception:                  # noqa: BLE001 - optional deck
                n_steps = None
            break
    frames, used_fps = render_svg_frames(code, out_dir / "frames", fps=fps,
                                         style=ctx.cfg.style,
                                         steps_dir=out_dir / "steps",
                                         n_steps=n_steps)
    if not frames:
        raise RenderError(f"no frames rendered for '{sample.id}'")

    produced = []
    if "frames" in formats:
        produced.append("frames")
    if "mp4" in formats:
        frames_to_mp4(frames, out_dir / "animation.mp4", fps=used_fps)
        produced.append("mp4")
    if "gif" in formats:
        frames_to_gif(frames, out_dir / "animation.gif", fps=used_fps)
        produced.append("gif")
    if "narrated_mp4" in formats:
        produced.append(_export_narrated(ctx, sample, frames, out_dir))

    skipped = sorted({"pdf", "pptx"} & set(formats))
    detail = f"{len(frames)} frames @ {used_fps}fps -> {', '.join(produced) or 'nothing'}"
    if skipped:
        detail += f" (no SVG path for {', '.join(skipped)})"
    return out_dir, detail


def _export_narrated(ctx: AgentContext, sample: PaperSample, frames: list[Path],
                     out_dir: Path) -> str:
    """Synthesize the narration and mux it against the frames.

    Returns a short status note rather than raising, so a missing narration
    script or an exhausted TTS quota costs only the narrated video -- the PDF,
    MP4, GIF and PPTX for this sample are already written by this point and
    must not be discarded over the audio track.
    """
    from ..export.narration import (
        NarrationError, mux_narrated_video, synthesize_narration,
    )

    script = ctx.paths.narration_jsonl(sample.id)
    if not script.exists():
        return "narrated_mp4 skipped (no narration; run the narrate stage)"

    try:
        keys = _tts_keys(ctx)
        clips = synthesize_narration(script, ctx.paths.audio(sample.id), api_keys=keys)
        mux_narrated_video(frames, clips, out_dir / "animation_narrated.mp4")
        spoken = sum(1 for c in clips if c.text)
        return f"narrated_mp4 ({spoken}/{len(clips)} steps spoken)"
    except NarrationError as e:
        return f"narrated_mp4 failed ({e})"


def _tts_keys(ctx: AgentContext) -> list[str]:
    """API keys for TTS, from the same pool the chat backends draw on."""
    from ..backends.keys import resolve_keys

    # Prefer whatever backend the narrative writer used, so TTS follows the
    # same credentials as the text it is speaking.
    writer = ctx.cfg.agents.get("narrative_writer")
    backend_name = writer.backend if writer else None
    cfg = ctx.cfg.backend_cfg(backend_name) if backend_name else {}
    return resolve_keys("gemini", cfg, default_env="GOOGLE_API_KEY")


def export_sample(ctx: AgentContext, sample: PaperSample, formats: list[str],
                  fps: int, dpi: int) -> tuple[Path, str]:
    """Render one sample's animation. Returns (output dir, detail)."""
    source_path = ctx.paths.animation_final(sample.id)
    if not source_path.exists():
        source_path = ctx.paths.animation(sample.id)
    if not source_path.exists():
        raise FileNotFoundError(
            f"no animation code for '{sample.id}'. Run the design stage first."
        )

    code = source_path.read_text(encoding="utf-8")
    out_dir = ctx.paths.exports(sample.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx.cfg.target == "svg":
        return _export_svg(ctx, sample, code, out_dir, formats, fps)

    frames_declared = frame_count(code)
    pdf_path = compile_pdf(to_multipage_pdf_source(code), out_dir / "animation.pdf")

    produced = ["pdf"]
    # "frames" is its own output token, not merely a by-product of the video
    # formats. Without it there is no way to ask for a slide deck: `outputs: []`
    # compiles the PDF (line above is unconditional) but leaves `needs_frames`
    # empty, so zero PNGs are written and a frame scrubber has nothing to show.
    needs_frames = {"frames", "mp4", "gif", "narrated_mp4"} & set(formats)
    frames: list[Path] = []
    if needs_frames:
        frames = pdf_to_frames(pdf_path, out_dir / "frames", dpi=dpi)
    if "frames" in formats:
        produced.append("frames")

    if "mp4" in formats:
        frames_to_mp4(frames, out_dir / "animation.mp4", fps=fps)
        produced.append("mp4")
    if "gif" in formats:
        frames_to_gif(frames, out_dir / "animation.gif", fps=fps)
        produced.append("gif")
    if "pptx" in formats:
        pdf_to_pptx(pdf_path, out_dir / "animation.pptx", dpi=dpi)
        produced.append("pptx")

    if "narrated_mp4" in formats:
        note = _export_narrated(ctx, sample, frames, out_dir)
        produced.append(note)

    detail = f"{len(frames) or '?'} frames"
    if frames_declared and frames and len(frames) != frames_declared:
        # Worth surfacing: the source declared N frames but the PDF has a
        # different page count, which usually means the \multiframe rewrite
        # or the frame gating is off.
        detail += f" (source declared {frames_declared})"
    detail += f" -> {', '.join(produced)}"
    return out_dir, detail


def run(cfg: PipelineConfig, samples: list[PaperSample], force: bool = False) -> StageReport:
    ctx = AgentContext(cfg, AGENT)
    report = StageReport(agent=AGENT)

    formats = list(ctx.agent.option("outputs", ["pdf", "mp4"]))
    fps = int(ctx.agent.option("fps", DEFAULT_FPS))
    dpi = int(ctx.agent.option("dpi", DEFAULT_DPI))

    # The cached-artifact marker differs by target: TikZ always writes a PDF
    # (every other format derives from it), while the SVG path has no PDF at
    # all and the mp4 is the thing worth not rebuilding.
    #
    # A frames-only run writes neither, so both would re-render every time --
    # which matters most for SVG, where frames come from a headless browser.
    # Fall back to the frames directory as the marker in that case.
    if not ({"mp4", "gif", "pptx", "narrated_mp4"} & set(formats)):
        marker = "frames"
    else:
        marker = "animation.pdf" if cfg.target == "tikz" else "animation.mp4"

    for sample in samples:
        out_dir = ctx.paths.exports(sample.id)
        cached = out_dir / marker
        # `frames` is a directory, and an empty one means a failed render, not
        # a cached result worth honouring.
        hit = (cached.is_dir() and any(cached.glob("*.png"))
               if marker == "frames" else cached.exists())
        if hit and not force:
            report.outcomes.append(SampleOutcome(sample.id, "skipped", out_dir, "cached"))
            continue
        try:
            path, detail = export_sample(ctx, sample, formats, fps, dpi)
            report.outcomes.append(SampleOutcome(sample.id, "ok", path, detail))
        except (RenderError, FileNotFoundError, Exception) as e:
            report.outcomes.append(
                SampleOutcome(sample.id, "failed", None, f"{type(e).__name__}: {e}"))

    return report
