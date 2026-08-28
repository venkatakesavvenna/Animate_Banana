"""Build the mp4 a video judge actually sees.

Sending `exports/<sample>/animation.mp4` straight to Gemini looks like the
obvious thing and is wrong, for one measurable reason:

**Gemini samples inline video at 1 frame per second.** The exporter writes at
`fps: 2` (both bench configs). Across the 93 exported animations the median deck
is 16 frames -- 8 seconds at fps=2 -- so a 1 fps sampler sees about **half the
animation frames**, chosen by the sampler rather than by us, and reports a
confident score on the half it saw. Nothing in the response would reveal it.

Re-timing the same frames to 1 fps removes the problem entirely: one sampled
frame per animation frame, no dropped content. It also buys a methodological
property worth more than the convenience -- the video judge then sees *exactly
the pixels the frame judge saw*, already normalized to `DEFAULT_LONG_EDGE`, so a
frames-vs-video disagreement is attributable to the modality and not to
resolution or to which frames each judge happened to get.

`google-genai` exposes `VideoMetadata(fps=...)` as an SDK-level alternative.
Re-timing is preferred here because it is provider-independent and because the
result is verifiable by inspecting the file, rather than depending on a request
field being honoured.

The export path stays reachable via `--video-source export`, as a robustness
check on this very decision.
"""
from __future__ import annotations

from pathlib import Path

from .frames import FrameSet

# One frame per second: matches Gemini's inline-video sampling rate, so every
# frame of the deck survives into what the model is shown.
JUDGE_VIDEO_FPS = 1

# The rate the exporter authors at (`exporter.fps` in both bench configs). The
# slowdown below is expressed relative to THIS, not to JUDGE_VIDEO_FPS, because
# "play it 4x slower" is a statement about the animation as authored.
EXPORT_FPS = 2

# The style-compliance video judge sees the animation slowed 4x: 0.5 fps, so
# each authored frame occupies two seconds and Gemini's 1 fps sampler lands on
# every frame at least twice.
#
# WHAT THIS DOES NOT FIX. Slowing playback cannot recover a moment that was
# never rendered. The deck is a sampling of the CSS animation at EXPORT_FPS, so
# if a transition completes between two sampled instants, the in-transit frames
# do not exist in the deck and no playback rate will produce them. That matters
# specifically for the two bounding-box styles, where hop and slide are
# distinguishable ONLY in transit -- a slide whose transit was never sampled
# reads as a hop, confidently and with a timestamp. The fix for that is a
# denser export, not a slower video.
SLOWDOWN_FACTOR = 4
SLOWDOWN_VIDEO_FPS = EXPORT_FPS / SLOWDOWN_FACTOR


def judge_video(frames: FrameSet, out_path: Path,
                fps: float = JUDGE_VIDEO_FPS) -> Path:
    """Re-time an already-prepared FrameSet into an mp4 for a video judge.

    Cached: if `out_path` exists and is newer than every frame it was built
    from, it is reused. Mtime rather than existence, because a re-prepared deck
    beside a stale video is exactly the "newest by existence is not newest"
    failure this repo has hit before.
    """
    from img_2_svg_pretraining.pipeline.export.render import frames_to_mp4

    out_path = Path(out_path)
    paths = [Path(p) for p in frames.paths]
    if not paths:
        raise ValueError("judge_video: FrameSet has no frames")

    if out_path.exists():
        newest_input = max(p.stat().st_mtime for p in paths)
        if out_path.stat().st_mtime >= newest_input:
            return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return frames_to_mp4(paths, out_path, fps=fps)
