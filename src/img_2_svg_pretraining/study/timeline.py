"""Frame/narration timing for the study player.

The exported mp4 is unusable as a stimulus: at `fps: 2` a 15-frame animation
plays in 7.5s while its narration is ~40s of authored text. So the player never
serves the mp4 -- it drives the frame deck directly from step durations, and
this module computes the mapping once, at bundle-build time.

Deliberately free of any `pipeline` import. The study app must not depend on
the pipeline package (see study/build/ for the one subtree that may), and the
arithmetic here is a *mirror* of `pipeline/export/narration.py::align_frames_to_clips`
rather than a call into it. That duplication is intentional and load-bearing:
if the two ever disagree, a caption drifts against the frame it describes and
nobody notices until a participant rates "narration alignment" on a lie. The
mirrored rule is small enough to keep honest, and `tests/test_study_bundle.py`
asserts the invariants both sides must satisfy.

Frames and steps are NOT 1:1. The designer emits sub-frames within a step for
smooth transitions, so observed ratios include 19 steps/17 frames, 7/41 and
exact matches -- per-sample and not knowable in advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Narration pace for steps that carry no authored duration. 170 wpm is an
# unhurried reading speed for technical prose; the floor keeps a three-word
# caption on screen long enough to be seen at all.
READING_WPM = 170.0
READING_FLOOR_SECONDS = 2.5

# A step whose narrative is empty still occupies time -- dropping it would
# desync every step after it.
SILENT_STEP_SECONDS = 1.5


@dataclass
class Cue:
    """One narration step, positioned on the video clock."""
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"i": self.index, "start": round(self.start, 3),
                "end": round(self.end, 3), "text": self.text}


@dataclass
class Timeline:
    duration: float
    holds: list[float] = field(default_factory=list)       # seconds per frame
    frame_step: list[int] = field(default_factory=list)    # frame -> step index
    cues: list[Cue] = field(default_factory=list)
    timing_source: str = "authored"                        # authored | estimated | mixed

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "frames": len(self.holds),
            "holds": [round(h, 3) for h in self.holds],
            "frame_step": self.frame_step,
            "cues": [c.to_dict() for c in self.cues],
            "timing_source": self.timing_source,
        }


def reading_seconds(text: str) -> float:
    """How long `text` needs to be on screen to be read."""
    words = len((text or "").split())
    if not words:
        return SILENT_STEP_SECONDS
    return max(READING_FLOOR_SECONDS, words / (READING_WPM / 60.0))


def step_durations(nodes: list[dict]) -> tuple[list[float], str]:
    """Per-step seconds, preferring the sequence's own authored `duration`.

    Two adjustments, both deliberate:

    - A step with no authored duration is paced by its reading time.
    - An authored duration shorter than its caption's reading time is stretched
      to fit. The designer timed the *motion*, not the text, and several
      authored values land near 3s against captions that need 4s+. A caption
      that leaves the screen mid-sentence makes "narration alignment"
      unratable, which is the one thing Experiment 2 exists to measure.

    Returns the durations and which source produced them, so the manifest can
    record that a sample was paced by estimate rather than by the designer --
    an auditable difference, not one to bury.
    """
    authored_count = 0
    durations = []

    for node in nodes:
        value = node.get("duration")
        text = node.get("narrative") or ""
        if isinstance(value, (int, float)) and value > 0:
            authored_count += 1
            floor = reading_seconds(text) if text else 0.0
            durations.append(max(float(value), floor))
        else:
            durations.append(reading_seconds(text))

    if not nodes:
        source = "estimated"
    elif authored_count == len(nodes):
        source = "authored"
    elif authored_count == 0:
        source = "estimated"
    else:
        source = "mixed"
    return durations, source


def build_timeline(n_frames: int, nodes: list[dict]) -> Timeline:
    """Distribute `n_frames` across the narration steps and place the cues.

    Frames are apportioned to steps proportionally; each step's duration is
    split evenly among the frames that fall to it. A step that renders no frame
    of its own donates its time to the previous frame, which is the only option
    that keeps the caption track and the frame track the same length.
    """
    if n_frames <= 0:
        raise ValueError("need at least one frame to build a timeline")
    if not nodes:
        raise ValueError("need at least one narration step to build a timeline")

    durations, source = step_durations(nodes)
    n_steps = len(nodes)

    holds: list[float] = []
    frame_step: list[int] = []
    cues: list[Cue] = []
    carried = 0.0
    lead: list[int] = []   # steps that precede the first frame

    for i, dur in enumerate(durations):
        text = nodes[i].get("narrative") or ""
        start_f = i * n_frames // n_steps
        stop_f = (i + 1) * n_frames // n_steps
        span = stop_f - start_f

        if span == 0:
            if holds:
                # Ride on the previous frame: extend its hold and let the cue
                # occupy the tail of that frame's window.
                at = sum(holds)
                holds[-1] += dur
                cues.append(Cue(i, at, at + dur, text))
            else:
                # No frame has been emitted yet; hold the time and place these
                # cues once the first real span fixes the clock.
                carried += dur
                lead.append(i)
                cues.append(Cue(i, 0.0, 0.0, text))
            continue

        at = sum(holds)
        share = (dur + carried) / span
        # Any carried time belongs to the leading cues, which are laid out
        # below across [at, at + carried]. This step's own caption starts
        # after them, even though its frames render underneath both.
        cue_start = at + carried
        carried = 0.0
        for _ in range(span):
            holds.append(share)
            frame_step.append(i)
        cues.append(Cue(i, cue_start, sum(holds), text))

    # Leading step(s) with no frame of their own: their time was folded into
    # the first real span, so lay them out sequentially from zero.
    clock = 0.0
    for i in lead:
        cues[i].start = clock
        clock += durations[i]
        cues[i].end = clock

    return Timeline(duration=sum(holds), holds=holds, frame_step=frame_step,
                    cues=cues, timing_source=source)
