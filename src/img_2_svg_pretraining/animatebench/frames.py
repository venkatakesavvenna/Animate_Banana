"""Frame selection and preparation for the judged animation suite.

Three jobs, kept together because they are the three ways a frame can reach a
judge wrongly:

1. **Order.** `pdftoppm` pads page numbers only to the width of the total
   count, so a 10+ page animation sorts as frame-1, frame-10, frame-2 under a
   plain `sorted()`. Every frame consumer in this repo sorts numerically for
   that reason; this module reuses `export.render._frame_index` rather than
   growing a fourth copy of the rule.

2. **Selection.** Each node of the evaluation tree looks at a different slice:
   the last frame, every frame, or every fourth. The slice is recorded
   alongside the result, because "scored 8/10" means something different when
   one frame was seen than when thirty were.

3. **Size.** Exports are 3446x1889 at 300 dpi -- around 2 MB per frame. A judge
   run walks thousands of them, so they are downscaled first. The size is the
   quiet failure mode here: shrink too far and the diagram's text labels stop
   being legible while the judge keeps returning confident numbers. So the
   long edge is a named constant, it is written into every record, and the
   cache directory is keyed by it -- changing the size produces new files
   rather than silently reusing frames prepared at the old one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from img_2_svg_pretraining.pipeline.export.render import _frame_index

# Large enough that 8-10pt labels in a 3446px-wide figure survive the
# downscale; small enough that a few thousand calls stay affordable.
DEFAULT_LONG_EDGE = 1568
JPEG_QUALITY = 85

# Selection policies. "every_4" is the design document's own sampling rule for
# sliding bounding box, which is also what keeps the 84-frame outlier tractable.
POLICIES = ("last", "all", "every_4")


class FrameError(Exception):
    pass


@dataclass
class FrameSet:
    """The frames actually sent to a judge, and the account of how they were
    chosen. `indices` are 0-based positions in the on-disk frame list, so a
    per-frame verdict can always be traced back to the file it scored."""

    paths: list[Path]
    indices: list[int]
    source_count: int
    policy: str
    long_edge: int
    labels: list[str] = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "frame_policy": self.policy,
            "frames_available": self.source_count,
            "frames_judged": len(self.paths),
            "frame_indices": self.indices,
            "frame_labels": self.labels,
            "frame_long_edge": self.long_edge,
        }


def list_frames(frames_dir: Path) -> list[Path]:
    """Every exported frame, in playback order."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        return []
    return sorted(frames_dir.glob("*.png"), key=_frame_index)


def select(frames: list[Path], policy: str) -> tuple[list[Path], list[int]]:
    """Apply a selection policy, returning the frames and their source indices."""
    if policy not in POLICIES:
        raise FrameError(f"unknown frame policy {policy!r}; known: {list(POLICIES)}")
    if not frames:
        return [], []
    if policy == "last":
        return [frames[-1]], [len(frames) - 1]
    if policy == "every_4":
        # Keep the final frame regardless: for the box styles it is the only
        # one guaranteed to show the box at rest on its last target.
        picked = list(range(0, len(frames), 4))
        if picked[-1] != len(frames) - 1:
            picked.append(len(frames) - 1)
        return [frames[i] for i in picked], picked
    return list(frames), list(range(len(frames)))


def step_frames(frames: list[Path], n_steps: int) -> tuple[list[Path], list[int]] | None:
    """The frame for each animation timestep, or None when that is ambiguous.

    The per-timestep judges need "the frame for step i". Two frame counts are
    exactly attributable, and both are accepted:

    - `len(frames) == n_steps` -- the identity. TikZ exports one PDF page per
      declared frame, so this holds for every TikZ cell measured (delta 0
      across all of them).
    - `len(frames) == n_steps + 1` -- frame 0 is the animation's INITIAL
      state, before any step has fired, and frame i is the state after step i.
      SVG exports look like this: frames come from sampling the CSS timeline
      in a browser, and the pre-keyframe state is a real, distinct sample.
      Dropping frame 0 recovers the identity exactly. This is not a guess --
      transition i-1 -> i *is* timestep i, which is precisely what the judge
      is shown as (previous frame, current frame).

    Anything else genuinely is ambiguous (17 steps / 67 frames on one sample,
    with no stated rule for which of the four frames is "the" frame for a
    step), and returns None so the caller records the cell as unmappable.
    Guessing there would be worse than skipping: a frame taken from the wrong
    step produces a band that is wrong while looking entirely plausible, and
    nothing downstream could ever detect it.
    """
    if not frames or n_steps <= 0:
        return None
    if len(frames) == n_steps:
        return list(frames), list(range(len(frames)))
    if len(frames) == n_steps + 1:
        return list(frames[1:]), list(range(1, len(frames)))
    return None


def prepare(paths: list[Path], long_edge: int = DEFAULT_LONG_EDGE,
            cache_dir: Path | None = None) -> list[Path]:
    """Downscale frames for judging, cached.

    `cache_dir` holds the prepared frames; the caller must make it unique per
    (sample, style), since frame filenames repeat across cells. Left unset,
    they land beside the originals in `<frames>/judge_<long_edge>/`, matching
    how the inspector caches its thumbnails.

    Either way the directory is keyed by `long_edge`, so changing the size
    produces new files instead of quietly serving frames prepared at the old
    one -- the same reason artifacts here are keyed by lineage.
    """
    from PIL import Image

    out: list[Path] = []
    for source in paths:
        target_dir = (Path(cache_dir) if cache_dir is not None
                      else source.parent / f"judge_{long_edge}")
        cached = target_dir / f"{source.stem}.jpg"
        if not cached.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            image = Image.open(source).convert("RGB")
            image.thumbnail((long_edge, long_edge), Image.LANCZOS)
            image.save(cached, "JPEG", quality=JPEG_QUALITY)
        out.append(cached)
    return out


def frame_set(frames_dir: Path, policy: str,
              long_edge: int = DEFAULT_LONG_EDGE,
              cache_dir: Path | None = None) -> FrameSet:
    """Locate, select and prepare the frames one judge node will see."""
    source = list_frames(frames_dir)
    if not source:
        raise FrameError(f"no exported frames under {frames_dir}")
    picked, indices = select(source, policy)
    return FrameSet(
        paths=prepare(picked, long_edge, cache_dir),
        indices=indices,
        source_count=len(source),
        policy=policy,
        long_edge=long_edge,
        labels=[source[i].name for i in indices],
    )
