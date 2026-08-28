"""Recover a deck of discrete animation states from a video.

`frames.py` decides *which of an existing deck* a judge sees. This module makes
a deck where none exists, which is a different job and lives in a different
file.

WHY IT IS NEEDED
----------------
The 64 ground-truth reference animations under
`<sample>/reference/videos/<target>__<style>__full.mp4` ship as video and
nothing else -- there are no exported frames beside them. Every per-frame node
in the animation tree (VFS, ASCS, omission, repetition) consumes a frame deck,
so without extraction the reference animations cannot be scored per frame at
all, and prediction-vs-reference comparison on those nodes is impossible.

Sampling the timeline every N frames would technically produce a deck, but a
bad one: these animations move and then hold, so uniform sampling lands
arbitrarily many frames mid-transition (blurred, half-drawn) and may miss a
settled state entirely. What the judges are asked about -- "is this element
present", "did the box hop or slide" -- is a question about *settled states*.
So each style gets a detector matched to how it actually moves.

THE THREE DETECTORS
-------------------
  sliding_bounding_box  motion-centroid tracking. A sliding box never settles
                        between waypoints, so SSIM-settle finds nothing; track
                        the moving blob's centroid, save when its velocity
                        drops, and save mid-slide after enough travel so a long
                        traverse is not one undifferentiated event.
  progressive_reveal    ink-coverage steps plus local SSIM. Content only ever
                        accumulates, so coverage against the first frame is a
                        monotone progress signal that a settle detector alone
                        would miss when reveals overlap.
  everything else       local-SSIM settle detection. Hopping boxes, alpha
                        masking and colour pop all have clean animate->settle
                        cycles. LOCAL (windowed) SSIM rather than a global
                        mean, because a fade or colour change confined to a
                        small region is diluted away by a global average.

These are the extractors from the project's Notion page, kept faithful to their
tuned thresholds; what is added here is the integration contract below and the
provenance sidecar.

INTEGRATION CONTRACT
--------------------
Output is written as `frame-01.png ... frame-NN.png`, zero-padded to the width
of the total count, so the directory is a drop-in `frames_dir` for
`frames.list_frames()` / `frames.frame_set()`. That padding is load-bearing:
`list_frames` sorts by `export.render._frame_index`, which concatenates every
digit in the stem, and the originals' mixed `0_frame_0000.png` /
`state_0001_frame_0012.png` scheme yields indices 0 and 10012 -- ordering that
holds by luck and breaks the moment a name changes. The source frame index is
preserved in the sidecar rather than in the filename.

Every run writes `keyframes.json` beside the PNGs recording the method, every
threshold, the source video's hash, and which source frames were chosen. A deck
produced by tunable heuristics and no record of the tuning is not auditable.

cv2/numpy are imported INSIDE the functions, the same way `frames.prepare()`
imports PIL lazily: the bare host has neither, and `run_eval` must keep working
there for every node that does not need them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .frames import FrameSet, list_frames

# Style -> extractor. Keyed by the pipeline's own style names.
EXTRACTORS = {
    "sliding_bounding_box": "motion_centroid",
    "progressive_reveal": "coverage_ssim",
    "hopping_bounding_box": "ssim_settle",
    "colour_pop": "ssim_settle",
    "alpha_masking": "ssim_settle",
}


class KeyframeError(Exception):
    pass


@dataclass
class Keyframes:
    """An extracted deck and the full account of how it was cut."""
    paths: list[Path]
    source_indices: list[int]
    source_frame_count: int
    method: str
    style: str
    video: Path
    video_sha256: str
    params: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        return {
            "keyframe_method": self.method,
            "keyframe_style": self.style,
            "keyframes_extracted": len(self.paths),
            "keyframe_source_indices": self.source_indices,
            "keyframe_source_frames": self.source_frame_count,
            "keyframe_video": str(self.video),
            "keyframe_video_sha256": self.video_sha256,
            "keyframe_params": self.params,
        }

    def frame_set(self, long_edge: int | None = None,
                  cache_dir: Path | None = None) -> FrameSet:
        """Hand the extracted deck to the judging pipeline unchanged."""
        from .frames import DEFAULT_LONG_EDGE, frame_set
        return frame_set(self.paths[0].parent, "all",
                         long_edge or DEFAULT_LONG_EDGE, cache_dir=cache_dir)


# -- SSIM ------------------------------------------------------------------

def _ssim_map(gray_a, gray_b, win_size: int = 7):
    """Per-pixel SSIM map, matching skimage's defaults for uint8 input.

    scikit-image is NOT installed in the pipeline container, and installing it
    is the wrong trade: it pulls scipy/networkx/tifffile and can move numpy off
    the 1.26.4 pin that cv2 and torch are built against. The container also sets
    PIP_CONSTRAINT (pinning typing-extensions==4.12.2), which has already broken
    google-genai once in a way that looked nothing like a dependency problem.
    Only one symbol was ever needed, so it is implemented here on cv2
    primitives.

    skimage is still preferred when present, and the parameters below mirror
    `structural_similarity(..., gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=255)` so a threshold tuned against
    one implementation transfers to the other. Getting that wrong would silently
    change every extracted deck.
    """
    import cv2
    import numpy as np

    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    data_range = 255.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # skimage uses truncate=3.5 with sigma=1.5 -> an 11x11 window.
    sigma = 1.5
    ksize = (11, 11)
    blur = lambda x: cv2.GaussianBlur(x, ksize, sigma, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = blur(a * a) - mu_aa
    sigma_bb = blur(b * b) - mu_bb
    sigma_ab = blur(a * b) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    return num / den


def _local_change_fraction(gray_a, gray_b, win_size: int = 7,
                           local_ssim_cutoff: float = 0.85) -> float:
    """Fraction of the frame whose local SSIM fell below the cutoff."""
    import numpy as np

    try:
        from skimage.metrics import structural_similarity
        _, ssim_map = structural_similarity(
            gray_a, gray_b, data_range=255, win_size=win_size, full=True)
    except ImportError:
        ssim_map = _ssim_map(gray_a, gray_b, win_size)
    return float(np.mean(ssim_map < local_ssim_cutoff))


# -- decoding --------------------------------------------------------------

def _decode(video_path: Path):
    """Yield BGR frames. One seam, so the decoder is swappable in one place.

    cv2.VideoCapture is used first and verified to open the exporter's libx264
    output. There is no ffmpeg or ffprobe binary on the host or in the
    container, so imageio (already a dependency, and what wrote these files) is
    the fallback rather than a shell-out.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            return
        finally:
            cap.release()

    import numpy as np
    try:
        import imageio.v2 as imageio
    except ImportError as exc:                       # pragma: no cover
        raise KeyframeError(
            f"cannot decode {video_path}: cv2 could not open it and imageio "
            "is unavailable") from exc
    reader = imageio.get_reader(str(video_path))
    try:
        for rgb in reader:
            yield np.asarray(rgb)[:, :, ::-1]        # RGB -> BGR
    finally:
        reader.close()


def _gray(frame):
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _activity_series(video_path: Path, win_size: int = 7,
                     local_ssim_cutoff: float = 0.85) -> list[float]:
    """Local-SSIM change fraction for every consecutive pair, in one pass.

    Computed once and reused by both the dwell guard and the settle detector.
    Frames are not retained (a 4236x4236 deck is gigabytes); the caller decodes
    a second time to write, which is far cheaper than a second SSIM pass.
    """
    out: list[float] = []
    prev = None
    for frame in _decode(video_path):
        gray = _gray(frame)
        if prev is not None:
            out.append(_local_change_fraction(prev, gray, win_size,
                                              local_ssim_cutoff))
        prev = gray
    return out


def adaptive_activity_threshold(activities: list[float],
                                floor: float = 1e-5,
                                share: float = 0.25) -> float:
    """A motion threshold scaled to this video's own activity range.

    The published extractors use a fixed `activity_threshold=0.01` -- one
    percent of the frame must change to count as motion. That number does not
    survive contact with this data. A hopping bounding box on a 4236x4236
    figure has a **maximum** activity of 0.0027, so at a fixed 0.01 the
    animation is classified as perfectly static from beginning to end, the
    settle detector never arms, and a 32-frame video extracts to one keyframe.
    The threshold is not measuring motion, it is measuring what fraction of the
    canvas the moving object happens to occupy.

    Taking a share of the 95th percentile instead makes the decision
    scale-invariant: it asks "is this frame moving, relative to how much this
    video ever moves", which is the question the state machine actually needs.
    p95 rather than max so one flash-cut cannot set the scale for the rest.
    """
    if not activities:
        return float("inf")
    ordered = sorted(activities)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    if p95 <= floor:
        return float("inf")          # genuinely static: nothing to detect
    return max(floor, share * p95)


def _largest_diff_centroid(gray_a, gray_b, diff_thresh: int = 15,
                           min_area: int = 25):
    """Centroid and rect of the largest moving region between two frames.

    Frame differencing rather than optical flow: the subject is a synthetic
    diagram with hard edges and no texture, where flow is both slower and less
    reliable than a thresholded absolute difference.
    """
    import cv2
    import numpy as np

    diff = cv2.absdiff(gray_a, gray_b)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return (x + w / 2.0, y + h / 2.0), (x, y, w, h)


# -- the three detectors ---------------------------------------------------
# Each returns (kept_frames, source_indices, total_decoded). Writing is done
# once, centrally, so all three share the naming and sidecar contract.

def _extract_motion_centroid(video_path: Path, *, diff_thresh: int = 15,
                             min_area: int = 25,
                             velocity_threshold: float = 0.5,
                             settle_frames: int = 8,
                             min_travel_between_saves: float = 250.0):
    """Sliding bounding box: save where the moving blob settles.

    `min_travel_between_saves` is the fallback that keeps a long uninterrupted
    traverse from collapsing into a single event -- without it, a box that
    slides across the whole figure without pausing yields one keyframe.
    """
    import numpy as np

    kept, indices = [], []
    prev_gray = None
    was_moving = False
    static = 0
    last_saved_centroid = None
    last_centroid = None
    total = 0

    for i, frame in enumerate(_decode(video_path)):
        total += 1
        gray = _gray(frame)
        if prev_gray is None:
            prev_gray = gray
            kept.append(frame)                       # the initial state
            indices.append(i)
            continue

        result = _largest_diff_centroid(prev_gray, gray, diff_thresh, min_area)
        if result is None:
            velocity, centroid = 0.0, last_centroid
        else:
            centroid, _ = result
            velocity = (np.hypot(centroid[0] - last_centroid[0],
                                 centroid[1] - last_centroid[1])
                        if last_centroid is not None else 0.0)
            last_centroid = centroid

        # `last_saved_centroid` must be seeded from the first centroid seen.
        # In the published script it starts as None and is only ever assigned
        # inside a save branch, so on a video that never settles -- exactly the
        # case the min_travel fallback was written for -- the fallback's own
        # `is not None` guard can never become true and the whole traverse
        # collapses to the single initial frame. Measured: a 57-frame sliding
        # box with per-frame velocities of 400-3600px extracted one keyframe.
        if last_saved_centroid is None and centroid is not None:
            last_saved_centroid = centroid

        if velocity > velocity_threshold:
            was_moving = True
            static = 0
            if centroid is not None and last_saved_centroid is not None:
                travelled = np.hypot(centroid[0] - last_saved_centroid[0],
                                     centroid[1] - last_saved_centroid[1])
                if travelled > min_travel_between_saves:
                    kept.append(frame)
                    indices.append(i)
                    last_saved_centroid = centroid
        elif was_moving:
            static += 1
            if static >= settle_frames:
                kept.append(frame)
                indices.append(i)
                last_saved_centroid = centroid
                was_moving = False
                static = 0

        prev_gray = gray

    return kept, indices, total


def _extract_coverage_ssim(video_path: Path, *, coverage_step: float = 0.024,
                           diff_thresh: int = 15,
                           fallback_anchor_threshold: float = 0.03,
                           dedupe_threshold: float = 0.025,
                           win_size: int = 7,
                           local_ssim_cutoff: float = 0.95,
                           settle_frames: int = 1):
    """Progressive reveal: coverage steps, with a settle fallback and a dedupe.

    Coverage is measured against the FIRST frame, which for this style is the
    emptiest, so it rises monotonically as content is revealed. The dedupe
    guard compares against the last *saved* frame rather than the last seen
    one, so a trigger that fires on an imperceptible change does not add a
    visual duplicate to the deck.
    """
    import cv2
    import numpy as np

    kept, indices = [], []
    blank_gray = None
    prev_gray = anchor_gray = last_saved_gray = None
    last_saved_coverage = 0.0
    is_animating = False
    static = 0
    total = 0

    for i, frame in enumerate(_decode(video_path)):
        total += 1
        gray = _gray(frame)
        if blank_gray is None:
            blank_gray = prev_gray = anchor_gray = last_saved_gray = gray
            total_px = gray.shape[0] * gray.shape[1]
            kept.append(frame)
            indices.append(i)
            continue

        diff_from_blank = cv2.absdiff(blank_gray, gray)
        coverage = float(np.count_nonzero(diff_from_blank > diff_thresh)) / total_px

        trigger = False
        if coverage - last_saved_coverage >= coverage_step:
            trigger = True
        else:
            activity = _local_change_fraction(prev_gray, gray, win_size,
                                              local_ssim_cutoff)
            if activity > 0.01:
                is_animating = True
                static = 0
            elif is_animating:
                static += 1
                if static >= settle_frames:
                    anchor_activity = _local_change_fraction(
                        anchor_gray, gray, win_size, local_ssim_cutoff)
                    if anchor_activity > fallback_anchor_threshold:
                        trigger = True
                    else:
                        is_animating = False
                        static = 0

        if trigger:
            change = _local_change_fraction(last_saved_gray, gray, win_size,
                                            local_ssim_cutoff)
            if change >= dedupe_threshold:
                kept.append(frame)
                indices.append(i)
                last_saved_gray = gray
            # Trackers advance even when the save was skipped, or the same
            # trigger fires on every subsequent frame.
            last_saved_coverage = coverage
            anchor_gray = gray
            is_animating = False
            static = 0

        prev_gray = gray

    return kept, indices, total


def _extract_ssim_settle(video_path: Path, *,
                         activity_threshold: float | None = None,
                         settle_frames: int = 3, win_size: int = 7,
                         local_ssim_cutoff: float = 0.85,
                         activities: list[float] | None = None):
    """Hopping box / alpha masking / colour pop: save when motion settles.

    `activity_threshold=None` (the default) adapts it to this video -- see
    `adaptive_activity_threshold` for why a fixed 0.01 misclassifies a small
    highlight on a large canvas as no motion at all. Pass a float to pin it.
    """
    if activities is None:
        activities = _activity_series(video_path, win_size, local_ssim_cutoff)
    threshold = (activity_threshold if activity_threshold is not None
                 else adaptive_activity_threshold(activities))

    kept, indices = [], []
    is_animating = False
    static = 0
    total = 0

    for i, frame in enumerate(_decode(video_path)):
        total += 1
        if i == 0:
            kept.append(frame)
            indices.append(i)
            continue
        activity = activities[i - 1] if i - 1 < len(activities) else 0.0
        if activity > threshold:
            is_animating = True
            static = 0
        elif is_animating:
            static += 1
            if static >= settle_frames:
                kept.append(frame)
                indices.append(i)
                is_animating = False
                static = 0

    return kept, indices, total


_DISPATCH = {
    "motion_centroid": _extract_motion_centroid,
    "coverage_ssim": _extract_coverage_ssim,
    "ssim_settle": _extract_ssim_settle,
}

# Below this share of near-identical consecutive frames, a video has no dwell
# time and there is nothing for a settle detector to detect.
MIN_DWELL_FRACTION = 0.15


def _dwell_fraction(activities: list[float],
                    threshold: float | None = None) -> float:
    """Share of consecutive frame pairs that barely change, and the frame count.

    THIS GUARD EXISTS BECAUSE OF A MEASURED SURPRISE. All three detectors were
    written for real-time video, where an animation moves for many frames and
    then holds -- the hold is what they key on. AnimateBench's reference
    animations are not that: they are renders at **2 fps** (32 of 64 videos;
    the rest are 4 or 6, and exactly one is 15). At that rate essentially every
    frame is already a distinct authored state and nothing ever holds, so the
    settle counter never fires and a 57-frame sliding-box video extracts to a
    single keyframe -- measured, not hypothetical.

    Keying on the dwell fraction rather than on fps is deliberate: fps is
    metadata that can lie or be absent, while dwell is the property the
    detectors actually depend on. A genuine screen recording will show a high
    dwell fraction at any frame rate and take the detector path.

    The threshold must be the ADAPTIVE one, for the same reason the detectors
    use it. Measured against the fixed 0.01, a hopping-box video whose every
    frame differs reported a dwell fraction of 1.00 -- "nothing ever moves" --
    which is the opposite of the truth and would have sent it down the detector
    path to extract a single frame.
    """
    if len(activities) < 2:
        return 1.0
    if threshold is None:
        threshold = adaptive_activity_threshold(activities)
    if threshold == float("inf"):
        return 1.0                   # genuinely static
    return sum(1 for a in activities if a <= threshold) / len(activities)


# -- public ----------------------------------------------------------------

def extract(video_path: Path, out_dir: Path, style: str,
            force: bool = False, **params) -> Keyframes:
    """Extract a keyframe deck from `video_path` using `style`'s detector.

    Cached on the video's content hash: re-running is free, but a re-exported
    video invalidates the deck. Existence alone is not freshness -- a stale deck
    beside a rebuilt video has cost this project real time before.
    """
    import cv2

    video_path, out_dir = Path(video_path), Path(out_dir)
    if not video_path.exists():
        raise KeyframeError(f"no video at {video_path}")
    if style not in EXTRACTORS:
        raise KeyframeError(
            f"no keyframe extractor for style '{style}'. Known: "
            f"{sorted(EXTRACTORS)}")

    method = EXTRACTORS[style]
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    sidecar = out_dir / "keyframes.json"

    if sidecar.exists() and not force:
        try:
            prior = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {}
        if prior.get("video_sha256") == digest and prior.get("method") == method:
            paths = list_frames(out_dir)
            if len(paths) == prior.get("count"):
                return Keyframes(
                    paths=paths, source_indices=prior.get("source_indices", []),
                    source_frame_count=prior.get("source_frame_count", 0),
                    method=method, style=style, video=video_path,
                    video_sha256=digest, params=prior.get("params", {}))

    activities = _activity_series(video_path)
    threshold = adaptive_activity_threshold(activities)
    dwell = _dwell_fraction(activities, threshold)

    if dwell < MIN_DWELL_FRACTION:
        # No hold to detect. Every frame is already a distinct authored state,
        # so the honest deck is every frame -- running a settle detector here
        # would silently discard all but the first.
        method = "passthrough_no_dwell"
        frames = list(_decode(video_path))
        indices = list(range(len(frames)))
        total = len(frames)
    else:
        kwargs = dict(params)
        if method == "ssim_settle":
            kwargs["activities"] = activities
        frames, indices, total = _DISPATCH[method](video_path, **kwargs)

        # Guard on the OUTCOME, not just on the input. A detector can clear the
        # dwell check and still return nothing usable -- a video with a dwell
        # fraction just over the cutoff has too few still frames in a row to
        # ever satisfy `settle_frames`, and returns the opening frame alone.
        # A 12-frame animation reduced to one keyframe is a detector failure,
        # not a finding about the animation, and passing it downstream would
        # silently hand every per-frame judge a single frame to score.
        if len(frames) < 2 and total > 2:
            method = f"passthrough_after_{method}_found_{len(frames)}"
            frames = list(_decode(video_path))
            indices = list(range(len(frames)))
            total = len(frames)
    params = {**params, "dwell_fraction": round(dwell, 4),
              "activity_threshold_used": (None if threshold == float("inf")
                                          else round(threshold, 6)),
              "min_dwell_fraction": MIN_DWELL_FRACTION}
    if not frames:
        raise KeyframeError(f"{video_path}: decoded {total} frame(s), kept none")

    # Rewrite the directory rather than merging into a previous deck, or a
    # shorter new extraction leaves the tail of the old one behind and the two
    # are silently interleaved.
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame-*.png"):
        stale.unlink()

    width = max(2, len(str(len(frames))))
    paths = []
    for n, frame in enumerate(frames, 1):
        path = out_dir / f"frame-{n:0{width}d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    result = Keyframes(paths=paths, source_indices=indices,
                       source_frame_count=total, method=method, style=style,
                       video=video_path, video_sha256=digest, params=params)
    sidecar.write_text(json.dumps({
        "method": method, "style": style, "count": len(paths),
        "video": str(video_path), "video_sha256": digest,
        "source_frame_count": total, "source_indices": indices,
        "params": params,
    }, indent=2), encoding="utf-8")
    return result
