"""Classical-CV node segmentation: point-seeded flood-fill bounded by dark
box borders, cross-checked by a rectangular-contour pass.

Explored 2026-07-08 as a no-ML, no-GPU alternative for clean, vector-rendered
architecture diagrams where every node is a closed dark-stroked or
gray-filled axis-aligned rectangle. Worked well on some `partial_test_set`
samples, but a batch run against real diagrams found this pipeline's
single-global-Otsu-threshold binarization misses low-contrast/light-colored
box borders (e.g. a light-blue-filled box with a border pixel value close to
its own fill) -- the leak guard correctly rejects the resulting flood, so it
fails safe as `needs_review` rather than a wrong answer, but this made a
meaningful fraction of real nodes unresolved (~27-79% resolved depending on
diagram style, worse on the more visually varied samples). `sam2_amg.py`
(SAM2 automatic mask generation) was tried next and found correct box masks
on every style tested including the ones that broke this pipeline, with no
per-image threshold tuning -- see that module's docstring. Kept here for
reference/comparison and because it's still the fastest, fully-deterministic
option on the subset of diagrams it does handle correctly (flat-fill,
solid-dark-stroke boxes).

Pipeline (5 stages, matching the spec this was built against):
    0. load_normalize   -- decode, composite alpha over white, grayscale,
                            confirm background polarity from the corners.
    1. binarize         -- inverse-Otsu threshold + 1px dilation turns dark
                            strokes/text/borders into "walls", everything
                            else (background + box interiors) into "floor".
    2. node_region       -- per point, multi-seed flood-fill (a small grid
                            around the point, not just the point itself,
                            since a seed landing on a glyph/letter-counter
                            floods a tiny garbage region instead of the
                            node) and keep the largest valid region.
    3. detect_box_contours -- independent rectangular-contour detection,
                            used to validate Stage 2 boxes and as a fallback
                            for points where flood-fill found nothing usable.
    4. reconcile         -- one box per input id: Stage 2 if valid, else the
                            smallest Stage 3 contour containing the point,
                            else `needs_review`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --- verified defaults (see module docstring / README for provenance) ---
WALL_DILATE_KSIZE = 3
WALL_DILATE_ITERS = 1
MIN_AREA = 800          # px; below this = letter counter / junk
SEED_WIN = 12            # px search half-window around each point
SEED_STEP = 3            # grid step within the search window
MAX_FRAC = 0.25          # region larger than this fraction of image = leak
RECT_EXTENT_MIN = 0.85   # contourArea / boundingRect area, rectangularity
APPROX_EPS_FRAC = 0.02   # fraction of perimeter for approxPolyDP epsilon
CONTOUR_DEDUP_IOU = 0.9  # inner/outer contour pairs from one stroke


@dataclass
class Point:
    id: str
    x: int
    y: int


@dataclass
class NodeResult:
    id: str
    box: tuple[int, int, int, int] | None  # x, y, w, h, or None if unresolved
    mask: np.ndarray | None                # HxW uint8, 255 = node
    source: str                            # "flood" | "contour" | "needs_review"
    iou_vs_contour: float | None = None    # Stage-2/3 cross-check, if both exist


@dataclass
class SegmentationResult:
    nodes: list[NodeResult] = field(default_factory=list)
    otsu_threshold: float = 0.0
    image_shape: tuple[int, int] = (0, 0)  # H, W

    def needs_review(self) -> list[NodeResult]:
        return [n for n in self.nodes if n.source == "needs_review"]


# ---------------------------------------------------------------------------
# Stage 0: load & normalize
# ---------------------------------------------------------------------------

def load_normalize(path: str | Path) -> np.ndarray:
    """Loads an image and returns a grayscale array with a light (~255)
    background -- composites alpha over white if present, and inverts if a
    corner-polarity check finds a dark background (guards against the
    rendered-figure assumption breaking on an unusual input rather than
    silently producing garbage)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr_channels = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
        bgr = (bgr_channels * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    else:
        bgr = img[:, :, :3]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    if float(np.median(corners)) < 128:
        gray = 255 - gray  # background reads dark -- invert so it's light
    return gray


# ---------------------------------------------------------------------------
# Stage 1: binarize to walls
# ---------------------------------------------------------------------------

def binarize(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Inverse-Otsu: dark strokes/text/borders -> 255, interiors +
    background -> 0 (`bw`). Returns both `bw` (used as-is for Stage 3
    contour detection, where dilation would distort box geometry) and a
    1px-dilated `walls` (used for Stage 2 flood-fill, where the dilation
    seals hairline anti-aliased gaps in borders that would otherwise let
    fill leak out of a node into its neighbors) -- keep dilation at 1
    iteration, more starts eating into small box interiors."""
    t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((WALL_DILATE_KSIZE, WALL_DILATE_KSIZE), np.uint8)
    walls = cv2.dilate(bw, kernel, iterations=WALL_DILATE_ITERS)
    return bw, walls, float(t)


# ---------------------------------------------------------------------------
# Stage 2: per-point flood fill
# ---------------------------------------------------------------------------

def _flood_region(seed_x: int, seed_y: int, walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(
        walls.copy(), mask, (seed_x, seed_y), 255,
        flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    return mask[1:-1, 1:-1] == 255


def node_region(
    px: int, py: int, walls: np.ndarray,
    min_area: int = MIN_AREA, seed_win: int = SEED_WIN, seed_step: int = SEED_STEP,
) -> tuple[int, np.ndarray] | None:
    """Multi-seed flood fill around (px, py): tries a grid of nearby seeds
    (not just the exact point) and keeps the largest resulting region above
    `min_area`. This is the fix for seeds landing on a glyph/letter-counter
    -- a seed inside the hole of an "o"/"e"/"D" floods a tiny (~tens of px)
    garbage region, while the real node interior is thousands of px, so
    taking the largest valid region across a small local search reliably
    recovers the node even when the exact point is unlucky."""
    h, w = walls.shape
    best: tuple[int, np.ndarray] | None = None
    for dy in range(-seed_win, seed_win + 1, seed_step):
        for dx in range(-seed_win, seed_win + 1, seed_step):
            x, y = px + dx, py + dy
            if not (0 <= x < w and 0 <= y < h):
                continue
            if walls[y, x] != 0:
                continue  # seed landed on a wall pixel itself -- skip
            region = _flood_region(x, y, walls)
            area = int(region.sum())
            if area >= min_area and (best is None or area > best[0]):
                best = (area, region)
    return best


def _region_touches_edge(region: np.ndarray) -> bool:
    return bool(
        region[0, :].any() or region[-1, :].any()
        or region[:, 0].any() or region[:, -1].any()
    )


def _region_to_box_and_mask(region: np.ndarray, shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Nodes are axis-aligned rectangles, so the deliverable is the filled
    boundingRect of the flooded region (this also fills in any interior
    holes punched by text glyphs)."""
    ys, xs = np.where(region)
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    mask = np.zeros(shape, np.uint8)
    mask[y0:y0 + h, x0:x0 + w] = 255
    return (x0, y0, w, h), mask


# ---------------------------------------------------------------------------
# Stage 3: contour cross-check
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_box_contours(
    bw: np.ndarray, min_area: int = MIN_AREA, rect_extent_min: float = RECT_EXTENT_MIN,
    approx_eps_frac: float = APPROX_EPS_FRAC, dedup_iou: float = CONTOUR_DEDUP_IOU,
) -> list[tuple[int, int, int, int]]:
    """Independent rectangle detection from the binarized strokes: finds
    every roughly-rectangular, roughly-convex contour above `min_area`.
    Each drawn border stroke produces both an inner and outer contour --
    dedup by IoU, keeping the larger box of any pair that overlaps a lot."""
    contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        approx = cv2.approxPolyDP(c, approx_eps_frac * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area <= 0:
            continue
        extent = cv2.contourArea(c) / (area + 1e-6)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > min_area and extent > rect_extent_min:
            boxes.append((x, y, w, h))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if not any(_iou(box, kept) > dedup_iou for kept in deduped):
            deduped.append(box)
    return deduped


def _point_in_box(px: int, py: int, box: tuple[int, int, int, int]) -> bool:
    x, y, w, h = box
    return x <= px < x + w and y <= py < y + h


def _smallest_containing_box(
    px: int, py: int, boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """When a point falls inside several nested boxes (e.g. a node inside a
    container panel), the node -- the smallest containing box -- is the
    correct match, not the outer container."""
    containing = [b for b in boxes if _point_in_box(px, py, b)]
    if not containing:
        return None
    return min(containing, key=lambda b: b[2] * b[3])


# ---------------------------------------------------------------------------
# Stage 4: reconcile
# ---------------------------------------------------------------------------

def reconcile(
    points: list[Point], walls: np.ndarray, bw: np.ndarray, shape: tuple[int, int],
    min_area: int = MIN_AREA, seed_win: int = SEED_WIN, seed_step: int = SEED_STEP,
    max_frac: float = MAX_FRAC, rect_extent_min: float = RECT_EXTENT_MIN,
    approx_eps_frac: float = APPROX_EPS_FRAC,
) -> SegmentationResult:
    h, w = shape
    max_area = max_frac * h * w
    contour_boxes = detect_box_contours(bw, min_area=min_area, rect_extent_min=rect_extent_min,
                                         approx_eps_frac=approx_eps_frac)

    result = SegmentationResult(image_shape=shape)
    seen_boxes: dict[tuple[int, int, int, int], str] = {}

    for point in points:
        flood = node_region(point.x, point.y, walls, min_area=min_area,
                             seed_win=seed_win, seed_step=seed_step)

        box: tuple[int, int, int, int] | None = None
        mask: np.ndarray | None = None
        source = "needs_review"
        iou_vs_contour: float | None = None

        if flood is not None:
            area, region = flood
            is_leak = area > max_area or _region_touches_edge(region)
            if not is_leak:
                box, mask = _region_to_box_and_mask(region, shape)
                source = "flood"
                match = _smallest_containing_box(point.x, point.y, contour_boxes)
                if match is not None:
                    iou_vs_contour = _iou(box, match)

        if box is None:
            match = _smallest_containing_box(point.x, point.y, contour_boxes)
            if match is not None:
                x, y, w_, h_ = match
                mask = np.zeros(shape, np.uint8)
                mask[y:y + h_, x:x + w_] = 255
                box, source = match, "contour"

        if box is not None and box in seen_boxes:
            # Two points resolved to the same box -- a data/annotation
            # error, not something to silently merge.
            print(f"WARNING: point '{point.id}' resolved to the same box as "
                  f"'{seen_boxes[box]}' -- check the input points for duplicates.")
        elif box is not None:
            seen_boxes[box] = point.id

        result.nodes.append(NodeResult(
            id=point.id, box=box, mask=mask, source=source, iou_vs_contour=iou_vs_contour,
        ))

    return result


# ---------------------------------------------------------------------------
# rendering / IO helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
]


def render_overlay(image_bgr: np.ndarray, result: SegmentationResult) -> np.ndarray:
    overlay = image_bgr.copy()
    for i, node in enumerate(result.nodes):
        if node.box is None:
            continue
        x, y, w, h = node.box
        color = (0, 0, 255) if node.source == "needs_review" else _PALETTE[i % len(_PALETTE)]
        if node.source != "needs_review":
            tint = overlay[y:y + h, x:x + w].astype(np.float32)
            tint = tint * 0.6 + np.array(color, dtype=np.float32) * 0.4
            overlay[y:y + h, x:x + w] = tint.astype(np.uint8)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, node.id, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return overlay


def load_points(path: str | Path) -> list[Point]:
    data = json.loads(Path(path).read_text())
    return [Point(id=p["id"], x=int(p["x"]), y=int(p["y"])) for p in data]


def write_outputs(result: SegmentationResult, image_bgr: np.ndarray, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    boxes = {}
    for node in result.nodes:
        if node.mask is not None:
            cv2.imwrite(str(masks_dir / f"{node.id}.png"), node.mask)
        if node.box is not None:
            boxes[node.id] = list(node.box)
    (out_dir / "boxes.json").write_text(json.dumps(boxes, indent=2))

    overlay = render_overlay(image_bgr, result)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
