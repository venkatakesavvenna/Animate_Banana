"""Box handling for the raster integrator: grid decoding and sub-panel expansion."""
from __future__ import annotations

from img_2_svg_pretraining.pipeline.transmuter.raster_integrator import _expand_to_placeholder
from img_2_svg_pretraining.pipeline.vision.gemini_boxes import to_pixels

# The filmstrip figure this stage was developed against.
SIZE = (978, 169)


# -- Gemini grid decoding ------------------------------------------------

def test_decodes_ymin_xmin_ymax_xmax_order():
    """The grid convention is [ymin, xmin, ymax, xmax], not [x0, y0, x1, y1].

    Getting this wrong transposes every crop, which still produces a
    plausible-looking box, so it needs pinning explicitly.
    """
    assert to_pixels([0, 0, 1000, 1000], (100, 200)) == (0.0, 0.0, 100.0, 200.0)
    # Left half of the image: x spans 0-500, y spans the full height.
    assert to_pixels([0, 0, 1000, 500], (100, 200)) == (0.0, 0.0, 50.0, 200.0)
    # Top half: y spans 0-500.
    assert to_pixels([0, 0, 500, 1000], (100, 200)) == (0.0, 0.0, 100.0, 100.0)


def test_scales_against_each_axis_independently():
    """Normalization is per-axis, so a non-square image must not be squashed."""
    assert to_pixels([250, 500, 750, 1000], SIZE) == (489.0, 42.25, 978.0, 126.75)


def test_clamps_out_of_range_coordinates():
    """A few units past the grid edge still locates the region correctly."""
    box = to_pixels([-20, -5, 1050, 1200], (100, 200))
    assert box == (0.0, 0.0, 100.0, 200.0)


def test_normalizes_swapped_corners():
    """A transposed pair is unambiguous, so recover it rather than discard."""
    assert to_pixels([800, 600, 200, 100], (100, 200)) == (10.0, 40.0, 60.0, 160.0)


def test_rejects_malformed_and_degenerate_boxes():
    for bad in (None, [], [1, 2, 3], "1,2,3,4", [1, 2, 3, 4, 5], ["a", "b", "c", "d"]):
        assert to_pixels(bad, SIZE) is None
    # Zero-area and sub-pixel boxes have nothing worth cropping.
    assert to_pixels([500, 500, 500, 500], SIZE) is None
    assert to_pixels([500, 500, 501, 501], SIZE) is None


# -- sub-panel expansion -------------------------------------------------

def test_expands_a_sub_panel_box():
    """Regression: a composite graphic came back as its bottom row only.

    The box was 114x57 (ratio 2.00) against a placeholder of ratio 1.29.
    """
    grown = _expand_to_placeholder((723, 88, 837, 145), (735, 22, 845, 107), SIZE)
    assert grown[3] - grown[1] > 57 + 10       # meaningfully taller
    assert grown[2] - grown[0] == 837 - 723    # width unchanged
    assert grown[3] <= SIZE[1]                 # clamped to the image


def test_leaves_well_proportioned_boxes_alone():
    """Neighbours whose boxes already match must pass through untouched."""
    for box, placeholder in (
        ((531, 34, 694, 143), (570, 22, 700, 107)),
        ((868, 34, 975, 142), (890, 22, 990, 107)),
    ):
        assert _expand_to_placeholder(box, placeholder, SIZE) == box


def test_expansion_never_shrinks():
    box = (100, 100, 300, 110)                  # very wide and flat
    grown = _expand_to_placeholder(box, (0, 0, 50, 100), SIZE)
    assert grown[2] - grown[0] >= box[2] - box[0]
    assert grown[3] - grown[1] >= box[3] - box[1]


def test_expansion_clamps_to_image_bounds():
    grown = _expand_to_placeholder((0, 0, 200, 20), (0, 0, 20, 200), SIZE)
    assert grown[0] >= 0 and grown[1] >= 0
    assert grown[2] <= SIZE[0] and grown[3] <= SIZE[1]


def test_expansion_tolerates_missing_boxes():
    assert _expand_to_placeholder(None, (0, 0, 1, 1), SIZE) is None
    assert _expand_to_placeholder((0, 0, 1, 1), None, SIZE) == (0, 0, 1, 1)


# -- animation-critic bloat guard ----------------------------------------

def test_bloat_guard_flags_invented_geometry():
    """Regression: asked to review already-compiling code, the critic
    invented geometry to fill deliberately-empty placeholder boxes, tripling
    the draw commands and rendering four pipeline stages as single dots."""
    from img_2_svg_pretraining.pipeline.animator.critic import _is_bloated

    before = "\\node{a};" * 23
    after = "\\node{a};" * 71
    assert _is_bloated(before, after)


def test_bloat_guard_allows_unchanged_and_small_edits():
    from img_2_svg_pretraining.pipeline.animator.critic import _is_bloated

    before = "\\node{a};" * 23
    assert not _is_bloated(before, before)
    assert not _is_bloated(before, before + "\\node{b};" * 3)   # small fix
    assert not _is_bloated(before, "\\node{a};" * 10)           # shrank


def test_bloat_guard_tolerates_growth_from_a_tiny_base():
    """A near-empty document legitimately grows a lot; the absolute floor
    keeps that from tripping the ratio."""
    from img_2_svg_pretraining.pipeline.animator.critic import _is_bloated

    assert not _is_bloated("\\node{a};", "\\node{a};" * 5)


# -- Cross-space aspect comparison ---------------------------------------
#
# `_expand_to_placeholder` weighs a detected box (source-image PIXELS) against a
# placeholder box (document units: SVG user units, TikZ pt). Those are only the
# same space when the document's aspect matches the image's. WACV_2023_set5_000024
# is the real counter-example that motivated `doc_extent`: a 995x441 figure
# (2.256) drawn into viewBox="0 0 1000 350" (2.857).

ANISO_IMG = (995, 441)
ANISO_DOC = (1000.0, 350.0)
# A placeholder 2.00 wide-to-tall in user units covers 2.00 * 0.79 in pixels.
ANISO_FACTOR = (ANISO_IMG[0] / ANISO_IMG[1]) / (ANISO_DOC[0] / ANISO_DOC[1])
PLACEHOLDER = (0.0, 0.0, 100.0, 50.0)


def _region(aspect: float, height: float = 50.0):
    return (0.0, 0.0, aspect * height, height)


def _expanded(region, doc_extent):
    out = _expand_to_placeholder(region, PLACEHOLDER, ANISO_IMG, doc_extent)
    return tuple(round(v, 2) for v in out) != tuple(round(v, 2) for v in region)


def test_svg_viewbox_extent_is_parsed():
    from img_2_svg_pretraining.pipeline.transmuter.rasters import document_extent

    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 350"></svg>'
    assert document_extent(svg, "svg") == (1000.0, 350.0)
    # width/height only, no viewBox.
    assert document_extent('<svg width="640px" height="480px"></svg>', "svg") == (640.0, 480.0)
    # Nothing parseable must be None, never a guess -- callers read None as
    # "do not correct".
    assert document_extent("<svg></svg>", "svg") is None


def test_tikz_extent_is_unknown_by_design():
    """TikZ's extent is not knowable without compiling, and the rendered figure
    is a crop of the picture anyway, so the correction collapses to 1."""
    from img_2_svg_pretraining.pipeline.transmuter.rasters import document_extent

    assert document_extent("\\begin{tikzpicture}\\end{tikzpicture}", "tikz") is None


def test_correctly_detected_box_is_left_alone_under_anisotropy():
    """The box already covers exactly what the placeholder declares."""
    assert not _expanded(_region(2.0 * ANISO_FACTOR), ANISO_DOC)


def test_extent_stops_spurious_expansion_of_a_near_correct_box():
    """A detection 10-20% off is ordinary crop-and-margin noise.

    Without the extent the placeholder's 2.00 is compared against a pixel
    aspect of 1.58, so the ratio starts at 0.79 rather than 1.0 and a merely
    slightly-narrow box is pushed past the 0.75 floor and needlessly grown.
    """
    for err in (0.90, 0.80):
        region = _region(2.0 * ANISO_FACTOR * err)
        assert _expanded(region, None), f"{err}: precondition -- old code expands"
        assert not _expanded(region, ANISO_DOC), f"{err}: fixed code must not"


def test_extent_still_catches_a_genuinely_truncated_box():
    """The band must not simply be widened; real sub-panel misses still fire."""
    assert _expanded(_region(2.0 * ANISO_FACTOR * 0.70), ANISO_DOC)


def test_extent_catches_an_over_wide_box_the_old_code_accepted():
    """The bias cuts both ways: the same 0.79 offset silently accepted boxes
    up to ~1.7x too wide, which is squarely in sub-panel territory."""
    for err in (1.40, 1.70):
        region = _region(2.0 * ANISO_FACTOR * err)
        assert not _expanded(region, None), f"{err}: precondition -- old code accepts"
        assert _expanded(region, ANISO_DOC), f"{err}: fixed code must catch it"


def test_matching_aspects_make_the_correction_a_no_op():
    """When document and image agree, passing an extent changes nothing --
    which is what makes it safe for TikZ and for square-on SVGs."""
    img = (800, 400)
    for region_aspect in (0.5, 1.0, 2.0, 4.0):
        region = _region(region_aspect)
        a = _expand_to_placeholder(region, PLACEHOLDER, img, None)
        b = _expand_to_placeholder(region, PLACEHOLDER, img, (400.0, 200.0))
        assert a == b
