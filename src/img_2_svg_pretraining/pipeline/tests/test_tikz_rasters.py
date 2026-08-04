"""Splice invariants for the raster integrator.

The splice rewrites generated LaTeX in place, so the failure modes are quiet:
a corrupted node still looks plausible in a diff but breaks compilation, and a
dropped `xml id` silently breaks every downstream `focus` reference. These
tests pin the invariants that matter.

Run: pytest src/img_2_svg_pretraining/pipeline/tests/ -q
"""
from __future__ import annotations

from img_2_svg_pretraining.pipeline.transmuter.tikz_rasters import (
    find_placeholders, splice,
)

HEADER = r"""\documentclass[tikz, border=10pt]{standalone}
\usetikzlibrary{fit, positioning}
\begin{document}
\begin{tikzpicture}[x=1pt, y=-1pt,
    xml id/.code={}, xml class/.code={}, xml parent/.code={},
    raster_node/.style={draw, dashed, fill=gray!10}]
"""
FOOTER = "\\end{tikzpicture}\n\\end{document}\n"


def _doc(*nodes: str) -> str:
    return HEADER + "\n".join(nodes) + "\n" + FOOTER


def test_finds_only_raster_nodes():
    code = _doc(
        r"\node[raster_node, xml id=img_a, xml class=raster_node] (img_a) at (10, 20) {Photo};",
        r"\node[block, xml id=blk_a, xml class=block] (blk_a) at (30, 40) {Encoder};",
    )
    found = find_placeholders(code)
    assert [p.xml_id for p in found] == ["img_a"]


def test_parses_dimensions_and_position():
    code = _doc(
        r"\node[raster_node, minimum width=110pt, minimum height=85pt, "
        r"xml id=img_a, xml class=raster_node] (img_a) at (790, 65) {Photo};"
    )
    p = find_placeholders(code)[0]
    assert (p.width, p.height) == (110.0, 85.0)
    assert (p.x, p.y) == (790.0, 65.0)


def test_converts_cm_to_points():
    code = _doc(
        r"\node[raster_node, minimum width=2cm, xml id=img_a, xml class=raster_node] "
        r"(img_a) at (0, 0) {P};"
    )
    assert round(find_placeholders(code)[0].width, 1) == 56.9


def test_splice_preserves_identity_and_geometry():
    """The options carry everything downstream depends on -- only the body
    may change."""
    code = _doc(
        r"\node[raster_node, minimum width=110pt, minimum height=85pt, "
        r"xml id=img_a, xml class=raster_node, xml parent=root] (img_a) at (790, 65) {Photo};"
    )
    new, replaced = splice(code, {"img_a": "/abs/crop.png"})

    assert replaced == ["img_a"]
    for token in ("xml id=img_a", "xml class=raster_node", "xml parent=root",
                  "(img_a)", "at (790, 65)", "minimum width=110pt"):
        assert token in new, token
    assert "width=110.0pt" in new and "height=85.0pt" in new
    assert "keepaspectratio" in new  # fit inside the box, never distort


def test_splice_handles_nested_braces_in_body():
    """A math body has inner braces; locating the body by searching for a
    brace would corrupt the node. Regression test for exactly that."""
    code = _doc(
        r"\node[raster_node, xml id=img_a, xml class=raster_node] (img_a) at (0, 0) "
        r"{$\Phi(\mathbf{x},\delta)$ Graphic};"
    )
    new, replaced = splice(code, {"img_a": "/abs/crop.png"})

    assert replaced == ["img_a"]
    assert new.count("{") == new.count("}")
    assert "\\mathbf" not in new          # body fully replaced
    assert new.count("includegraphics") == 1


def test_splice_multiple_keeps_all_nodes_intact():
    code = _doc(
        r"\node[raster_node, xml id=img_a, xml class=raster_node] (img_a) at (0, 0) {A};",
        r"\node[block, xml id=blk, xml class=block] (blk) at (5, 5) {Keep me};",
        r"\node[raster_node, xml id=img_b, xml class=raster_node] (img_b) at (10, 0) {B};",
    )
    new, replaced = splice(code, {"img_a": "/a.png", "img_b": "/b.png"})

    assert set(replaced) == {"img_a", "img_b"}
    assert "{Keep me}" in new              # untouched node survives verbatim
    assert new.count("includegraphics") == 2
    assert new.count("{") == new.count("}")


def test_splice_adds_graphicx_once():
    code = _doc(
        r"\node[raster_node, xml id=img_a, xml class=raster_node] (img_a) at (0, 0) {A};"
    )
    new, _ = splice(code, {"img_a": "/a.png"})
    assert new.count(r"\usepackage{graphicx}") == 1

    # Already present: must not be added a second time.
    again, _ = splice(new, {"img_a": "/b.png"})
    assert again.count(r"\usepackage{graphicx}") == 1


def test_splice_ignores_unknown_ids():
    code = _doc(
        r"\node[raster_node, xml id=img_a, xml class=raster_node] (img_a) at (0, 0) {A};"
    )
    new, replaced = splice(code, {"nonexistent": "/a.png"})
    assert replaced == []
    assert new == code                     # nothing touched, not even graphicx
