"""Data engine CLI: image -> structured XML -> TikZ/SVG, filtered by a VLM judge.

Run inside the project docker container (needs transformers/torch, the
LaTeX/poppler toolchain for the tikz backend, and cairosvg for the svg
backend).

Design: stage-batch, not per-sample pipeline. Each stage runs over the WHOLE
sample set before the next stage starts, handing off through on-disk
artifacts under `data_engine/cache/` -- not a single process holding every
model resident and looping per sample. This means:
  - Only the model(s) a given stage needs are ever loaded at once (pointing
    and SAM3 for `segment`, one chat VLM for `edges`/`codegen`/`judge`) --
    never five models resident in GPU memory simultaneously.
  - Each stage is independently re-runnable: e.g. re-run `edges` alone after
    tweaking the Set-of-Mark prompt, without recomputing points/masks.
  - A stage's output is always inspectable on disk before moving on, which
    is what the "thin end-to-end first" milestone needs.

Stages (run in this order, each its own subcommand):
    point     image -> cache/points/<id>.json (+ overlay PNG for inspection)
    segment   cache/points/*.json -> cache/masks/<id>.json + mask PNGs
    assemble  cache/masks/*.json -> cache/diagrams/pre_edges/<id>.xml (no model)
    edges     cache/diagrams/pre_edges/*.xml -> cache/diagrams/final/<id>.xml
    codegen   cache/diagrams/final/*.xml -> cache/code/<backend>/<id>.{tex,svg}
    render    cache/code/<backend>/*.{tex,svg} -> cache/renders/<backend>/*.png (no model)
    judge     cache/renders/<backend>/*.png -> output/<backend>/summary.jsonl

Example, one full pass over 8 samples:
    python -m img_2_svg_pretraining.data_engine.run_data_engine point   --limit 8
    python -m img_2_svg_pretraining.data_engine.run_data_engine segment --limit 8
    python -m img_2_svg_pretraining.data_engine.run_data_engine assemble --limit 8
    python -m img_2_svg_pretraining.data_engine.run_data_engine edges   --limit 8
    python -m img_2_svg_pretraining.data_engine.run_data_engine codegen --limit 8 --backend tikz
    python -m img_2_svg_pretraining.data_engine.run_data_engine render  --limit 8 --backend tikz
    python -m img_2_svg_pretraining.data_engine.run_data_engine judge   --limit 8 --backend tikz

First milestone (per plan): run this over a handful of samples, inspect every
intermediate artifact under cache/, before scaling --limit up.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--data-root", type=str, default="/code/data/partial_test_set")
        p.add_argument("--limit", type=int, default=8, help="0 = all samples")
        p.add_argument("--gpu", type=str, default="1", help="single physical GPU id to run on")

    p_point = sub.add_parser("point", help="node/raster point discovery")
    common(p_point)
    p_point.add_argument("--pointing-model", default="molmo2-8b",
                          help="registry key from models.py: molmo-point-8b or molmo2-8b")
    p_point.add_argument("--node-query", type=str, default=None)
    p_point.add_argument("--raster-query", type=str, default=None)

    p_segment = sub.add_parser("segment", help="SAM3 segmentation of cached points")
    common(p_segment)
    p_segment.add_argument("--pointing-model", default="molmo2-8b",
                            help="which cache/points/<id>.json to read (must match a prior `point` run)")
    p_segment.add_argument("--segmentation-model", default="sam3")

    p_assemble = sub.add_parser("assemble", help="cached masks -> pre-edges Diagram XML (no model)")
    common(p_assemble)
    p_assemble.add_argument("--pointing-model", default="molmo2-8b")
    p_assemble.add_argument("--segmentation-model", default="sam3")

    p_edges = sub.add_parser("edges", help="Set-of-Mark edge discovery over assembled diagrams")
    common(p_edges)
    p_edges.add_argument("--edge-model", default="qwen-3.5-9b", help="benchmark.models registry key")

    p_codegen = sub.add_parser("codegen", help="(image, final diagram XML) -> TikZ/SVG code")
    common(p_codegen)
    p_codegen.add_argument("--backend", choices=["tikz", "svg"], default="tikz")
    p_codegen.add_argument("--codegen-model", default="qwen-3.5-9b", help="benchmark.models registry key")

    p_render = sub.add_parser("render", help="compile/rasterize cached generated code (no model)")
    common(p_render)
    p_render.add_argument("--backend", choices=["tikz", "svg"], default="tikz")

    p_judge = sub.add_parser("judge", help="VLM-judge cached renders against source images")
    common(p_judge)
    p_judge.add_argument("--backend", choices=["tikz", "svg"], default="tikz")
    p_judge.add_argument("--judge-model", default="qwen-3.5-9b", help="benchmark.models registry key")
    p_judge.add_argument("--keep-threshold", type=int, default=4)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    return Path(__file__).parent / "cache"


def _points_path(sample_id: str, pointing_model: str) -> Path:
    return _cache_dir() / "points" / pointing_model / f"{sample_id}.json"


def _masks_manifest_path(sample_id: str, pointing_model: str, segmentation_model: str) -> Path:
    return _cache_dir() / "masks" / f"{pointing_model}__{segmentation_model}" / f"{sample_id}.json"


def _pre_edges_xml_path(sample_id: str, pointing_model: str, segmentation_model: str) -> Path:
    return _cache_dir() / "diagrams" / "pre_edges" / f"{pointing_model}__{segmentation_model}" / f"{sample_id}.xml"


def _final_xml_path(sample_id: str, pointing_model: str, segmentation_model: str, edge_model: str) -> Path:
    return (_cache_dir() / "diagrams" / "final"
            / f"{pointing_model}__{segmentation_model}__{edge_model}" / f"{sample_id}.xml")


def _code_path(sample_id: str, backend: str, codegen_model: str) -> Path:
    ext = "tex" if backend == "tikz" else "svg"
    return _cache_dir() / "code" / backend / codegen_model / f"{sample_id}.{ext}"


def _render_cache_dir(backend: str) -> Path:
    return _cache_dir() / "renders" / backend


def _overlay_points(image, node_points, raster_points):
    from PIL import ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    def draw_group(points, color, prefix):
        for p in points:
            r = 8
            draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), fill=color, outline="black", width=2)
            draw.text((p.x + r + 2, p.y - r), f"{prefix}{p.object_id}", fill=color, font=font)

    draw_group(node_points, "lime", "n")
    draw_group(raster_points, "deepskyblue", "r")
    return marked


def _color_rgb(name: str) -> tuple[int, int, int]:
    return {"lime": (0, 255, 0), "deepskyblue": (0, 191, 255)}[name]


def _overlay_masks(image, mask_items):
    """mask_items: list of (object_id, mask ndarray, bbox, is_raster)."""
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for object_id, mask, bbox, is_raster in mask_items:
        color = "deepskyblue" if is_raster else "lime"
        mask_img = PILImage.fromarray((mask * 255).astype(np.uint8)).convert("L")
        edge = mask_img.filter(__import__("PIL.ImageFilter", fromlist=["FIND_EDGES"]).FIND_EDGES)
        edge_arr = np.array(edge) > 0
        ys, xs = np.where(edge_arr)
        for x, y in zip(xs, ys):
            marked.putpixel((int(x), int(y)), _color_rgb(color))
        draw.rectangle(bbox, outline=color, width=2)
        prefix = "r" if is_raster else "n"
        draw.text((bbox[0] + 2, bbox[1] + 2), f"{prefix}{object_id}", fill=color, font=font)
    return marked


# ---------------------------------------------------------------------------
# stage: point
# ---------------------------------------------------------------------------

def _run_point(args, samples):
    from img_2_svg_pretraining.data_engine.models import get_pointing_model
    from img_2_svg_pretraining.data_engine.pointing import NODE_QUERY, RASTER_QUERY, make_point_runner
    from PIL import Image

    node_query = args.node_query or NODE_QUERY
    raster_query = args.raster_query or RASTER_QUERY

    print(f"Loading pointing model ({args.pointing_model})...")
    point_runner = make_point_runner(get_pointing_model(args.pointing_model))
    print("Model loaded.")

    for i, sample in enumerate(samples, 1):
        t0 = time.time()
        try:
            node_points = point_runner.point(sample.image_path, node_query)
            raster_points = point_runner.point(sample.image_path, raster_query)

            out_path = _points_path(sample.id, args.pointing_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "nodes": [asdict(p) for p in node_points],
                "rasters": [asdict(p) for p in raster_points],
            }, indent=2))

            image = Image.open(sample.image_path)
            _overlay_points(image, node_points, raster_points).save(out_path.with_suffix(".png"))

            print(f"[{i}/{len(samples)}] {sample.id}: {len(node_points)} nodes, "
                  f"{len(raster_points)} rasters ({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001 -- one bad sample shouldn't kill the run
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at {_points_path('<id>', args.pointing_model).parent}")


# ---------------------------------------------------------------------------
# stage: segment
# ---------------------------------------------------------------------------

def _load_points(sample_id: str, pointing_model: str):
    from img_2_svg_pretraining.data_engine.pointing import PointResult

    path = _points_path(sample_id, pointing_model)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached points for {sample_id} at {path} -- run `point` first "
            f"with --pointing-model {pointing_model}."
        )
    data = json.loads(path.read_text())
    return (
        [PointResult(**p) for p in data["nodes"]],
        [PointResult(**p) for p in data["rasters"]],
    )


def _run_segment(args, samples):
    from img_2_svg_pretraining.data_engine.models import get_segmentation_model
    from img_2_svg_pretraining.data_engine.segmentation import Sam3Runner, save_mask
    from PIL import Image

    print(f"Loading segmentation model ({args.segmentation_model})...")
    sam_runner = Sam3Runner(get_segmentation_model(args.segmentation_model))
    print("Model loaded.")

    mask_cache_dir = _cache_dir() / "masks" / f"{args.pointing_model}__{args.segmentation_model}" / "png"

    for i, sample in enumerate(samples, 1):
        t0 = time.time()
        try:
            node_points, raster_points = _load_points(sample.id, args.pointing_model)

            mask_items = []
            mask_records = []
            for is_raster, points in ((False, node_points), (True, raster_points)):
                prefix = "raster" if is_raster else "node"
                for point in points:
                    node_id = f"{prefix}_{point.object_id}"
                    mask_result = sam_runner.segment_point(sample.image_path, point.x, point.y)
                    mask_path = save_mask(mask_result.mask, mask_cache_dir, sample.id, node_id)
                    mask_items.append((point.object_id, mask_result.mask, mask_result.bbox, is_raster))
                    mask_records.append({
                        "id": node_id, "text": node_id, "bbox": list(mask_result.bbox),
                        "score": mask_result.score, "mask_path": str(mask_path),
                        "is_raster": is_raster,
                    })

            manifest_path = _masks_manifest_path(sample.id, args.pointing_model, args.segmentation_model)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(mask_records, indent=2))

            image = Image.open(sample.image_path)
            _overlay_masks(image, mask_items).save(manifest_path.with_suffix(".png"))

            print(f"[{i}/{len(samples)}] {sample.id}: {len(node_points)} nodes, "
                  f"{len(raster_points)} rasters segmented ({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at {_masks_manifest_path('<id>', args.pointing_model, args.segmentation_model).parent}")


# ---------------------------------------------------------------------------
# stage: assemble (no model)
# ---------------------------------------------------------------------------

def _run_assemble(args, samples):
    from img_2_svg_pretraining.data_engine.assemble import DetectedItem, assemble_diagram
    from img_2_svg_pretraining.data_engine.schema import BBox, write_xml_file

    for i, sample in enumerate(samples, 1):
        try:
            manifest_path = _masks_manifest_path(sample.id, args.pointing_model, args.segmentation_model)
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"No cached masks for {sample.id} at {manifest_path} -- run `segment` first."
                )
            records = json.loads(manifest_path.read_text())
            items = [
                DetectedItem(
                    id=r["id"], text=r["text"], bbox=BBox(*r["bbox"]),
                    mask_path=r["mask_path"], conf=r["score"], is_raster=r["is_raster"],
                )
                for r in records
            ]
            diagram = assemble_diagram(items)

            out_path = _pre_edges_xml_path(sample.id, args.pointing_model, args.segmentation_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_xml_file(diagram, out_path)

            print(f"[{i}/{len(samples)}] {sample.id}: {len(diagram.blocks)} blocks, "
                  f"{len(diagram.all_nodes())} nodes, {len(diagram.all_rasters())} rasters")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at "
          f"{_pre_edges_xml_path('<id>', args.pointing_model, args.segmentation_model).parent}")


# ---------------------------------------------------------------------------
# stage: edges
# ---------------------------------------------------------------------------

def _run_edges(args, samples):
    from img_2_svg_pretraining.benchmark.models import get_model
    from img_2_svg_pretraining.data_engine.edges import EdgeRunner
    from img_2_svg_pretraining.data_engine.schema import parse_xml_file, write_xml_file
    from PIL import Image

    # pointing/segmentation model keys aren't relevant beyond locating the
    # right pre_edges XML on disk; discover them from whatever's cached.
    pre_edges_root = _cache_dir() / "diagrams" / "pre_edges"

    print(f"Loading edge-discovery model ({args.edge_model})...")
    edge_runner = EdgeRunner(get_model(args.edge_model))
    print("Model loaded.")

    for i, sample in enumerate(samples, 1):
        t0 = time.time()
        try:
            candidates = sorted(pre_edges_root.glob(f"*/{sample.id}.xml"))
            if not candidates:
                raise FileNotFoundError(
                    f"No cached pre-edges diagram for {sample.id} under {pre_edges_root} -- "
                    f"run `assemble` first."
                )
            pre_edges_path = candidates[0]
            diagram = parse_xml_file(pre_edges_path)

            image = Image.open(sample.image_path).convert("RGB")
            diagram = edge_runner.discover_edges(image, diagram)

            variant = pre_edges_path.parent.name  # "<pointing_model>__<segmentation_model>"
            out_path = _final_xml_path(*variant.split("__"), args.edge_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_xml_file(diagram, out_path)

            print(f"[{i}/{len(samples)}] {sample.id}: {len(diagram.all_arrows())} edges "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at {_cache_dir() / 'diagrams' / 'final'}")


# ---------------------------------------------------------------------------
# stage: codegen
# ---------------------------------------------------------------------------

def _run_codegen(args, samples):
    from img_2_svg_pretraining.benchmark.models import get_model
    from img_2_svg_pretraining.data_engine.codegen import CodegenRunner

    final_root = _cache_dir() / "diagrams" / "final"

    print(f"Loading codegen model ({args.codegen_model})...")
    codegen_runner = CodegenRunner(get_model(args.codegen_model))
    print("Model loaded.")

    for i, sample in enumerate(samples, 1):
        t0 = time.time()
        try:
            candidates = sorted(final_root.glob(f"*/{sample.id}.xml"))
            if not candidates:
                raise FileNotFoundError(
                    f"No cached final diagram for {sample.id} under {final_root} -- run `edges` first."
                )
            from img_2_svg_pretraining.data_engine.schema import parse_xml_file
            diagram = parse_xml_file(candidates[0])

            result = codegen_runner.generate(sample.image_path, diagram, args.backend)

            out_path = _code_path(sample.id, args.backend, args.codegen_model)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if result.code is not None:
                out_path.write_text(result.code)
            (out_path.parent / f"{sample.id}.raw.txt").write_text(result.raw_output)

            print(f"[{i}/{len(samples)}] {sample.id}: code_extracted={result.code is not None} "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at {_code_path('<id>', args.backend, args.codegen_model).parent}")


# ---------------------------------------------------------------------------
# stage: render (no model)
# ---------------------------------------------------------------------------

def _run_render(args, samples):
    from img_2_svg_pretraining.data_engine.render import render

    code_root = _cache_dir() / "code" / args.backend
    render_cache_dir = _render_cache_dir(args.backend)

    for i, sample in enumerate(samples, 1):
        try:
            ext = "tex" if args.backend == "tikz" else "svg"
            candidates = sorted(code_root.glob(f"*/{sample.id}.{ext}"))
            if not candidates:
                raise FileNotFoundError(
                    f"No cached {ext} for {sample.id} under {code_root} -- run `codegen` first."
                )
            code = candidates[0].read_text()
            result = render(code, args.backend, render_cache_dir)
            print(f"[{i}/{len(samples)}] {sample.id}: ok={result.ok} "
                  f"png={result.png_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done. Output at {render_cache_dir}")


# ---------------------------------------------------------------------------
# stage: judge
# ---------------------------------------------------------------------------

def _run_judge(args, samples):
    from img_2_svg_pretraining.benchmark.metrics.judge import JudgeRunner
    from img_2_svg_pretraining.benchmark.models import get_model
    from img_2_svg_pretraining.data_engine.judge import judge_and_filter
    from img_2_svg_pretraining.data_engine.render import render

    code_root = _cache_dir() / "code" / args.backend
    render_cache_dir = _render_cache_dir(args.backend)
    output_dir = Path(__file__).parent / "output" / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    print(f"Loading judge model ({args.judge_model})...")
    judge_spec = get_model(args.judge_model)
    judge_runner = JudgeRunner(judge_spec.hf_repo, judge_spec.trust_remote_code, judge_spec.attn_implementation)
    print("Model loaded.")

    n_kept = 0
    with summary_path.open("a") as f:
        for i, sample in enumerate(samples, 1):
            t0 = time.time()
            try:
                ext = "tex" if args.backend == "tikz" else "svg"
                candidates = sorted(code_root.glob(f"*/{sample.id}.{ext}"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No cached {ext} for {sample.id} under {code_root} -- run `codegen` first."
                    )
                code = candidates[0].read_text()
                render_result = render(code, args.backend, render_cache_dir)

                kept, judge_score, judge_rationale = False, None, ""
                if render_result.ok and render_result.png_path is not None:
                    kept, judge_result = judge_and_filter(
                        judge_runner, sample.image_path, render_result.png_path, args.keep_threshold,
                    )
                    judge_score, judge_rationale = judge_result.score, judge_result.rationale

                n_kept += int(kept)
                record = {
                    "sample_id": sample.id, "backend": args.backend,
                    "render_ok": render_result.ok,
                    "render_png_path": str(render_result.png_path) if render_result.png_path else None,
                    "judge_score": judge_score, "judge_rationale": judge_rationale, "kept": kept,
                }
                f.write(json.dumps(record) + "\n")

                print(f"[{i}/{len(samples)}] {sample.id}: render_ok={render_result.ok} "
                      f"judge_score={judge_score} kept={kept} ({time.time()-t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(samples)}] {sample.id}: ERROR {e}")

    print(f"Done: {n_kept}/{len(samples)} kept. Summary at {summary_path}")


STAGE_RUNNERS = {
    "point": _run_point,
    "segment": _run_segment,
    "assemble": _run_assemble,
    "edges": _run_edges,
    "codegen": _run_codegen,
    "render": _run_render,
    "judge": _run_judge,
}


def main():
    import os
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Deferred: torch locks in device visibility at first CUDA-touching
    # import, so CUDA_VISIBLE_DEVICES must be set (above) before any
    # torch/transformers-importing module is loaded.
    from img_2_svg_pretraining.data_engine.samples import discover_partial_test_set

    samples = discover_partial_test_set(Path(args.data_root))
    if args.limit:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} samples from {args.data_root}")

    STAGE_RUNNERS[args.stage](args, samples)


if __name__ == "__main__":
    main()
