from pathlib import Path
from img_2_svg_pretraining.pipeline.config import load_config
from img_2_svg_pretraining.pipeline.cache import CachePaths
from img_2_svg_pretraining.pipeline.export.tikz_source import to_multipage_pdf_source
from img_2_svg_pretraining.pipeline.export.render import compile_pdf, RenderError

STYLES = ["progressive_reveal","colour_pop","alpha_masking",
          "hopping_bounding_box","sliding_bounding_box"]
for cfgname in ["bench_qwen","bench_gemma4"]:
    cfg = load_config(f"src/img_2_svg_pretraining/pipeline/configs/{cfgname}.yaml")
    for style in STYLES:
        cfg.style = style; cfg.raw["animation_style"] = style
        p = CachePaths.from_config(cfg)
        for sid in ["CVPR_2025_pipe00002","CVPR_2025_pipe00041"]:
            src = p.animation(sid)
            if not src.exists():
                continue
            mp4 = p.exports(sid)/"animation.mp4"
            if mp4.exists():
                print(f"  {cfgname[6:]:7} {style:22} {sid[-8:]}  already MP4")
                continue
            try:
                compile_pdf(to_multipage_pdf_source(src.read_text()), Path("/tmp/x.pdf"))
                print(f"  {cfgname[6:]:7} {style:22} {sid[-8:]}  COMPILES (export should work)")
            except RenderError as e:
                hit = [l for l in str(e).splitlines() if l.startswith("!")]
                print(f"  {cfgname[6:]:7} {style:22} {sid[-8:]}  {hit[0][:60] if hit else 'unknown'}")
