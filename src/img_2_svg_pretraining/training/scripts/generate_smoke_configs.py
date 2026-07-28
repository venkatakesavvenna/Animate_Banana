#!/usr/bin/env python3
"""Generate smoke-test config variants (max_steps=200) from production configs.

Reads:
  - configs/mn_*_pt.yaml     (native PT configs)
  - configs/encoder_swap/*_pt.yaml (encoder-swap PT configs)

Writes:
  - configs/smoke/encoder_swap/{name}_smoke.yaml

Skips:
  - 72B configs (too expensive for smoke tests)
  - FT configs (smoke tests PT only)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parents[1]))

from omegaconf import OmegaConf

SMOKE_STEPS = 200
OUT_DIR = REPO_ROOT / "configs" / "smoke" / "encoder_swap"


def make_smoke(cfg_path: Path, out_dir: Path) -> Path:
    cfg = OmegaConf.load(cfg_path)
    smoke_name = cfg_path.stem + "_smoke"

    # Override smoke-specific fields
    cfg.run_name = smoke_name
    cfg.wandb_project = "img-2-svg-pretraining-smoke"
    cfg.logging_dir = f"/code/src/img_2_svg_pretraining/training/outputs/{smoke_name}/logs"
    cfg.output_dir = f"/code/src/img_2_svg_pretraining/training/outputs/{smoke_name}/checkpoints"

    # Set max_steps, remove num_train_epochs (max_steps takes precedence)
    cfg.trainer.max_steps = SMOKE_STEPS
    if "num_train_epochs" in cfg.trainer:
        del cfg.trainer["num_train_epochs"]

    # Reduce eval frequency so we get at least one eval point
    cfg.trainer.eval_steps = 50
    cfg.trainer.save_strategy = "no"
    if "captioning_eval" in cfg:
        cfg.captioning_eval.eval_steps = 50
        cfg.captioning_eval.num_samples = 16

    out_path = out_dir / f"{smoke_name}.yaml"
    out_path.write_text(OmegaConf.to_yaml(cfg))
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources: list[Path] = []

    # Native PT configs
    for p in sorted((REPO_ROOT / "configs").glob("mn_*_pt.yaml")):
        if "72b" in p.stem:
            continue
        sources.append(p)

    # Encoder-swap PT configs
    for p in sorted((REPO_ROOT / "configs" / "encoder_swap").glob("*_pt.yaml")):
        if "72b" in p.stem:
            continue
        sources.append(p)

    generated = []
    for src in sources:
        out = make_smoke(src, OUT_DIR)
        generated.append(out)

    print(f"Generated {len(generated)} smoke configs → {OUT_DIR}")
    for p in generated[:5]:
        print(f"  {p.name}")
    if len(generated) > 5:
        print(f"  ... and {len(generated) - 5} more")


if __name__ == "__main__":
    main()
