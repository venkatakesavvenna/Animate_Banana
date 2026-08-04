"""Write a single-model pipeline config for the sweep.

    sweep_config.py <name> <hf_repo> <attn> <batch> <limit> <dataset> <outdir>

A separate file rather than a heredoc inside `docker exec bash -c "..."`:
nesting a heredoc in a double-quoted command string loses the quoting and
silently writes nothing, which cost a sweep launch.
"""
from __future__ import annotations

import pathlib
import sys

import yaml


def main() -> None:
    name, repo, attn, batch, limit, dataset, outdir = sys.argv[1:8]

    src = (pathlib.Path(__file__).parent / "src" / "img_2_svg_pretraining"
           / "pipeline" / "configs" / "hf_local.yaml")
    cfg = yaml.safe_load(src.read_text())

    cfg["dataset"] = {"root": dataset, "limit": int(limit)}
    cfg["cache_root"] = f"{outdir}/cache"
    cfg["backends"] = {
        name: {
            "type": "hf_local",
            "hf_repo": repo,
            "model": name,
            "model_class": "image_text_to_text",
            "attn_implementation": attn,
            "dtype": "bfloat16",
            "device": "cuda",
            "batch_size": int(batch),
        }
    }
    for section in ("transmuter", "planner", "animator"):
        for agent in cfg.get(section, {}).values():
            if isinstance(agent, dict) and "backend" in agent:
                agent["backend"] = name
    # Stage 1b is out of scope for this sweep.
    cfg["transmuter"]["raster_integrator"]["enabled"] = False

    out = pathlib.Path(outdir) / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
