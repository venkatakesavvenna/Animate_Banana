"""Is a served model actually usable by the pipeline? Ask through the backend.

`/health` returning 200 is necessary and nowhere near sufficient. It says the
HTTP server is up; it says nothing about whether the multimodal path works,
whether the served name matches what the config asks for, or whether the
context window can hold what the pipeline sends. Every single stage of this
pipeline sends an image, so a server that is healthy but broken on images fails
on the first real call, twenty minutes into a batch.

docs/INFRA.md states the rule this follows: diagnose through the backend, never
a hand-rolled probe. A hand-rolled client once reported every key in a pool as
dead because `genai.Client` closes on garbage collection -- an hour lost to a
probe that was wrong in a way the real code was not. So this constructs the
same backend from the same config the run will use, and sends a real image.

    python scripts/vllm_smoke.py --config <cfg> --backend served --image <png>

Exit 0 means the pipeline can use this server.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_2_svg_pretraining.pipeline.backends import make_backend  # noqa: E402
from img_2_svg_pretraining.pipeline.backends.types import Message  # noqa: E402
from img_2_svg_pretraining.pipeline.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default="served")
    ap.add_argument("--image", type=Path,
                    help="a real sample figure; defaults to the first in the dataset")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    image = args.image
    if image is None:
        root = Path(cfg.dataset_root)
        image = next(iter(sorted(root.glob("*/*.png"))), None)
    if image is None or not image.exists():
        print(f"SMOKE FAIL: no sample image to send (looked in {cfg.dataset_root})")
        return 2

    backend = make_backend(args.backend, cfg.backend_cfg(args.backend), cache_root=None)
    print(f"backend={args.backend} model={cfg.backend_cfg(args.backend).get('model')}")
    print(f"image={image.name}")

    started = time.time()
    reply = backend.generate(
        [Message.user("Reply with the single word OK.", images=[str(image)])],
        max_tokens=args.max_tokens)
    elapsed = time.time() - started

    if not reply.ok:
        print(f"SMOKE FAIL after {elapsed:.1f}s: {reply.error}")
        return 1
    text = (reply.text or "").strip()
    if not text:
        # A server that thinks past max_tokens returns an empty completion with
        # no error -- this repo already has that scar from OpenRouter, where the
        # failure surfaced as "response was not a JSON object".
        print(f"SMOKE FAIL after {elapsed:.1f}s: empty completion "
              f"(all {args.max_tokens} tokens likely spent on reasoning)")
        return 1
    print(f"SMOKE OK in {elapsed:.1f}s: {text[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
