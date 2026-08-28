"""Generate the v5 bring-up configs: one LOCAL and one OPENROUTER per model.

WHY A GENERATOR, AND WHY PAIRS
------------------------------
Same reasoning as make_v4_configs.py: `config.py` has no `extends`, so the
backend block and all ten `backend:` lines are repeated per model, and files
that must stay byte-identical except one line are exactly what drifts silently.

The v5 twist is that every model needs TWO configs -- the local vLLM server and
the same model on OpenRouter -- so their outputs can be compared. Those two must
differ in *nothing* but the endpoint, or the parity check measures the config
instead of the transport.

    python3 scripts/make_v5_configs.py          # write them
    python3 scripts/make_v5_configs.py --check  # verify none drifted

THE COLLISION THIS FILE EXISTS TO PREVENT
-----------------------------------------
`CachePaths` builds its lineage from the MODEL STRING ONLY -- not base_url, not
the backend name, not the config name. Point a local and an OpenRouter config at
the same model and they write to the SAME directory; the second run then finds
every artifact present and skips all work, silently, because every stage skips
when its output exists. The parity check would then compare a run against
itself and always pass.

Two independent separations are applied here, deliberately belt-and-braces:
  1. a distinct `cache_root` per route (the only fully safe mechanism), and
  2. a distinct backend name, which also splits the RESPONSE cache
     (`<cache_root>/<dataset>/responses/<backend_name>/`).
"""
from __future__ import annotations

import argparse, difflib, sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "src/img_2_svg_pretraining/pipeline/configs"
REPO = "/fsxvision_new/venkat.kesav/img_2_svg_pretraining"

# stem -> (HF repo id, OpenRouter slug, GB, architecture, serving venv)
#
# Sizes are true safetensors totals from the HF API (NOT `usedStorage`, which
# counts every format in the repo). Every architecture was checked against the
# INSTALLED vLLM registry before any download -- see the GLM-5.3 note below.
MODELS = {
    "qwen38_27b":  ("Qwen/Qwen3.8-27B", "qwen/qwen3.8-27b", 55.6,
                    "Qwen3_5ForConditionalGeneration", "v5serve"),
    "gemma4_31b":  ("google/gemma-4-31B-it", "google/gemma-4-31b-it", 62.5,
                    "Gemma4ForConditionalGeneration", "gemma4"),
    "glm46v":      ("zai-org/GLM-4.6V", "z-ai/glm-4.6v", 215.4,
                    "Glm4vMoeForConditionalGeneration", "v5serve"),
    "kimi_k26":    ("moonshotai/Kimi-K2.6", "moonshotai/kimi-k2.6", 595.2,
                    "KimiK25ForConditionalGeneration", "v5serve"),
    "qwen3vl235b": ("Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "qwen/qwen3-vl-235b-a22b-instruct", 471.3,
                    "Qwen3VLMoeForConditionalGeneration", "v5serve"),
}

# GLM-5.3-Flash IS NOT HERE, AND CANNOT BE
# ----------------------------------------
# `zai-org/GLM-5.3-Flash` (328.3GB) declares Glm5NextForConditionalGeneration.
# That architecture is absent from the vLLM registry in 0.19.0, in 0.26.0, in
# the latest release 0.28.0, AND on the unreleased main branch -- checked
# directly, zero `Glm5*` entries anywhere. There is no released vLLM that can
# serve it; the checkpoint shipped ahead of inference support.
#
# It is reachable on OpenRouter (`z-ai/glm-5.3-flash`, text+image+video), so an
# API-only row is possible, but no LOCAL row and therefore no parity check.
#
# Note also that its HF `pipeline_tag` says `text-generation` and is WRONG --
# config.json carries a real vision_config plus image_token_id /
# image_start_token_id / image_end_token_id. Do not use that tag to judge
# modality. `zai-org/GLM-5.3` proper (755.6GB, GlmMoeDsaForCausalLM) IS
# genuinely text-only, which is a different model and a real exclusion.
#
# Kimi K3 is likewise absent: 1560.9GB needs ~20 H100s (NVIDIA's NVFP4 repo is
# no smaller at 1609.9GB). K2.6 above is the vision-capable model that fits.

GEN_PORT, JUDGE_PORT = 8011, 8010
MAX_TOKENS = 65536   # 16384 and 32768 were both MEASURED truncating stage-1
                     # SVGs mid-element -- see make_v4_configs.py's long note.

# Per-model cap, where the SERVED CONTEXT cannot honour the default.
#
# gemma4_31b is served at TP=1 (its venv's NCCL is the wrong CUDA generation, so
# multi-GPU SIGSEGVs). 62.5GB of weights on one 80GB card leaves ~10GB of KV
# cache, which caps --max-model-len at ~12288. vLLM does not clamp an oversized
# request -- it REJECTS it with HTTP 400 ("max_tokens=65536 cannot be greater
# than max_model_len"), so every stage would fail outright rather than degrade.
#
# Recorded here rather than silently applied, because it is a REAL ASYMMETRY:
# this model is not offered the same completion budget as the others, and a
# truncated SVG scores worse. Its row is not strictly comparable, and the same
# cap must be applied to its OpenRouter twin or the parity check would compare
# a 12k-budget local run against a 64k-budget remote one and blame the model.
MAX_TOKENS_OVERRIDE = {"gemma4_31b": 8192}

TEMPLATE = '''\
# AnimateBench v5 bring-up -- {repo} via {route}, SVG target.
#
# GENERATED by scripts/make_v5_configs.py. Edit that, not this file.
#
# THE MODEL
#   repo          {repo}
#   weights       {size} GB
#   architecture  {arch}
#   serving venv  {venv}
{serve_note}
# CACHE SEPARATION. `CachePaths` keys artifacts on the MODEL STRING ALONE --
# not base_url, not the backend name. The local and OpenRouter configs for this
# model therefore use DIFFERENT cache_roots; without that the second route finds
# the first's artifacts already on disk, skips every stage, and the parity check
# silently compares a run against itself.

dataset:
  root: {repo_root}/data/animatebench_v3
  limit: 0

cache_root: {repo_root}/data/{cache_dir}

animation_style: progressive_reveal
target: svg

backends:
  {backend}:
{backend_body}

  # The judge. Run AFTER generation, so one load of the 471GB model scores every
  # config rather than reloading per model. A different family from most
  # generators here -- judging a model's output with itself is the circular
  # evaluation this project's notes warn against, and it is why the
  # qwen3vl235b row needs a footnote.
  qwen_judge:
    type: openai_compat
    base_url: http://127.0.0.1:{judge_port}/v1
    model: Qwen/Qwen3-VL-235B-A22B-Instruct
    api_key: EMPTY
    max_concurrency: 8

transmuter:
  code_converter:
    backend: {backend}
    prompt: diagram_transmuter/svg_diagram_to_code.yaml#prompt
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  raster_integrator:
    enabled: true
    backend: {backend}
    params:
      temperature: 0.0
      max_tokens: {max_tokens}

  diagram_critic:
    enabled: true
    backend: {backend}
    prompt: diagram_transmuter/critic.yaml
    fidelity_threshold: 0.7
    max_rounds: 3
    params:
      temperature: 0.1
      max_tokens: {max_tokens}

planner:
  # Declared, never run: its model is baked into the existing lineage and
  # removing it would rewrite every scored path. `run_pipeline strategize`
  # refuses to execute.
  strategizer:
    backend: {backend}
    prompt: animation_planner/strategizer.yaml#full-context-prompt
    context_tier: full

  parser:
    backend: {backend}
    prompt: animation_planner/svg_parser.yaml#prompt
    params:
      temperature: 0.0
      max_tokens: {max_tokens}

  sequencer:
    backend: {backend}
    prompt: animation_planner/svg_sequencer.yaml#new_prompt
    params:
      temperature: 0.3
      max_tokens: {max_tokens}

  critic:
    enabled: true
    backend: {backend}
    prompt: animation_planner/critic.yaml#prompt
    max_rounds: 2
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  narrative_writer:
    backend: {backend}
    prompt: animation_planner/narrative_writer.yaml
    context_tier: full
    params:
      temperature: 0.3
      max_tokens: {max_tokens}

animator:
  designer:
    backend: {backend}
    # `{{style}}` is expanded at run time by designer._resolve_prompt.
    prompt: animation_designer/svg_designer.yaml#{{style}}
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  critic:
    enabled: true
    backend: {backend}
    prompt: animation_designer/critic.yaml#prompt
    max_rounds: 3
    compile_check: true
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  exporter:
    # mp4 only -- an SVG animation is CSS keyframes, which no rasteriser can
    # seek. mp4 also writes exports/frames/, which the animation tree reads.
    outputs: [mp4]
    fps: 2
    dpi: 300
'''

LOCAL_BODY = """\
    type: openai_compat
    base_url: http://127.0.0.1:{gen_port}/v1
    model: {repo}
    # vLLM ignores the key; the OpenAI client requires one to be present.
    api_key: EMPTY
    # Against a dedicated local server, batching only pays off with several
    # requests in flight. At 4 the v4 server was measured running ONE request
    # at a time -- 144 tok/s aggregate on eight H100s. Ceiling is --max-num-seqs.
    max_concurrency: 12"""

OR_BODY = """\
    type: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: {slug}
    api_key_env: OPEN_ROUTER_KEY
    # 4, not 12: this is a shared paid endpoint with a hard credit cap.
    max_concurrency: 4"""

SERVE_NOTE = """#
# Bring the server up first:
#
#   /environments/{venv}/bin/python -m vllm.entrypoints.openai.api_server \\
#     --model {repo} --served-model-name {repo} \\
#     --tensor-parallel-size {tp} --gpu-memory-utilization 0.90 \\
#     --max-model-len 131072 --limit-mm-per-prompt.image 2 --max-num-seqs 16 \\
#     --port {gen_port} --host 127.0.0.1
#
# `-m vllm.entrypoints.openai.api_server`, never the `vllm` console script --
# its shebang is a hard-coded /environments path. `--served-model-name` must
# match `model:` below exactly, and run it INSIDE the container: on the host
# python3 is 3.10 and every 3.12 venv's site-packages drop off sys.path.
"""

def tp_for(stem: str) -> int:
    # gemma4's venv has cu130 torch on a cu128 image, so its NCCL is the wrong
    # generation and multi-GPU SIGSEGVs. 62.5GB fits one 80GB card, so TP=1
    # sidesteps NCCL entirely.
    return 1 if stem == "gemma4_31b" else 8


def render(stem: str, route: str) -> str:
    repo, slug, size, arch, venv = MODELS[stem]
    if route == "local":
        backend, body = "served", LOCAL_BODY.format(gen_port=GEN_PORT, repo=repo)
        note = SERVE_NOTE.format(venv=venv, repo=repo, tp=tp_for(stem), gen_port=GEN_PORT)
        cache_dir = "animatebench_v5_cache"
    else:
        backend, body = f"openrouter_{stem}", OR_BODY.format(slug=slug)
        note = "#\n# No server: this route is OpenRouter. Needs OPEN_ROUTER_KEY in .env.\n"
        cache_dir = "animatebench_v5_or_cache"
    return TEMPLATE.format(
        repo=repo, size=size, arch=arch, venv=venv, route=route,
        serve_note=note, repo_root=REPO, cache_dir=cache_dir,
        backend=backend, backend_body="\n".join("  " + l for l in body.splitlines()),
        judge_port=JUDGE_PORT,
        max_tokens=MAX_TOKENS_OVERRIDE.get(stem, MAX_TOKENS))


def path_for(stem: str, route: str) -> Path:
    return CONFIGS / (f"bench_v5_svg_{stem}.yaml" if route == "local"
                      else f"bench_v5_or_{stem}.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk configs match the template; write nothing")
    args = ap.parse_args()

    drifted = []
    for stem in MODELS:
        for route in ("local", "or"):
            want, dest = render(stem, route), path_for(stem, route)
            if args.check:
                have = dest.read_text(encoding="utf-8") if dest.exists() else ""
                if have != want:
                    drifted.append(dest.name)
                    print(f"DRIFTED {dest.name}")
                    sys.stdout.writelines(difflib.unified_diff(
                        have.splitlines(True), want.splitlines(True),
                        fromfile="on disk", tofile="template", n=1))
            else:
                dest.write_text(want, encoding="utf-8")
                print(f"wrote {dest.name}")

    if args.check:
        if drifted:
            raise SystemExit(f"\n{len(drifted)} config(s) drifted")
        print(f"{2 * len(MODELS)} config(s) match the template")


if __name__ == "__main__":
    main()
