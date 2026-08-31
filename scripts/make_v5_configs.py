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

# API-ONLY. Gemini is a closed model: there are no weights to serve, so it has
# an OpenRouter config and NO local twin. Everything above is the mirror image
# -- open weights, served locally, never sent to an API. That split is
# deliberate and must hold: routing an open-weights model through OpenRouter
# would silently make the comparison about a vendor's serving stack instead of
# the model, and it costs money for a thing we already run for free.
OR_ONLY = {
    "gemini37flash": ("google/gemini-3.7-flash", "google/gemini-3.7-flash",
                      0.0, "closed", "api"),
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
# EMPTY now, and that is the fix, not an oversight. gemma4_31b was capped at
# 8192 only because it was pinned to TP=1 (its old venv's NCCL SIGSEGVd), which
# left ~10GB of KV cache and a 12288 context. The clean v5serve venv does TP=8,
# so it gets the full 131072 context and the same 65536 completion budget as
# every other model -- removing the one real asymmetry in the roster.
# Kimi's completion budget is capped at 32768 while every other model gets
# 65536. This is a TAIL control, not a memory one, and the distinction matters:
#
# Kimi is a thinking model -- it spends completion tokens reasoning before it
# emits any content. On the densest figures it reasons for thousands of tokens,
# and a single such request decodes ALONE at ~79 tok/s while the other 15 batch
# slots sit idle. Measured: a full batch sustains ~690 tok/s, so one straggler
# costs ~9x the throughput of the whole server. Averaged over a run the tail,
# not the batch, sets the wall clock.
#
# 32768 still clears p95 completion (20,328 tokens) with room, so typical cells
# are unaffected; it truncates only the pathological ones, which were failing
# anyway with "no svg document in the response".
#
# THE SERVED WINDOW STAYS AT 131072. Shrinking that was tried twice and broke
# the designer stage both times -- vLLM rejects prompt+max_tokens >
# max_model_len outright rather than clamping, so the window must exceed the
# completion cap by a whole prompt. Cap the REQUEST, never the window.
#
# This is a real asymmetry in the comparison and the Kimi row should carry it.
MAX_TOKENS_OVERRIDE: dict[str, int] = {"kimi_k26": 32768}

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
  root: {repo_root}/data/animatebench_v5
  limit: 0

cache_root: {repo_root}/data/{cache_dir}

animation_style: progressive_reveal
target: svg

backends:
  {backend}:
{backend_body}

  # The judge. Run AFTER generation, so one load scores every config rather
  # than reloading per model. A different family from most generators here --
  # judging a model's output with itself is the circular evaluation this
  # project's notes warn against, and it is why the qwen3vl235b row needs a
  # footnote.
  qwen_judge:
    type: openai_compat
    base_url: http://127.0.0.1:{judge_port}/v1
    model: Qwen/Qwen3-VL-235B-A22B-Instruct
    api_key: EMPTY
    max_concurrency: 8

  # Kimi K2.6 as judge, served across 2 nodes on :8011.
  #
  # max_concurrency 64, not 8. The v3 animation-tree run is the cautionary
  # tale: it scored 45 cells in 3.8h at "Running: 1 reqs" against
  # --max-num-seqs 16, leaving ~94% of the server idle, because the DRIVER was
  # sequential -- one run_eval subprocess per cell. The judging work is
  # embarrassingly parallel (every cell independent), so the only thing that
  # ever limited it was how many requests were in flight. This server sustains
  # Running: 32 at ~1090 tok/s; 64 keeps the queue non-empty as cells finish.
  kimi_judge:
    type: openai_compat
    base_url: http://127.0.0.1:{gen_port}/v1
    model: moonshotai/Kimi-K2.6
    api_key: EMPTY
    max_concurrency: 8
    # THINKING OFF FOR JUDGING. Kimi emits a <think> block before any content;
    # its chat template takes `thinking: false` and pre-fills an empty one.
    # Measured on one banding call: 11.1s / 710 completion tokens with thinking
    # on, 0.2s / 6 tokens with it off -- and the reply is cleaner, a bare
    # {{"band":"B"}} rather than prose the parser must dig through. Over ~22k
    # judged calls that is a night instead of a week.
    #
    # A judge assigns a band against a rubric; it is not the task reasoning
    # helps. Generation keeps thinking ON -- that is `served`, a different
    # backend, and this setting does not touch it.
    default_params:
      extra_body:
        chat_template_kwargs:
          thinking: false

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
    enabled: {critics}
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
    enabled: {critics}
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
    enabled: {critics}
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
    max_concurrency: 16"""

OR_BODY = """\
    type: openai_compat
    base_url: https://openrouter.ai/api/v1
    model: {slug}
    api_key_env: OPEN_ROUTER_KEY
    # Raised from 4 once the key's credit cap was lifted. This stream must
    # cover all 91 cells by itself overnight, and OpenRouter fans out far
    # better than a single local server does.
    max_concurrency: 12"""

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


# CRITICS OFF FOR THE OVERNIGHT GENERATION RUN
# --------------------------------------------
# All three critics (transmuter/critic.py, planner/critic.py,
# animator/critic.py) call `backend.generate()` on ONE sample at a time inside a
# per-sample loop -- they never use `generate_batch`. So during a critic pass the
# server sits at num_requests_running == 1 no matter how large the batch or how
# high max_concurrency is; measured directly, pinned at exactly 1.0 for four
# minutes straight. With max_rounds 3+2+3 that is up to eight serialised round
# trips per sample, and it projected the 91-cell x 4-model sweep at ~22h against
# an 8h window.
#
# They are refinement passes: the deliverable here is generated animations, and
# a critic improves an artifact that already exists. Turning them off is the one
# change that makes the run fit, and it is reversible -- rerunning with
# CRITICS=on refines the same artifacts in place later.
#
# NOTE this changes NOTHING about lineage: critics are not part of any cache
# lineage key, so a later critic pass writes to code_reviewed/ etc. beside these
# artifacts rather than colliding with them.
CRITICS = False


def render(stem: str, route: str) -> str:
    repo, slug, size, arch, venv = {**MODELS, **OR_ONLY}[stem]
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
        judge_port=JUDGE_PORT, gen_port=GEN_PORT, critics=str(CRITICS).lower(),
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
    for stem in list(MODELS) + list(OR_ONLY):
        routes = ("or",) if stem in OR_ONLY else ("local", "or")
        for route in routes:
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
        print(f"{2 * len(MODELS) + len(OR_ONLY)} config(s) match the template")


if __name__ == "__main__":
    main()
