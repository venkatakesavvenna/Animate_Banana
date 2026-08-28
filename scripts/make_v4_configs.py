"""Generate the six v4 open-weights configs from one template.

WHY A GENERATOR AND NOT SIX HAND-WRITTEN FILES
----------------------------------------------
`config.py` has no `extends`, no include and no env expansion, and
`run_pipeline` has neither `--target` nor `--backend`. So the backend block,
the dataset root and all ten `backend:` lines have to be repeated in full, once
per model. Six near-identical files that must stay byte-identical except for a
single `model:` line is precisely the shape that drifts -- and the way it drifts
is silent: one config ends up judged by a different judge, or pointed at a
different dataset, and the resulting table compares two things that were never
comparable.

So the template lives here, once, and the files are generated. `--check`
regenerates in memory and diffs against disk, which is a stronger guarantee
than asserting a list of properties: it catches anything that differs, not only
what someone thought to assert.

    python3 scripts/make_v4_configs.py            # write them
    python3 scripts/make_v4_configs.py --check    # verify none has drifted

The generated files are meant to be read and committed. Edit THIS file, not
them.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "src/img_2_svg_pretraining/pipeline/configs"
REPO = "/fsxvision_new/venkat.kesav/img_2_svg_pretraining"

# stem -> (HF repo id, bf16 GB, declared architecture, vLLM registry entry)
#
# Sizes are the true safetensors totals from the HF API -- NOT `usedStorage`,
# which counts every format in the repo and overstates Gemma 3 threefold.
# Every architecture was checked against `vllm/model_executor/models/registry.py`
# in the installed 0.26.0; a repo whose `architectures` has no entry there fails
# at load, not at download, so it is worth knowing before pulling 200GB.
MODELS = {
    "qwen3vl32b":     ("Qwen/Qwen3-VL-32B-Instruct", 66.7,
                       "Qwen3VLForConditionalGeneration", "qwen3_vl"),
    "gemma3_27b":     ("google/gemma-3-27b-it", 54.9,
                       "Gemma3ForConditionalGeneration", "gemma3_mm"),
    "gemma4_31b":     ("google/gemma-4-31B-it", 62.5,
                       "Gemma4ForConditionalGeneration", "gemma4_mm"),
    "qwen36_27b":     ("Qwen/Qwen3.6-27B", 55.0,
                       "Qwen3_5ForConditionalGeneration", "qwen3_5"),
    "qwen36_35b_a3b": ("Qwen/Qwen3.6-35B-A3B", 71.0,
                       "Qwen3_5MoeForConditionalGeneration", "qwen3_5"),
    "glm46v":         ("zai-org/GLM-4.6V", 215.0,
                       "Glm4vMoeForConditionalGeneration", "glm4_1v"),
    # -- the large end. Both fit on 8xH100 only because they are MoE. --------
    "qwen3vl235b":    ("Qwen/Qwen3-VL-235B-A22B-Instruct", 471.3,
                       "Qwen3VLMoeForConditionalGeneration", "qwen3_vl_moe"),
    "internvl35_241b": ("OpenGVLab/InternVL3_5-241B-A28B-Flash", 483.4,
                       "InternVLChatModel", "internvl"),
}

# THE REST OF THE PAPER'S ABLATION ROSTER, AND WHY THEY ARE NOT HERE
# ------------------------------------------------------------------
# Measured against this node's real budget: 8 x H100 80GB = 640GB, of which
# ~576GB is addressable at --gpu-memory-utilization 0.90 before KV cache.
# Sizes are true bf16 safetensors totals from the HF API.
#
#   MiniMax-M3                    854.2 GB  needs ~11 H100s
#   Kimi K3                      1560.9 GB  needs ~20 H100s
#     (Kimi K2.5 is 595.2 GB -- still over budget once KV cache is counted)
#   DeepSeek-V4 Pro               864.7 GB  over budget AND **text-only**: the
#                                           HF card carries no
#                                           image-text-to-text tag, and every
#                                           stage of this pipeline sends an
#                                           image, so it could not serve a
#                                           single agent at any size.
#   Llama 4 Scout                 217.3 GB  fits comfortably and vLLM 0.19.0
#                                           registers Llama4ForConditionalGeneration
#                                           -- but the repo is GATED and this
#                                           HF_TOKEN gets 403 on it. Accept the
#                                           license at
#                                           huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
#                                           and it can be added with one line.
#
# NOTE ON qwen3vl235b: it is also the JUDGE. Its row is therefore the one place
# in the table where a model scores its own output, which is exactly the
# circular evaluation the judge choice exists to avoid. Report that row with
# the caveat, or judge it separately with a different model.

GEN_PORT = 8011      # generation server
JUDGE_PORT = 8010    # Qwen3-VL-235B judge, per docs/INFRA.md

# Every generating agent asks for this.
#
# 16384 was tried first and was WRONG, measurably: on Qwen3-VL-32B it truncated
# 10 of 20 stage-1 outputs. The evidence was a clean cliff -- every raw response
# at or above ~42KB ended mid-element with no closing </svg>, every one at or
# below ~35KB was complete. That is our configuration deciding the benchmark's
# result rather than the model, which is the one failure mode a model comparison
# cannot tolerate.
#
# The cause is not document size alone: these models emit reasoning tokens that
# are spent from this same budget BEFORE any content, exactly as bench_v3_or.yaml
# records for the OpenRouter route. So the usable content budget is well under
# the nominal cap.
#
# 32768 was then tried and STILL bound: raw outputs grew from ~46KB to ~85-100KB
# (the doubling the cap predicts) but 7 of 20 figures were still cut off with no
# closing tag. These are simply dense diagrams -- a faithful SVG of one runs past
# 100KB. Raising again is the only option that keeps the number a property of the
# model rather than of this file.
#
# 65536 against the served --max-model-len of 131072 leaves ~65k for the prompt,
# which is far more than any stage sends (the largest is the diagram critic
# echoing a stage-1 SVG back, ~25-30k tokens). This deliberately diverges from
# the Gemini configs' 32768 -- those ran against a paid per-token endpoint where
# the cap was also a budget. Here it is neither a budget nor a rate limit, and
# the truncation it caused was measured, not hypothetical.
MAX_TOKENS = 65536

# Per-model override, where the CHECKPOINT cannot honour the default.
# prompt + completion share one window, so a model with a short context cannot
# be given the roster's completion budget. InternVL3.5 declares 40960; 32768
# leaves ~8k for the figure and instructions, which every stage fits inside.
# Recorded here rather than silently clamped, because it is a real asymmetry in
# the comparison: that model was not offered the same budget as the others.
MAX_TOKENS_OVERRIDE = {"internvl35_241b": 32768}

TEMPLATE = '''\
# AnimateBench v4 (Set5/WACV) -- {repo}, SVG target.
#
# GENERATED by scripts/make_v4_configs.py. Edit that, not this file:
# `make_v4_configs.py --check` fails if these drift apart, because six configs
# that differ only in one `model:` line are otherwise guaranteed to diverge.
#
# One of six open-weights models run over the same 20 measurable SVG cells, with
# every other knob held fixed -- same prompts, same critic rounds, same token
# caps, same judge -- so a difference in the table is attributable to the model
# rather than to the setup.
#
# THE MODEL
#   repo          {repo}
#   weights       {size} GB bf16
#   architecture  {arch}  -> vLLM `{entry}`
#
# SERVED, NOT `hf_local`. `openai_compat` sends images as data URLs and needs no
# new backend code for a served endpoint, and `bench_qwen.yaml`'s own header
# measured the in-process route (`device_map: auto` + eager attention) at
# ">20 min per sample without finishing" against ~30 s for a served one.
# Continuous batching is the difference between an overnight run and a week.
#
# Bring the server up before running anything. The serving venv is
# ocr_env_vllm, NOT gemma4: gemma4's newer stack does not import in this
# container (its numpy 2.4.4 against the image's numpy-1.x-built scipy, then
# aiohttp against an older aiohappyeyeballs). See scripts/run_bench_v4_oss.sh.
#
#   /environments/ocr_env_vllm/bin/python -m vllm.entrypoints.openai.api_server \\
#     --model {repo} --served-model-name {repo} \\
#     --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \\
#     --max-model-len 131072 --limit-mm-per-prompt.image 2 --max-num-seqs 16 \\
#     --port {gen_port} --host 127.0.0.1
#
# `-m vllm.entrypoints.openai.api_server`, never the `vllm` console script --
# its shebang is a hard-coded /environments path. `--served-model-name` must
# match `model:` below exactly.
#
# LINEAGE. `CachePaths` keys artifacts on the MODEL, not the backend name, so
# every config here can call its backend `served` and still never collide.
# Do not replace `model:` with a generic alias -- that would collapse all six
# models onto one cache lineage and silently overwrite each other.

dataset:
  root: {repo_root}/data/animatebench_v4
  limit: 0

cache_root: {repo_root}/data/animatebench_v4_cache

animation_style: progressive_reveal
target: svg

backends:
  served:
    type: openai_compat
    base_url: http://127.0.0.1:{gen_port}/v1
    model: {repo}
    # vLLM ignores the key but the OpenAI client requires one to be present.
    api_key: EMPTY
    # 4 was the OpenRouter value, chosen to be polite to a shared paid endpoint.
    # Against a dedicated local server the opposite applies: vLLM's continuous
    # batching only pays off with several requests in flight, and at 4 (with
    # BATCH=2 upstream capping it further) the server was measured running ONE
    # request at a time -- 144 tok/s aggregate on eight H100s. There is no
    # rate limit and no bill here; the only ceiling is --max-num-seqs 16.
    max_concurrency: 12

  # The judge, run AFTER every model has generated -- one load of a 440GB model
  # scoring all six, rather than six loads. A different family from every
  # generator above, which is the point: judging a model's own output with
  # itself is the circular evaluation this project's notes warn about.
  #
  #   run_eval all --config <this file> --style <style> --judge-backend qwen_judge
  qwen_judge:
    type: openai_compat
    base_url: http://127.0.0.1:{judge_port}/v1
    model: Qwen/Qwen3-VL-235B-A22B-Instruct
    api_key: EMPTY
    max_concurrency: 8

transmuter:
  code_converter:
    backend: served
    prompt: diagram_transmuter/svg_diagram_to_code.yaml#prompt
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  raster_integrator:
    enabled: true
    backend: served
    params:
      temperature: 0.0
      max_tokens: {max_tokens}

  diagram_critic:
    enabled: true
    backend: served
    prompt: diagram_transmuter/critic.yaml
    fidelity_threshold: 0.7
    max_rounds: 3
    params:
      temperature: 0.1
      max_tokens: {max_tokens}

planner:
  # Declared, never run: its model is baked into the existing lineage and
  # removing it would rewrite every scored path. `run_pipeline strategize`
  # refuses.
  strategizer:
    backend: served
    prompt: animation_planner/strategizer.yaml#full-context-prompt
    context_tier: full

  parser:
    backend: served
    prompt: animation_planner/svg_parser.yaml#prompt
    params:
      temperature: 0.0
      max_tokens: {max_tokens}

  sequencer:
    backend: served
    prompt: animation_planner/svg_sequencer.yaml#new_prompt
    # context_tier deliberately unset, matching bench_v3_or_svg.yaml: setting it
    # appends `ctx-<tier>` to the sequence lineage, so the two variants never
    # share a cache entry.
    params:
      temperature: 0.3
      max_tokens: {max_tokens}

  critic:
    enabled: true
    backend: served
    prompt: animation_planner/critic.yaml#prompt
    max_rounds: 2
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  narrative_writer:
    backend: served
    prompt: animation_planner/narrative_writer.yaml
    context_tier: full
    params:
      temperature: 0.3
      max_tokens: {max_tokens}

animator:
  designer:
    backend: served
    # Style-keyed for SVG. `{{style}}` is expanded at run time by
    # designer._resolve_prompt, which also maps the two keys whose names differ
    # from the pipeline's. All three styles this bench measures --
    # progressive_reveal, alpha_masking, colour_pop -- are present verbatim, so
    # `--style` alone is sufficient.
    prompt: animation_designer/svg_designer.yaml#{{style}}
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  critic:
    enabled: true
    backend: served
    prompt: animation_designer/critic.yaml#prompt
    max_rounds: 3
    compile_check: true
    params:
      temperature: 0.2
      max_tokens: {max_tokens}

  exporter:
    # mp4 only -- an SVG animation is CSS keyframes, which no rasteriser can
    # seek, so there is no pdf path. mp4 is also what writes exports/frames/,
    # which the animation evaluation tree reads.
    outputs: [mp4]
    fps: 2
    dpi: 300
'''


def render(stem: str) -> str:
    repo, size, arch, entry = MODELS[stem]
    return TEMPLATE.format(repo=repo, size=size, arch=arch, entry=entry,
                           repo_root=REPO, gen_port=GEN_PORT,
                           judge_port=JUDGE_PORT,
                           max_tokens=MAX_TOKENS_OVERRIDE.get(stem, MAX_TOKENS))


def path_for(stem: str) -> Path:
    return CONFIGS / f"bench_v4_svg_{stem}.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk configs match the template; write nothing")
    args = ap.parse_args()

    drifted = []
    for stem in MODELS:
        want, dest = render(stem), path_for(stem)
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
            print(f"wrote {dest.relative_to(CONFIGS.parents[4])}")

    if args.check:
        if drifted:
            raise SystemExit(f"\n{len(drifted)} config(s) drifted from the template")
        print(f"{len(MODELS)} config(s) match the template")


if __name__ == "__main__":
    main()
