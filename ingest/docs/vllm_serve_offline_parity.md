# Replicating `vllm serve` exactly in the offline AsyncLLM engine

**Status:** Divergence 1 (scheduler `usage_context`) and Divergence 2 (`generation_config.json`
merge) implemented 2026-07-28 — see `docs/v1.6_changes.md`'s "2026-07-28" addendum for exactly what
shipped and where it deviates from the "Recommended changes" section below. Divergence 3 (image
loading) remains investigation-only / explicitly deferred. This document is left as originally
written (the investigation), since it's still the accurate root-cause reference.

**Environment investigated:** `/environments/gemma4_new`, vLLM **0.25.1**, 4× **H100 80GB HBM3**,
model `google/gemma-4-31B-it` (HF cache `/hf_cache/hub/models--google--gemma-4-31B-it`,
snapshot `842da3794eaa0b77d5f08bae87a17459d91ff475`).

**Question this answers:** why does offline `AsyncLLM` inference produce worse output than
`vllm serve` for the same model and the same `engine_args`, and what is the exact,
complete set of things needed to close the gap?

**Short answer:** matching `engine_args` is *not* sufficient. There are three independent
divergence mechanisms, only one of which is visible in `engine_args`. The
`max_num_batched_tokens` discrepancy that prompted this investigation is a symptom of
the largest one.

All vLLM paths below are relative to:

```
/fsxvision_new/srihari.bandarupalli/environments/gemma4_new/lib/python3.12/site-packages/vllm/
```

---

## Table of contents

1. [Divergence 1 — `UsageContext` decides the scheduler defaults](#divergence-1)
2. [Divergence 2 — `generation_config.json` is merged only by the HTTP layer](#divergence-2)
3. [Divergence 3 — image loading (`cv2.imread` vs vLLM's PIL pipeline)](#divergence-3)
4. [Verified identical — ruled out, do not spend time here](#ruled-out)
5. [Recommended changes](#recommended-changes)
6. [Honest caveat on "exact same outputs"](#caveat)
7. [Appendix — how vLLM assembles its config](#appendix)

---

<a name="divergence-1"></a>

## Divergence 1 — `UsageContext` decides the scheduler defaults

This is the root cause of the observed `max_num_batched_tokens` 8192-vs-2048 difference,
and it is the highest-impact finding.

vLLM selects scheduler defaults from a lookup table keyed by **who is constructing the
engine**, not by what the model needs. See `engine/arg_utils.py:2396` (`get_batch_defaults`).

For a GPU with ≥ 70 GiB that is not an A100 — which is our H100 80GB — the table is:

| Constructor | `UsageContext` passed | `max_num_batched_tokens` | `max_num_seqs` |
|---|---|---|---|
| `vllm serve` | `OPENAI_API_SERVER` | **8192** | **1024** |
| `LLM(...)` — our `"batch"` backend | `LLM_CLASS` | 16384 | 1024 |
| `AsyncLLM.from_engine_args(...)` — our `"async"` backend | **`ENGINE_CONTEXT`** | **2048** | **128** |

### Why the third row happens

`AsyncLLM.from_engine_args` defaults its parameter to `ENGINE_CONTEXT`:

```python
# v1/engine/async_llm.py:232-242
def from_engine_args(
    cls,
    engine_args: AsyncEngineArgs,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,   # <-- here
    ...
):
    vllm_config = engine_args.create_engine_config(usage_context)
```

But `ENGINE_CONTEXT` **is not a key in either defaults dict** — the dicts only contain
`LLM_CLASS` and `OPENAI_API_SERVER`. So the lookup silently falls through to the `.get()`
fallback:

```python
# engine/arg_utils.py:2605-2620
self.max_num_batched_tokens = default_max_num_batched_tokens.get(
    usage_context,
    SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,   # 2048
)
...
self.max_num_seqs = default_max_num_seqs.get(
    usage_context,
    SchedulerConfig.DEFAULT_MAX_NUM_SEQS,             # 128
)
```

with, from `config/scheduler.py:42-44`:

```python
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
DEFAULT_MAX_NUM_SEQS:           ClassVar[int] = 128
```

This is a silent fallthrough, not a documented default. Nothing warns about it.

### The "raised from 2048 to ~21xx to accommodate video" log line

This was correctly remembered and is real. At `engine/arg_utils.py:2643`:

```python
# For multimodal prefix-LM models (e.g., Gemma 4) that disable
# chunked MM input, a single multimodal item must fit in one batch.
# Raise the floor to accommodate the largest per-item token count.
if model_config.is_multimodal_model and model_config.is_mm_prefix_lm:
    result = self._get_min_mm_batched_tokens(model_config)
    ...
        logger.info(
            "Raising max_num_batched_tokens from %d to %d to "
            "accommodate '%s' input for prefix-LM model %s.", ...)
```

Gemma 4 qualifies as a prefix-LM: its `config.json` has
`text_config.use_bidirectional_attention: "vision"`.

The floor value for Gemma 4 comes from `model_executor/models/gemma4_mm.py`:

```python
_VIDEO_MAX_SOFT_TOKENS = 70   # soft tokens per video frame (vs 280 for images)
_VIDEO_MAX_FRAMES      = 32   # max sampled frames per video
...
tokens["video"] = num_frames * (_VIDEO_MAX_SOFT_TOKENS + 2 + 6)
```

→ `32 × 78 = ` **2496** tokens.

So offline defaulted to 2048, then got raised to 2496 for the `video` modality.
Online never hit this because 8192 > 2496.

Two important notes:

- This floor-raise **only fires when `max_num_batched_tokens` was not set explicitly**
  (guarded by `if orig_max_num_batched_tokens is None`). Setting it explicitly skips it.
- Setting `limit_mm_per_prompt: {video: 0}` does **not** avoid it. The computation iterates
  `info.supported_mm_limits` (model capability), and the `num_frames` reduction only applies
  when the limit is a `VideoDummyOptions` object with `num_frames` set — a plain `0` does not
  reduce it.

### Why the manual `max_num_batched_tokens: 8192` fix helped but was still slow

Because only one of the two knobs was corrected. `max_num_seqs` remained at **128** versus
serve's **1024** — an 8× reduction in maximum concurrent sequences. That fully explains
"better quality than the 2048 run, but much slower than online".

The long startup with 100% GPU utilisation and no throughput is CUDA-graph capture /
`torch.compile` warmup across a larger set of shapes, which grows with the batch budget.

### Scope of this mechanism — tightly bounded (good news)

`usage_context` appears **exactly once** in the body of `create_engine_config`: in the call
to `_set_default_max_num_seqs_and_batched_tokens_args`. Verified by scanning the whole
function body (`engine/arg_utils.py:1818`–~2300).

**`max_num_batched_tokens` and `max_num_seqs` are the only two engine settings affected by
usage context.** There is nothing else hiding behind this mechanism.

### Why this changes output quality at all

Mathematically it should not. In practice it does, for the reason described in
<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>: vLLM's
kernels are not batch-invariant, so reduction order — and therefore floating-point
results — depends on how many tokens/sequences are batched together. See
[the caveat section](#caveat).

---

<a name="divergence-2"></a>

## Divergence 2 — `generation_config.json` is merged only by the HTTP layer

The startup warning

```
Default vLLM sampling parameters have been overridden by the model's `generation_config.json`:
`{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`.
```

is emitted from `ModelConfig.get_diff_sampling_param()` (`config/model.py:1483-1535`), which
reads `GenerationConfig.from_pretrained(model).to_diff_dict()` and keeps only
`repetition_penalty, temperature, top_k, top_p, min_p, max_new_tokens`.

**Crucially, the actual merge is implemented in the OpenAI protocol layer, not the engine.**
`ChatCompletionRequest.to_sampling_params()`
(`entrypoints/openai/chat_completion/protocol.py:588-615`):

```python
if (temperature := self.temperature) is None:
    temperature = default_sampling_params.get(
        "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
if (top_p := self.top_p) is None:
    top_p = default_sampling_params.get("top_p", ...)
if (top_k := self.top_k) is None:
    top_k = default_sampling_params.get("top_k", ...)
# ... same for min_p, repetition_penalty
```

i.e. **per-field**: any field the HTTP request leaves `None` is filled from the model's
generation config.

The offline engines do no such thing:

- `AsyncLLM.generate()` has **no** merging logic whatsoever — the `SamplingParams` you pass
  is used verbatim.
- `LLM.generate()` applies model defaults **only if `sampling_params is None` entirely**
  (`entrypoints/llm.py:415-420`, `473-474`). Passing any explicit `SamplingParams` object —
  which `vllm_config.py` always does — bypasses it completely.

Anything left unset therefore falls back to vLLM's neutral class defaults
(`sampling_params.py:232-246`): `repetition_penalty=1.0, temperature=1.0, top_p=1.0,
top_k=0, min_p=0.0`.

> **Trap:** adding `generation_config: "auto"` to `engine_args` does **nothing** for the
> offline paths. It is already the `ModelConfig` default, and no offline code path ever
> calls `get_diff_sampling_param()`.

### Gemma 4's actual generation config

`/hf_cache/.../generation_config.json`:

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": [1, 106, 50],
  "pad_token_id": 0,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95,
  "transformers_version": "5.5.0.dev0"
}
```

### Current effective state

Against `vllm_config.py:get_sampling_params()`, which sets only
`temperature`, `max_tokens`, `stop_token_ids`, `repetition_penalty`:

| Param | Online today | Offline today | Diverges? |
|---|---|---|---|
| `temperature` | 0.2 — sent explicitly by `online_worker` | 0.2 (yaml fallback) | no |
| `top_p` | **0.95** (server fills from gen config) | **1.0** (vLLM neutral) | **yes** |
| `top_k` | **64** (server fills from gen config) | **0 = disabled** | **yes** |
| `repetition_penalty` | 1.0 | 1.0 | no |
| `max_tokens` | 4096 | 4096 | no |

Important nuance: **temperature is not currently a divergence.**
`online_worker._build_chat_payload` sends `"temperature": getattr(sp, "temperature", 0.2)`
explicitly, which overrides the model's 1.0 on the server side too. The live gaps are
`top_p` and `top_k` — and `top_k=0` (unrestricted) versus `top_k=64` (truncated to the
top 64 logits) is a substantial behavioural difference, especially at higher temperature.

### Two different parity targets — a decision is required

- **True `vllm serve` default parity** → do not send `temperature` at all; let
  `generation_config.json` supply `1.0`.
- **Parity with our current online output** → keep `temperature: 0.2`, and only add
  `top_p: 0.95` / `top_k: 64`.

These are different targets and must be chosen deliberately.

---

<a name="divergence-3"></a>

## Divergence 3 — image loading (`cv2.imread` vs vLLM's PIL pipeline)

Affects VLM stages only; not the text-only Hindi-caption task.

**Server side** — `file://` URLs go through `ImageMediaIO.load_file` → `load_bytes`
(`multimodal/media/image.py:73-95`):

```python
image = Image.open(BytesIO(data))
# enforces VLLM_MAX_IMAGE_PIXELS
image = normalize_image(image)       # ImageOps.exif_transpose
image.load()
image = self._convert_image_mode(image)   # RGBA -> RGB composited on WHITE (255,255,255)
```

The RGBA→RGB conversion **composites onto a white background**
(`multimodal/image.py:28-36`, `rgba_to_rgb`), and `_has_transparency` also catches
`LA`/`PA` modes and PNG `tRNS` chunks.

**Our offline side** — `src/vision_ingest/utils/utils.py:load_cv2_pil`:

```python
img = cv2.imread(str(p))                                    # BGR, alpha DISCARDED
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
```

`cv2.imread` with the default `IMREAD_COLOR` **drops the alpha channel** rather than
compositing it, exposing whatever raw RGB sits underneath.

### Empirically verified

Test: 8×8 RGBA PNG, fully transparent region whose underlying RGB is pure red.

```
vLLM serve  transparent-region pixel: [255 255 255]
offline cv2 transparent-region pixel: [255   0   0]
identical arrays? False
```

For any RGBA / `tRNS` PNG the two paths feed **different pixels** into the vision tower.
Transparent regions become white online and typically black (or arbitrary) offline.

Also note `VLLM_MAX_IMAGE_PIXELS` is enforced server-side only — offline oversized images
are silently accepted rather than rejected.

---

<a name="ruled-out"></a>

## Verified identical — ruled out, do not spend time here

These were each specifically investigated and confirmed to be **non-issues**. Recorded so
they are not re-investigated later.

### Chat template — identical (verified by execution)

`AutoTokenizer.apply_chat_template` (our `is_llm: true` branch) and
`AutoProcessor.apply_chat_template` (what the server uses) produce **byte-identical**
strings for text-only input:

```
'<bos><|turn>user\nTranslate to Hindi: hello world<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
IDENTICAL: True
```

Both resolve the same `chat_template.jinja` shipped in the model repo. (Note the model has
no `chat_template` key in `tokenizer_config.json`; it ships a standalone
`chat_template.jinja`, which transformers 5.x loads automatically.)

### `enable_thinking` — correct on both sides

The template does `{%- set enable_thinking = enable_thinking | default(false) -%}`, so both
paths default to `false`. When thinking is disabled the template deliberately emits an
**empty** thought channel to suppress reasoning:

```jinja
{%- if not enable_thinking -%}
    {{- '<|channel>thought\n<channel|>' -}}
{%- endif -%}
```

The trailing `<|channel>thought\n<channel|>` in the rendered prompt is therefore correct
behaviour, not a bug.

### Double-BOS — not an issue for this model

A classic failure mode when feeding an already-templated string back through the tokenizer.
Specifically tested:

```
add_special_tokens=True   len=19 first6=[2, 105, 2364, 107, 40414, 531]
add_special_tokens=False  len=19 first6=[2, 105, 2364, 107, 40414, 531]
```

Identical — the Gemma 4 tokenizer does not auto-prepend BOS.

For reference, vLLM's own defaults do differ by path
(`renderers/base.py:340-369`): completion/plain-prompt → `add_special_tokens=True`;
chat → `add_special_tokens=False`; but for **multimodal** models both defer to
`mm_processor.info.default_tok_params`, which is `add_special_tokens=True`
(`multimodal/processing/context.py:321-332`). Harmless here, but worth knowing for other
models whose tokenizers *do* auto-add BOS.

### EOS / stop tokens — applied in both paths

`generation_config.json` lists `eos_token_id: [1, 106, 50]` while `config.json` lists
`[1, 106]` — an extra stop id. This is applied via
`SamplingParams.update_from_generation_config` (`sampling_params.py:627-654`), called from
`v1/engine/input_processor.py:323`, which is **shared** by online and offline. Not a
divergence.

### Gemma4-specific config patching — applied in both paths

`Gemma4Config.verify_and_update_config` (`model_executor/models/config.py:198-245`) is
registered in `MODELS_CONFIG_MAP` for `Gemma4ForConditionalGeneration`. Gemma 4 has
heterogeneous head dims (`head_dim=256`, `global_head_dim=512`), so it forces **FA4 for all
layers** when available, else `TRITON_ATTN` — its docstring explicitly cites avoiding
"mixed backend selection and numerical divergence".

It runs from `VllmConfig.__post_init__` (`config/vllm.py:878`
→ `try_verify_and_update_config`), so it applies to **both** paths identically.

Same for the platform-level patch in `platforms/cuda.py:302-322`, which forces
`disable_chunked_mm_input = True` for prefix-LM multimodal models.

### `--chat-template-content-format` — irrelevant offline

```
Detected the chat template content format to be 'openai'.
```

This concerns how the server parses **incoming JSON message bodies** (OpenAI-style content
parts vs plain strings) before templating. There is no offline equivalent because we build
the message list ourselves. No effect on output.

---

<a name="recommended-changes"></a>

## Recommended changes

### 1. Pass the usage context — highest value, lowest risk

**✅ Implemented 2026-07-28, exactly as below.** See `docs/v1.6_changes.md` addendum §A.

In `src/vision_ingest/vllm_module/async_worker.py` (~line 75):

```python
from vllm.usage.usage_lib import UsageContext

temp = AsyncLLM.from_engine_args(
    AsyncEngineArgs(**engine_args),
    usage_context=UsageContext.OPENAI_API_SERVER,
)
```

Preferred over hardcoding `max_num_batched_tokens: 8192` / `max_num_seqs: 1024` in the yaml,
because it:

- inherits whatever `vllm serve` would pick **on the current hardware** (the table branches
  on GPU memory, A100 vs H100, TPU chip, CPU);
- stays correct across vLLM upgrades if the defaults are retuned;
- keeps the "raise floor to fit one MM item" logic behaving as it does online.

Hardcoded numbers would silently drift from serve on any of those axes.

### 2. Merge `generation_config.json` into sampling params

**✅ Implemented 2026-07-28, with one deliberate deviation from the "stop sending temperature"
suggestion below.** See `docs/v1.6_changes.md` addendum §A/§B for what actually shipped: the
[parity-target decision](#divergence-2) resolved in favor of true `vllm serve` default parity, but
via merging generation_config.json into `sp` on **our** side (both backends) and then still sending
the resolved value online, rather than omitting the field so the server fills it. Same end result
(the value *is* what the server would have computed), simpler than making offline and online take
different code paths, and it's what makes the offline `AsyncLLM` backend match too — the HTTP-layer
merge this section describes only ever helps the online backend on its own.

In `VLLMConfig.get_sampling_params()`, mirror what `get_diff_sampling_param()` does:
read `GenerationConfig.from_pretrained(self.model_name).to_diff_dict()`, take
`temperature / top_p / top_k / min_p / repetition_penalty`, and layer precedence as:

```
explicit yaml value  >  model generation_config.json  >  vLLM neutral default
```

This is the generic fix — it makes every future model match serve automatically, not just
Gemma 4.

Requires the [parity-target decision](#divergence-2) on `temperature` first.

Two related cleanups worth folding in:

- `get_engine_args()` currently does `engine_args.update(model_specific_config["engine_args"])`
  wholesale, so sampling-only keys such as `sampling_temperature` / `sampling_max_tokens`
  would be passed straight into `AsyncEngineArgs(**engine_args)` (TypeError) and translated
  into bogus `--sampling-temperature` serve flags by
  `online_worker._engine_args_to_serve_flags`. The current `gemma_testing/vllm_model.yaml`
  does not set them, so this is latent rather than live — but it is a trap.
  **✅ Fixed** — `get_engine_args()` now explicitly excludes them.
- If true serve parity on `temperature` is chosen, `online_worker._build_chat_payload` must
  **stop sending `temperature`** so the server fills it from the generation config.
  **Not done this way** — see the deviation note above; `_build_chat_payload` still sends
  `temperature` (and now `top_p`/`top_k`/`repetition_penalty` too), just always the
  generation_config-resolved value rather than a hardcoded `0.2`.

### 3. Composite alpha onto white in `load_cv2_pil`

**⏳ Deferred, not implemented.** Explicitly out of scope — see `docs/v1.6_changes.md`'s "Deferred"
list.

Match `rgba_to_rgb(image, (255, 255, 255))`. Either read with `cv2.IMREAD_UNCHANGED` and
composite manually, or use `PIL.Image.open` + `ImageOps.exif_transpose` +
vLLM's own `convert_image_mode` for an exact match.

Only affects VLM stages, not the text-only LLM path.

---

<a name="caveat"></a>

## Honest caveat on "exact same outputs"

Even with all three changes applied, outputs will **not** be bitwise identical to
`vllm serve`.

Per <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>, vLLM's
default kernels are **not batch-invariant**: floating-point reduction order depends on how
many requests happen to be co-batched at each step. Under continuous batching that
composition varies run to run, so identical configuration makes the output *distributions*
match — it does not make individual generations reproducible.

vLLM 0.25.1 does ship the real fix:

- `VLLM_BATCH_INVARIANT=1` (`envs.py:89`, `envs.py:579`)
- implementation: `model_executor/layers/batch_invariant.py`

It costs throughput, but it is the only route to true determinism. Recommended to enable it
**on both sides during A/B comparison**, so that genuine configuration differences can be
distinguished from batching noise. It can then be turned off for production throughput once
parity is confirmed.

---

<a name="appendix"></a>

## Appendix — how vLLM assembles its config

Useful mental model for future debugging. Ordered by when things happen:

1. **`EngineArgs` / `AsyncEngineArgs`** — the raw dataclass of CLI-equivalent flags.
   `vllm serve` populates it from argparse; offline we populate it from `engine_args` in the
   model yaml. *Equivalent by construction, provided the same keys are supplied.*

2. **`EngineArgs.create_engine_config(usage_context)`** — `engine/arg_utils.py:1818`.
   The **only** place `usage_context` is consulted; it reaches
   `_set_default_max_num_seqs_and_batched_tokens_args`, which fills
   `max_num_batched_tokens` / `max_num_seqs` **only if they are still `None`**.
   → [Divergence 1](#divergence-1)

3. **`VllmConfig.__post_init__`** — `config/vllm.py:870+`. Runs
   `try_verify_and_update_config()` (per-architecture patches from `MODELS_CONFIG_MAP`,
   e.g. `Gemma4Config`) and platform patches (`platforms/cuda.py:check_and_update_config`).
   *Identical in both paths.*

4. **Prompt → tokens.**
   - Online: `render_chat` (`renderers/base.py:1034`) → `render_messages` (applies the
     Jinja chat template) → `tokenize_prompts` → `process_for_engine`.
   - Offline: we call `processor.apply_chat_template(..., tokenize=False)` ourselves in
     `vllm_config.py` and hand vLLM a pre-templated string plus `multi_modal_data`.
   *Verified equivalent for this model* — see [ruled out](#ruled-out).

5. **`SamplingParams` construction.**
   - Online: `ChatCompletionRequest.to_sampling_params(max_tokens, default_sampling_params)`
     — per-field merge from `generation_config.json`.
   - Offline: whatever we constructed, verbatim.
   → [Divergence 2](#divergence-2)

6. **`InputProcessor`** — `v1/engine/input_processor.py:315+`. Applies
   `update_from_generation_config` (eos/stop ids) and `update_from_tokenizer` (bad words),
   and defaults `max_tokens` to `max_model_len - prompt_len` when unset.
   *Identical in both paths.*

7. **Media loading** (multimodal only).
   - Online: `ImageMediaIO.load_file` — PIL + `exif_transpose` + white-background RGBA
     compositing.
   - Offline: our `load_cv2_pil` — `cv2.imread`, alpha discarded.
   → [Divergence 3](#divergence-3)

### Quick reference — reproducing the key checks

```bash
VLLM_DIR=/fsxvision_new/srihari.bandarupalli/environments/gemma4_new/lib/python3.12/site-packages/vllm

# the usage-context defaults table
sed -n '2396,2478p' $VLLM_DIR/engine/arg_utils.py

# the silent ENGINE_CONTEXT fallthrough
sed -n '2600,2625p' $VLLM_DIR/engine/arg_utils.py

# AsyncLLM's default usage_context
grep -n "def from_engine_args" -A 12 $VLLM_DIR/v1/engine/async_llm.py

# the per-field generation_config merge (HTTP layer only)
sed -n '588,615p' $VLLM_DIR/entrypoints/openai/chat_completion/protocol.py

# server-side image pipeline
sed -n '73,95p' $VLLM_DIR/multimodal/media/image.py
```
